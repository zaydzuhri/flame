import fla # Potentially used by custom attention implementations
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import gc

# Helper functions to get and set attention implementation for a specific layer.
# These assume a model structure like model.model.layers[i].attn.attn_impl.
# You MAY NEED TO ADJUST these if your model's structure is different.

def get_layer_attn_impl(model, layer_idx):
    """Gets the attention implementation name for a specific layer."""
    try:
        # Example path: model.model.layers[layer_idx].attention.implementation_name
        # Adjust this path based on your model's specific architecture.
        return model.model.layers[layer_idx].attn.attn_impl
    except AttributeError:
        print(f"Warning: Could not get attn_impl for layer {layer_idx}. Path 'model.model.layers[{layer_idx}].attn.attn_impl' not found.")
        return None

def set_layer_attn_impl(model, layer_idx, impl_name):
    """Sets the attention implementation for a specific layer."""
    try:
        # Example path: model.model.layers[layer_idx].attention.implementation_name = impl_name
        # Adjust this path based on your model's specific architecture.
        model.model.layers[layer_idx].attn.attn_impl = impl_name
        # print(f"Layer {layer_idx} attn_impl set to {impl_name}") # Verbose
    except AttributeError:
        print(f"Warning: Could not set attn_impl for layer {layer_idx} to {impl_name}. Path 'model.model.layers[{layer_idx}].attn.attn_impl' not found.")
    except Exception as e:
        print(f"Error setting attn_impl for layer {layer_idx} to {impl_name}: {e}")


def calculate_sink_rate_for_single_layer(attention_map_single_layer, epsilon=0.3):
    # Extract attention on first token (BOS) across all heads
    first_token_attention = attention_map_single_layer[:, :, :, 0]  # [batch, heads, seq_len]
    # print("first token attentions", first_token_attention)
    
    # Calculate mean attention on first token across sequence length
    mean_first_token_attention = first_token_attention.mean(dim=-1)  # [batch, heads]
    # print("mean first token attentions", mean_first_token_attention)
    
    # Apply indicator function - whether mean attention > epsilon
    indicator = (mean_first_token_attention > epsilon).float()  # [batch, heads]
    # print("indicator", indicator)
    
    # Average across heads
    batch_sink_rates = indicator.mean(dim=(1))  # [batch]
    
    return batch_sink_rates.mean().item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")

    print(f"Loading config for model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    
    # This ensures that the model's forward pass can return attentions if requested.
    config.output_attentions = True 

    # Determine if attention switching will be performed
    do_switching = args.attn_impl_slow and args.attn_impl_fa and (args.attn_impl_slow != args.attn_impl_fa)

    if do_switching:
        print(f"Attention switching enabled: FA='{args.attn_impl_fa}', SLOW='{args.attn_impl_slow}'")
        # Set the initial config to FA if switching, so model loads with FA by default.
        if hasattr(config, 'attn_impl'):
            print(f"Setting initial config.attn_impl to '{args.attn_impl_fa}' for model loading.")
            config.attn_impl = args.attn_impl_fa
        else:
            print(f"Warning: Config for {args.model_name} does not have 'attn_impl'. Cannot set initial FA via config. Relies on manual per-layer setting post-load.")
    elif args.attn_impl_slow:
        print(f"Attention switching NOT active. Attempting to use '{args.attn_impl_slow}' if model config supports it.")
        if hasattr(config, 'attn_impl'):
            config.attn_impl = args.attn_impl_slow
        else:
            print(f"Warning: Config for {args.model_name} does not have 'attn_impl'. Model will use its default attention mechanism.")
    else: # No switching, no specific slow impl -> use model default
         print("Attention switching NOT active. Model will use its default attention mechanism.")


    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code,
        # device_map="auto" # Consider if model is very large and needs sharding
    )
    model.to(device)
    model.eval()

    num_layers = 0
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'): # Common for GPT-like models
        num_layers = len(model.model.layers)
    else:
        print("Attempting to infer number of layers from a dummy forward pass...")
        try:
            dummy_input_text = tokenizer.decode(tokenizer.encode("hello world", add_special_tokens=False)[:10]) # Short sample
            if not dummy_input_text: dummy_input_text = "hello" # Fallback if decode is empty
            dummy_encodings = tokenizer(dummy_input_text, return_tensors="pt", max_length=16, truncation=True).to(device)
            with torch.no_grad():
                dummy_outputs = model(**dummy_encodings, output_attentions=True)
            if hasattr(dummy_outputs, 'attentions') and dummy_outputs.attentions is not None:
                num_layers = len(dummy_outputs.attentions)
            else:
                raise ValueError("Could not determine number of layers from dummy forward pass attentions.")
            del dummy_outputs, dummy_encodings
            if device.type == 'cuda': torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during dummy forward pass for layer count: {e}")
            print("Please ensure your model structure allows layer detection or manually set num_layers.")
            return
    
    if num_layers == 0:
        print("Error: Number of layers is 0 or could not be determined. Cannot proceed.")
        return
    print(f"Model has {num_layers} layers.")

    # If switching, ensure all layers are set to FA after model loading.
    if do_switching:
        print(f"Ensuring all layers are set to FA implementation: '{args.attn_impl_fa}' post-load.")
        for i in range(num_layers):
            set_layer_attn_impl(model, i, args.attn_impl_fa)

    print(f"Loading dataset: {args.dataset_name} (config: {args.dataset_config_name}, split: {args.split})")
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split, streaming=args.streaming_dataset)
    
    all_layers_sink_rates_accumulated = [0.0] * num_layers
    samples_processed_count = 0
    
    actual_n_samples = args.n_samples
    if args.streaming_dataset:
        print(f"Processing {args.n_samples} samples from a streaming dataset...")
        dataset_iterable = dataset.take(args.n_samples)
    else:
        if args.n_samples < 0 or args.n_samples > len(dataset):
            actual_n_samples = len(dataset)
        print(f"Processing {actual_n_samples} samples from a static dataset...")
        dataset_iterable = dataset.select(range(actual_n_samples))

    # Batching logic
    batch_iterator = []
    current_batch_texts = []
    for sample_idx, sample in enumerate(dataset_iterable):
        if args.streaming_dataset and sample_idx >= actual_n_samples:
            break # Ensure we don't exceed n_samples for streaming
        current_batch_texts.append(sample["text"])
        if len(current_batch_texts) == args.batch_size:
            batch_iterator.append(list(current_batch_texts)) # Append a copy
            current_batch_texts.clear()
    if current_batch_texts: # Add last partial batch
        batch_iterator.append(list(current_batch_texts))

    if not batch_iterator:
        print("No batches to process. Check dataset and n_samples.")
        return

    # Main processing loop
    for batch_texts in tqdm(batch_iterator, desc="Processing Batches"):
        if not batch_texts: continue
        
        current_batch_size = len(batch_texts)
        if current_batch_size == 0: continue

        try:
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            ).to(device)
        except Exception as e:
            print(f"Error during tokenization for a batch: {e}. Skipping batch.")
            print(f"Problematic texts (first few chars): {[text[:50] for text in batch_texts]}")
            continue


        for layer_idx in range(num_layers):
            if do_switching:
                set_layer_attn_impl(model, layer_idx, args.attn_impl_slow) # Set current layer to SLOW

            with torch.no_grad():
                outputs = model(**encodings, output_attentions=True) 
            
            layer_attention_map = None
            if outputs.attentions is not None and len(outputs.attentions) > layer_idx and outputs.attentions[layer_idx] is not None:
                layer_attention_map = outputs.attentions[layer_idx].detach() # Detach from graph
            else:
                # This can happen if the 'FA' mode (even for other layers) or 'SLOW' mode for current layer
                # doesn't produce an attention tensor, or if output_attentions=False was somehow forced.
                print(f"Warning: No attention map found for layer {layer_idx} in batch. Outputs.attentions length: {len(outputs.attentions) if outputs.attentions else 'None'}.")


            batch_sink_rate_for_layer = calculate_sink_rate_for_single_layer(
                layer_attention_map, args.epsilon
            )
            
            all_layers_sink_rates_accumulated[layer_idx] += batch_sink_rate_for_layer * current_batch_size

            if do_switching:
                set_layer_attn_impl(model, layer_idx, args.attn_impl_fa) # Revert current layer to FA
            
            # Modest cleanup within layer loop
            del layer_attention_map 
            del outputs 
            if device.type == 'cuda': torch.cuda.empty_cache()
        
        # After processing all layers for this batch
        samples_processed_count += current_batch_size
        del encodings
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()

    # Final averaging
    final_per_layer_sink_rates = [0.0] * num_layers
    if samples_processed_count > 0:
        for i in range(num_layers):
            final_per_layer_sink_rates[i] = all_layers_sink_rates_accumulated[i] / samples_processed_count
    else:
        print("Warning: No samples were processed successfully. Sink rates will be 0.")

    print("\n--- Per-Layer Sink Rates ---")
    for i, rate in enumerate(final_per_layer_sink_rates):
        print(f"Layer {i:02d} (ε={args.epsilon}): {rate:.4f}")
    
    overall_sink_rate = sum(final_per_layer_sink_rates) / num_layers if num_layers > 0 and samples_processed_count > 0 else 0.0
    print(f"\nOverall Average Sink Rate (ε={args.epsilon}): {overall_sink_rate:.4f}")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.dataset_name} (Config: {args.dataset_config_name}, Split: {args.split})\n")
            f.write(f"Samples Processed: {samples_processed_count} / Requested: {args.n_samples}\n")
            f.write(f"Epsilon: {args.epsilon}\n")
            f.write(f"Max Length: {args.max_length}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Device: {device}\n")
            if args.attn_impl_fa: f.write(f"Attn Impl (Fast): {args.attn_impl_fa}\n")
            if args.attn_impl_slow: f.write(f"Attn Impl (Slow/Output): {args.attn_impl_slow}\n")
            f.write(f"Switching Active: {do_switching}\n")
            f.write("\n--- Per-Layer Sink Rates ---\n")
            for i, rate in enumerate(final_per_layer_sink_rates):
                f.write(f"Layer {i:02d}: {rate:.4f}\n")
            f.write(f"\nOverall Average Sink Rate: {overall_sink_rate:.4f}\n")
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Sink Rate Per Layer for Transformer Models")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Huggingface model name")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for AutoClass (needed for some custom models)")
    
    parser.add_argument("--attn_impl_fa", type=str, default="parallel_softpick_attn", 
                        help="Name of the 'fast' attention implementation (e.g., 'flash_attn', 'triton_softpick_attn'). Used for switching.")
    parser.add_argument("--attn_impl_slow", type=str, default="naive_softpick_attn",
                        help="Name of the 'slow' attention implementation that outputs attentions (e.g., 'naive_softpick_attn', 'sdpa', 'eager'). Used for switching or as default.")

    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Huggingface dataset name (e.g., wikitext, c4)")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="Specific config for dataset (e.g., 'wikitext-2-raw-v1', 'en' for c4)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test, validation, train)")
    parser.add_argument("--streaming_dataset", action="store_true", help="Load dataset in streaming mode (for very large datasets)")

    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to process. -1 for all samples in non-streaming mode.")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for processing. Effective batch size for memory is this * num_heads * seq_len^2 for one layer's attention.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Threshold for sink rate calculation")
    parser.add_argument("--output_file", type=str, default="sink_rate_results.txt", help="File to save results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (overrides CUDA availability)")
    
    args = parser.parse_args()
    main(args)
