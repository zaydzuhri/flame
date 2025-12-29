import fla # Potentially used by custom attention implementations
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import gc

# Helper functions to get and set attention implementation (assumed to be correct by user)
# You MAY NEED TO ADJUST these if your model's structure is different.
def get_layer_attn_impl(model, layer_idx):
    try:
        return model.model.layers[layer_idx].attn.attn_impl
    except AttributeError:
        # print(f"Warning: Could not get attn_impl for layer {layer_idx}. Path 'model.model.layers[{layer_idx}].attn.attn_impl' not found.")
        return None

def set_layer_attn_impl(model, layer_idx, impl_name):
    try:
        model.model.layers[layer_idx].attn.attn_impl = impl_name
    except AttributeError:
        # print(f"Warning: Could not set attn_impl for layer {layer_idx} to {impl_name}. Path 'model.model.layers[{layer_idx}].attn.attn_impl' not found.")
        pass # Keep it less verbose for successful cases
    except Exception as e:
        print(f"Error setting attn_impl for layer {layer_idx} to {impl_name}: {e}")


# --- NEW FUNCTION TO CALCULATE COMPONENTS FROM LOWER TRIANGLE ---
def calculate_lower_triangle_components(attention_map_single_layer, epsilon=1e-5):
    """
    Calculates non-zero count and element count ONLY for the lower triangle 
    (including the diagonal) of an attention map.
    The upper triangle is ignored as it's assumed to be zero for causal attention.

    Args:
        attention_map_single_layer (torch.Tensor): The attention map, typically
                                                   (batch_size, num_heads, seq_len, seq_len).
                                                   Can be None or empty.
        epsilon (float): Threshold for considering a value as non-zero.

    Returns:
        tuple: (nonzero_count_in_lower_triangles, num_elements_in_lower_triangles)
               Both are torch tensors on the same device as input or CPU if input is None.
    """
    if attention_map_single_layer is None or attention_map_single_layer.numel() == 0:
        # Determine device for return tensors even if input is None
        device_for_empty = 'cpu'
        if attention_map_single_layer is not None: # it means numel is 0 but tensor exists
             device_for_empty = attention_map_single_layer.device
        return torch.tensor(0, device=device_for_empty, dtype=torch.long), \
               torch.tensor(0, device=device_for_empty, dtype=torch.long)

    seq_len = attention_map_single_layer.size(-1)
    
    # Create a lower triangular mask for a single N_seq x N_seq matrix
    tril_mask_single = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_map_single_layer.device)
    )
    
    # Expand mask to match attention map dimensions
    if attention_map_single_layer.ndim == 4: # B, H, N_seq, N_seq
        expanded_mask = tril_mask_single.unsqueeze(0).unsqueeze(0)
    elif attention_map_single_layer.ndim == 3: # B, N_seq, N_seq
        expanded_mask = tril_mask_single.unsqueeze(0)
    elif attention_map_single_layer.ndim == 2: # N_seq, N_seq
        expanded_mask = tril_mask_single
    else:
        # Fallback for unexpected dimensions: process as a flat list of values
        # This is unlikely for standard attention maps but provides robustness.
        print(f"Warning: Unsupported attention map ndim: {attention_map_single_layer.ndim}. Processing all elements.")
        all_values = attention_map_single_layer.flatten()
        nonzero_count = torch.count_nonzero(torch.abs(all_values) > epsilon)
        num_elements = all_values.numel()
        return nonzero_count, torch.tensor(num_elements, device=nonzero_count.device, dtype=torch.long)

    # Select elements in the lower triangle(s) using the mask.
    # torch.masked_select flattens the output, containing only the selected elements.
    lower_triangle_values = torch.masked_select(attention_map_single_layer, expanded_mask)
    
    nonzero_count = torch.count_nonzero(torch.abs(lower_triangle_values) > epsilon)
    num_elements = lower_triangle_values.numel() # This is the count of elements in all lower triangles
    
    return nonzero_count, torch.tensor(num_elements, device=nonzero_count.device, dtype=torch.long)


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
    config.output_attentions = True 

    do_switching = args.attn_impl_slow and args.attn_impl_fa and (args.attn_impl_slow != args.attn_impl_fa)

    if do_switching:
        print(f"Attention switching enabled: FA='{args.attn_impl_fa}', SLOW='{args.attn_impl_slow}'")
        if hasattr(config, 'attn_impl'):
            print(f"Setting initial config.attn_impl to '{args.attn_impl_fa}' for model loading.")
            config.attn_impl = args.attn_impl_fa
        else:
            print(f"Warning: Config for {args.model_name} does not have 'attn_impl'. Relies on manual per-layer setting post-load.")
    elif args.attn_impl_slow: # Only slow is specified, use it as default if possible
        print(f"Attention switching NOT active. Model will use '{args.attn_impl_slow}' if config supports it.")
        if hasattr(config, 'attn_impl'):
            config.attn_impl = args.attn_impl_slow
        else:
            print(f"Warning: Config for {args.model_name} does not have 'attn_impl'. Model will use its default attention mechanism for generating attentions.")
    else: # No switching, no specific slow impl -> use model default
        print("Attention switching NOT active. Model will use its default attention mechanism.")


    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    num_layers = 0
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        print("Attempting to infer number of layers from a dummy forward pass...")
        try:
            dummy_input_text = tokenizer.decode(tokenizer.encode("hello world", add_special_tokens=False)[:10]) 
            if not dummy_input_text: dummy_input_text = "hello" 
            dummy_encodings = tokenizer(dummy_input_text, return_tensors="pt", max_length=16, truncation=True, padding=True).to(device)
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

    if do_switching:
        print(f"Ensuring all layers are set to FA implementation: '{args.attn_impl_fa}' post-load.")
        for i in range(num_layers):
            set_layer_attn_impl(model, i, args.attn_impl_fa)

    print(f"Loading dataset: {args.dataset_name} (config: {args.dataset_config_name}, split: {args.split})")
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split, streaming=args.streaming_dataset)
    
    all_sample_sparsities = [] 
    samples_processed_actual = 0
    
    actual_n_samples = args.n_samples
    if args.streaming_dataset:
        print(f"Processing {args.n_samples} samples from a streaming dataset...")
        dataset_iterable = dataset.take(args.n_samples)
    else:
        if args.n_samples < 0 or args.n_samples > len(dataset):
            actual_n_samples = len(dataset)
        print(f"Processing {actual_n_samples} samples from a static dataset...")
        dataset_iterable = dataset.select(range(actual_n_samples))

    batch_iterator_texts = []
    current_batch_texts = []
    for sample_idx, sample in enumerate(dataset_iterable):
        if args.streaming_dataset and sample_idx >= actual_n_samples: 
            break
        # Attempt to get text, being robust to different possible key names
        text_content = sample.get("text", sample.get("content", None)) 
        if text_content is None: # Try other common keys if primary ones fail
            for key in sample.keys(): # Fallback to first string field if common ones missing
                if isinstance(sample[key], str):
                    text_content = sample[key]
                    break
        
        if not isinstance(text_content, str) or not text_content.strip():
            print(f"Warning: Sample {sample_idx} has no valid 'text' field or it's empty. Skipping sample.")
            continue
        current_batch_texts.append(text_content)

        if len(current_batch_texts) == args.batch_size:
            batch_iterator_texts.append(list(current_batch_texts))
            current_batch_texts.clear()
    if current_batch_texts: # Add last partial batch
        batch_iterator_texts.append(list(current_batch_texts))

    if not batch_iterator_texts:
        print("No valid batches to process. Check dataset, n_samples, and text content.")
        return

    # --- Main processing loop with MODIFIED AGGREGATION ---
    for batch_texts in tqdm(batch_iterator_texts, desc="Processing Batches"):
        if not batch_texts: continue
        
        current_batch_actual_size = len(batch_texts)
        if current_batch_actual_size == 0: continue

        try:
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True, 
                truncation=True, 
                max_length=args.max_length 
            ).to(device)
        except Exception as e:
            print(f"Error during tokenization for a batch: {e}. Skipping batch.")
            print(f"Problematic texts (first few chars): {[text[:50] for text in batch_texts]}")
            continue
        
        current_batch_total_nonzero = torch.tensor(0, device=device, dtype=torch.long)
        current_batch_total_elements = torch.tensor(0, device=device, dtype=torch.long)

        for layer_idx in range(num_layers):
            if do_switching:
                set_layer_attn_impl(model, layer_idx, args.attn_impl_slow)

            with torch.no_grad():
                outputs = model(**encodings, output_attentions=True)
            
            layer_attention_map = None
            if outputs.attentions is not None and len(outputs.attentions) > layer_idx and outputs.attentions[layer_idx] is not None:
                layer_attention_map = outputs.attentions[layer_idx].detach()
            else:
                print(f"Warning: No attention map found for layer {layer_idx} in current batch.")
            
            # Use the new function for lower-triangle components
            nonzero_in_lower, elements_in_lower = \
                calculate_lower_triangle_components(layer_attention_map, args.epsilon)
            
            current_batch_total_nonzero += nonzero_in_lower
            current_batch_total_elements += elements_in_lower

            if do_switching:
                set_layer_attn_impl(model, layer_idx, args.attn_impl_fa)
            
            del layer_attention_map, outputs # Modest cleanup
            if device.type == 'cuda': torch.cuda.empty_cache()
        
        if current_batch_total_elements.item() > 0:
            batch_sparsity = 1.0 - (current_batch_total_nonzero.float() / current_batch_total_elements.float())
            all_sample_sparsities.append(batch_sparsity.item())
        else:
            print(f"Warning: No elements processed for batch. First text: {batch_texts[0][:50] if batch_texts else 'N/A'}")
            all_sample_sparsities.append(float('nan')) # Record NaN to keep counts aligned if needed, will be filtered

        del encodings # After processing all layers for this batch
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        samples_processed_actual += current_batch_actual_size


    # --- After processing all batches ---
    if all_sample_sparsities:
        valid_sparsities = [s for s in all_sample_sparsities if not np.isnan(s)] # Filter out NaNs
        if valid_sparsities:
            average_model_sparsity = np.mean(valid_sparsities)
            min_model_sparsity = np.min(valid_sparsities)
            max_model_sparsity = np.max(valid_sparsities)
            std_model_sparsity = np.std(valid_sparsities)

            print(f"\n===== AGGREGATED MODEL SPARSITY STATISTICS (LOWER TRIANGLE ONLY) =====")
            print(f"Samples/Batches contributing to sparsity stats: {len(valid_sparsities)}")
            print(f"Total Individual Samples Tokenized: {samples_processed_actual}")
            print(f"Average model sparsity across samples/batches: {average_model_sparsity:.2%}")
            print(f"Minimum model sparsity: {min_model_sparsity:.2%}")
            print(f"Maximum model sparsity: {max_model_sparsity:.2%}")
            print(f"Standard deviation: {std_model_sparsity:.2%}")

            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(f"Model: {args.model_name}\n")
                    f.write(f"Dataset: {args.dataset_name} (Config: {args.dataset_config_name}, Split: {args.split})\n")
                    f.write(f"Samples/Batches Processed (for sparsity stats): {len(valid_sparsities)}\n")
                    f.write(f"Total Individual Samples Tokenized: {samples_processed_actual} (Requested: {args.n_samples}, Max: {actual_n_samples})\n")
                    f.write(f"Epsilon: {args.epsilon}\n")
                    f.write(f"Max Length for Tokenizer: {args.max_length}\n")
                    f.write(f"Batch Size: {args.batch_size}\n")
                    f.write(f"Device: {device}\n")
                    if args.attn_impl_fa: f.write(f"Attn Impl (Fast): {args.attn_impl_fa}\n")
                    if args.attn_impl_slow: f.write(f"Attn Impl (Slow/Output): {args.attn_impl_slow}\n")
                    f.write(f"Switching Active: {do_switching}\n")
                    f.write(f"\n--- Sparsity calculated on LOWER TRIANGLE of attention maps --- \n")
                    f.write(f"Average Sparsity Rate: {average_model_sparsity:.4%}\n")
                    f.write(f"Min Sparsity: {min_model_sparsity:.4%}\n")
                    f.write(f"Max Sparsity: {max_model_sparsity:.4%}\n")
                    f.write(f"Std Dev Sparsity: {std_model_sparsity:.4%}\n")
                print(f"Results saved to {args.output_file}")
        else:
            print("No valid sparsity values were recorded (all samples might have resulted in NaN or no elements).")
    else:
        print("No samples were processed successfully to calculate sparsity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Sparsity Per Layer for Transformer Models (Lower Triangle)")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Huggingface model name")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for AutoClass")
    
    parser.add_argument("--attn_impl_fa", type=str, default=None, 
                        help="Name of 'fast' attention (e.g., 'flash_attn'). For switching.")
    parser.add_argument("--attn_impl_slow", type=str, default=None,
                        help="Name of 'slow' attention that outputs attentions (e.g., 'sdpa', 'eager'). For switching or as default.")

    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Huggingface dataset name")
    parser.add_argument("--dataset_config_name", type=str, default=None, 
                        help="Specific config for dataset (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g., test, validation)")
    parser.add_argument("--streaming_dataset", action="store_true", help="Load dataset in streaming mode")

    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples. -1 for all in non-streaming.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenizer.")
    parser.add_argument("--epsilon", type=float, default=1e-5, help="Threshold for sparsity calculation.")
    parser.add_argument("--output_file", type=str, default="lower_triangle_sparsity_results.txt", help="File to save results.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage.")
    
    args = parser.parse_args()
    
    if (args.attn_impl_fa or args.attn_impl_slow) and not (args.attn_impl_fa and args.attn_impl_slow):
        if args.attn_impl_fa and not args.attn_impl_slow:
            print("Warning: --attn_impl_fa specified but --attn_impl_slow is not. Switching will not occur. Will attempt to use model's default for attention output if --attn_impl_fa cannot produce attentions.")
        if not args.attn_impl_fa and args.attn_impl_slow:
             print("Warning: --attn_impl_slow specified but --attn_impl_fa is not. Switching will not occur. Will use --attn_impl_slow if supported.")
    elif not args.attn_impl_fa and not args.attn_impl_slow:
         print("No specific attention implementations provided. Model will use its defaults. Attention output relies on config.output_attentions=True and model support.")


    main(args)