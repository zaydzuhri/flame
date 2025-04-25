import fla
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def calculate_sink_rate(attention_maps, epsilon=0.3):
    """
    Calculate sink rate using the formula:
    sink_rate = (1/(L*H))*sum_L_H(1_((1/T)*sum_T(a_l_h_1_t) > epsilon))
    
    Where:
    - L is the number of layers
    - H is the number of attention heads
    - T is the sequence length
    - 1_() is the indicator function
    - a_ is the attention score at that index
    - epsilon is the threshold (default: 0.3)
    
    Args:
        attention_maps: Attention maps from the model as a list with length L of tensors with shape [batch, heads, seq_len, seq_len]
        epsilon: Threshold for attention
        
    Returns:
        sink_rate: The calculated sink rate
    """
    sink_rate = 0
    for i, attention in enumerate(attention_maps):
        # Extract attention on first token (BOS) across all heads
        first_token_attention = attention[:, :, :, 0]  # [batch, heads, seq_len]
        # print("first token attentions", first_token_attention)
        
        # Calculate mean attention on first token across sequence length
        mean_first_token_attention = first_token_attention.mean(dim=-1)  # [batch, heads]
        # print("mean first token attentions", mean_first_token_attention)
        
        # Apply indicator function - whether mean attention > epsilon
        indicator = (mean_first_token_attention > epsilon).float()  # [batch, heads]
        # print("indicator", indicator)
        
        # Average across heads
        batch_sink_rates = indicator.mean(dim=(1))  # [batch]
        
        # Average across batch
        sink_rate += batch_sink_rates.mean().item()

    # Normalize by number of layers
    num_layers = len(attention_maps)
    sink_rate /= num_layers
        
    return sink_rate

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, return_dict_in_generate=True, output_attentions=True).half()
    model.to(device)
    model.eval()
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # Take n_samples from the dataset
    if args.n_samples < len(dataset):
        dataset = dataset.select(range(args.n_samples))
    else:
        args.n_samples = len(dataset)
    
    print(f"Processing {args.n_samples} samples...")
    
    # Process samples
    sink_rate = 0
    
    for i in tqdm(range(0, args.n_samples, args.batch_size)):
        batch_samples = dataset[i:min(i + args.batch_size, args.n_samples)]
        
        # Tokenize
        encodings = tokenizer(
            batch_samples["text"], 
            padding=True, 
            truncation=True, 
            max_length=args.max_length, 
            return_tensors="pt"
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**encodings)
        
        # Calculate sink rate for this batch
        batch_sink_rate = calculate_sink_rate(outputs.attentions, args.epsilon)
        attention_maps = None  # Free memory
        sink_rate += batch_sink_rate
    
    # Average sink rate
    sink_rate /= args.n_samples
    
    print(f"Sink Rate (ε={args.epsilon}): {sink_rate:.4f}")
    
    # Optional: Save sink rate results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Samples: {args.n_samples}\n")
            f.write(f"Epsilon: {args.epsilon}\n")
            f.write(f"Sink Rate: {sink_rate:.4f}\n")
            
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Sink Rate for Transformer Models")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Huggingface model name")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Huggingface dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Threshold for sink rate calculation")
    parser.add_argument("--output_file", type=str, default="", help="File to save results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)