"""
Calculates acceptance rate (LCP) and ranking quality (NDCG) for language
models with speculative decoding heads (TOP or MTP).

The evaluation compares the sequence proposed by the special head against the
sequence greedily verified by the standard NTP head.
"""
import fla
import argparse
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For full reproducibility, you might also need these, but they can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def calculate_lcp_from_sequences(predicted_sequences, verified_sequences):
    """Calculates LCP and second-token-match rate by comparing two token sequences."""
    batch_size, eval_length = predicted_sequences.size()
    if eval_length == 0: return 0, 0
    if eval_length == 1: return batch_size, 0

    second_token_matches_tensor = (predicted_sequences[:, 1] == verified_sequences[:, 1])
    second_token_matches = torch.sum(second_token_matches_tensor).item()

    tail_matches = (predicted_sequences[:, 1:] == verified_sequences[:, 1:])
    prefix_matches = torch.cumprod(tail_matches.int(), dim=1)
    tail_lcp_lengths = torch.sum(prefix_matches, dim=1)
    total_lcp = torch.sum(1 + tail_lcp_lengths).item()
    return total_lcp, second_token_matches

def calculate_ndcg_from_sequences(predicted_sequences, verified_sequences):
    """Calculates NDCG based on position-wise matches between two sequences."""
    k = predicted_sequences.size(1)
    if k == 0: return 0.0

    gains = (predicted_sequences == verified_sequences).float()
    discounts = torch.log2(torch.arange(k, device=gains.device).float() + 2.0)
    dcg = torch.sum(gains / discounts, dim=1)
    ideal_gains, _ = torch.sort(gains, dim=1, descending=True)
    idcg = torch.sum(ideal_gains / discounts, dim=1)
    ndcg = dcg / (idcg + 1e-8)
    ndcg[idcg == 0] = 0.0
    return torch.sum(ndcg).item()

def main(args):
    """Main function to run the metric calculations."""
    if args.seed is not None:
        print(f"Setting random seed to {args.seed} for reproducibility.")
        set_random_seed(args.seed)

    print(f"Starting model evaluation for model_type='{args.model_type}'...")
    print(f"Arguments: {vars(args)}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("⚠️ Warning: No CUDA device found. Running on CPU will be very slow.")

    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir
    ).to(torch.bfloat16).to(device)
    model.eval()

    print(f"Loading dataset: {args.dataset_name} (Subset: {args.dataset_subset}, Split: {args.dataset_split})...")
    dataset = load_dataset(
        args.dataset_name, name=args.dataset_subset, split=args.dataset_split,
        cache_dir=args.cache_dir, streaming=True
    )

    total_samples_processed = 0
    total_lcp_length, total_second_token_matches = 0, 0
    total_ndcg_scores = {k: 0.0 for k in args.ndcg_k}
    max_k_needed = max(args.ndcg_k + [args.lcp_eval_length])

    dataset_iterator = iter(dataset)
    pbar_total = args.num_samples if args.num_samples != -1 else None

    with tqdm(total=pbar_total, desc="Processing samples") as pbar:
        while True:
            if args.num_samples != -1 and total_samples_processed >= args.num_samples: break
            
            batch_input_ids = []
            num_to_fetch = min(args.batch_size, (args.num_samples - total_samples_processed) if args.num_samples != -1 else args.batch_size)
            
            while len(batch_input_ids) < num_to_fetch:
                try:
                    sample = next(dataset_iterator)
                    text = sample[args.text_column]
                    if not text or not isinstance(text, str): continue
                    tokens = tokenizer.encode(text)
                    if len(tokens) >= args.context_length:
                        start = random.randint(0, len(tokens) - args.context_length)
                        cropped_tokens = tokens[start : start + args.context_length]
                        batch_input_ids.append(torch.tensor(cropped_tokens))
                except StopIteration: break
            
            if not batch_input_ids: break

            input_ids = torch.stack(batch_input_ids).to(device)
            
            with torch.no_grad():
                # Step 1: Get the proposed sequence based on the model type
                input_dict = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
                
                if args.model_type == 'top':
                    _, top_logits = model(**input_dict, output_top_logits=True).logits
                    top_sequences_full = torch.argsort(top_logits[:, -1, :], dim=-1, descending=True)[:, :max_k_needed]
                
                elif args.model_type == 'mtp':
                    logits = model(**input_dict, output_mtp_logits=True).logits
                    num_heads = logits.size(2)
                    if max_k_needed > num_heads:
                        print(f"Warning: Requested max evaluation length ({max_k_needed}) is greater than the MTP model's head count ({num_heads}). Clamping to {num_heads}.")
                        max_k_needed = num_heads
                    top_sequences_full = torch.argmax(logits, dim=-1)[:, -1, :max_k_needed]

                elif args.model_type == 'dsmtp':
                    logits = model(**input_dict, output_dsmtp_logits=True).logits.transpose(1, 2)
                    num_heads = logits.size(2)
                    if max_k_needed > num_heads:
                        print(f"Warning: Requested max evaluation length ({max_k_needed}) is greater than the MTP model's head count ({num_heads}). Clamping to {num_heads}.")
                        max_k_needed = num_heads
                    top_sequences_full = torch.argmax(logits, dim=-1)[:, -1, :max_k_needed]
                
                # Step 2: Run verification pass to get the ground-truth greedy sequence
                verify_input_ids = torch.cat([input_ids, top_sequences_full], dim=1)
                verify_dict = {'input_ids': verify_input_ids, 'attention_mask': torch.ones_like(verify_input_ids)}
                verify_ntp_logits = model(**verify_dict).logits
                verify_output_sequence = torch.argmax(verify_ntp_logits, dim=-1)
                verified_tokens_full = verify_output_sequence[:, args.context_length-1:-1]
                
                # --- Metric Calculations ---
                lcp_eval_len = min(args.lcp_eval_length, max_k_needed)
                predicted_lcp = top_sequences_full[:, :lcp_eval_len]
                verified_lcp = verified_tokens_full[:, :lcp_eval_len]
                lcp_sum, second_token_sum = calculate_lcp_from_sequences(predicted_lcp, verified_lcp)

                for k in args.ndcg_k:
                    if k > max_k_needed: continue
                    predicted_ndcg = top_sequences_full[:, :k]
                    verified_ndcg = verified_tokens_full[:, :k]
                    ndcg_sum_k = calculate_ndcg_from_sequences(predicted_ndcg, verified_ndcg)
                    total_ndcg_scores[k] += ndcg_sum_k

            # --- Update statistics ---
            current_batch_size = input_ids.size(0)
            total_lcp_length += lcp_sum
            total_second_token_matches += second_token_sum
            total_samples_processed += current_batch_size
            pbar.update(current_batch_size)
            
            if total_samples_processed > 0:
                avg_lcp = total_lcp_length / total_samples_processed
                ndcg_descs = [f"NDCG@{k}: {(total_ndcg_scores[k] / total_samples_processed):.3f}" for k in sorted(args.ndcg_k) if k <= max_k_needed]
                pbar.set_description(f"Avg LCP: {avg_lcp:.2f} | " + " | ".join(ndcg_descs))

    print("\n--- Calculation Complete ---")
    if total_samples_processed == 0: print("No valid samples were processed."); return

    print(f"Total samples processed: {total_samples_processed}")
    print("\n--- LCP Metrics ---")
    print(f"Average Acceptance Rate (LCP): {(total_lcp_length / total_samples_processed):.4f}")
    print(f"Second Token Accuracy Rate: {(total_second_token_matches / total_samples_processed):.4%}")
    
    print("\n--- NDCG Metrics (Sequence Match) ---")
    for k in sorted(args.ndcg_k):
        if k <= max_k_needed:
            final_avg_ndcg = total_ndcg_scores[k] / total_samples_processed
            print(f"Average NDCG@{k}: {final_avg_ndcg:.4f}")
    print("-------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LCP and NDCG for custom HF language models.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name.")
    parser.add_argument("--model_type", type=str, default="top", choices=["top", "mtp", "dsmtp"], help="Type of the model head to evaluate (TOP or MTP).")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--dataset_subset", type=str, default=None, help="The subset/configuration of the dataset to use (e.g., 'en.noblocklist').")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the text column in the dataset.")
    parser.add_argument("--num_samples", type=int, default=1024, help="Total samples to process. Set to -1 for the entire dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples to process in a single batch.")
    parser.add_argument("--context_length", type=int, default=512, help="The length of the token sequence to use as context.")
    parser.add_argument("--lcp_eval_length", type=int, default=10, help="The number of candidate tokens to verify for the LCP metric.")
    parser.add_argument("--ndcg_k", type=int, nargs='+', default=[10], help="A list of k-values for which to calculate NDCG@k (e.g., 5 10 20).")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache models and datasets.")
    parser.add_argument("--seed", type=int, default=79, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)