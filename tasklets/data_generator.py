import os
import csv
import argparse
import random
from typing import List, Dict, Tuple, Callable
from tqdm import tqdm

Token = str

def make_int_seq(length: int, vocab: List[int]) -> List[int]:
    # return [random.choice(vocab) for _ in range(length)]
    # ensure the random choices are done WITHOUT replacement
    if length > len(vocab):
        raise ValueError("Length exceeds vocabulary size for unique sampling")
    return random.sample(vocab, length)

def make_recall_seq(seq_len: int, vocab_size: int) -> List[int]:
    # make sure to have two different vocabularies for key and value
    key_vocab_size = vocab_size // 2
    keys = make_int_seq(seq_len, list(range(1, key_vocab_size + 1)))
    values = make_int_seq(seq_len, list(range(key_vocab_size + 1, vocab_size + 1)))
    # interleave keys and values
    s: List[int] = []
    for k, v in zip(keys, values):
        s.append(k)
        s.append(v)
    return s

def to_str(tokens) -> List[str]:
    return [str(t) for t in tokens]


def upload_dataset_to_hf(
    train_path: str,
    test_path: str,
    repo_id: str,
    private: bool,
) -> None:
    from datasets import load_dataset

    dataset_dict = load_dataset(
        "csv",
        data_files={"train": train_path, "test": test_path},
    )
    dataset_dict.push_to_hub(repo_id, private=private)


def serialize_sample(sample: Dict) -> Dict[str, str]:
    serialized_sample = {}
    for key, value in sample.items():
        if isinstance(value, list):
            serialized_sample[key] = " ".join(str(v) for v in value)
        else:
            serialized_sample[key] = str(value)
    return serialized_sample


def write_split(
    path: str,
    num_samples: int,
    sample_fn: Callable[[], Dict],
    desc: str,
) -> None:
    with open(path, "w", newline="") as f:
        writer = None
        expected_keys = None
        for idx in tqdm(range(num_samples), desc=desc, unit="sample"):
            sample = sample_fn()
            serialized_sample = serialize_sample(sample)
            sample_keys = list(serialized_sample.keys())

            if writer is None:
                expected_keys = sample_keys
                writer = csv.DictWriter(f, fieldnames=expected_keys)
                writer.writeheader()
            elif sample_keys != expected_keys:
                raise ValueError(
                    f"Inconsistent sample columns at index {idx}: {sample_keys} != {expected_keys}"
                )

            writer.writerow(serialized_sample)

        if writer is None:
            writer = csv.DictWriter(f, fieldnames=["x", "y"])
            writer.writeheader()


def write_dataset_streaming(
    output_dir: str,
    num_train: int,
    train_sample_fn: Callable[[], Dict],
    num_test: int,
    test_sample_fn: Callable[[], Dict],
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    train_path = f"{output_dir}/train.csv"
    test_path = f"{output_dir}/test.csv"
    write_split(train_path, num_train, train_sample_fn, desc="Generating train")
    write_split(test_path, num_test, test_sample_fn, desc="Generating test")
    return train_path, test_path

# === Recall tasks ===

def gen_single_query_recall_sample(seq_len: int, vocab_size: int) -> Dict:
    # make sure to have two different vocabularies for key and value
    s = make_recall_seq(seq_len, vocab_size)
    # choose a query not in the last position and at an even index (to pick a key)
    pos = random.choice(range(0, len(s) - 2, 2))
    q = s[pos]
    x = to_str(s) + ["|", str(q)]
    y = ["_"] * (len(s) + 1) + [str(s[pos + 1])]
    return {"x": x, "y": y}


def gen_multi_query_recall_sample(seq_len: int, vocab_size: int, num_queries: int) -> Dict:
    # make sure to have two different vocabularies for key and value
    s = make_recall_seq(seq_len, vocab_size)
    # sample distinct positions, avoid last index for successor
    positions = sorted(random.sample(range(seq_len - 1), num_queries))
    queries = [s[p] for p in positions]
    x = to_str(s) + ["|"]
    # queries separated by 'm'
    for i, q in enumerate(queries):
        x.append(str(q))
        if i != len(queries) - 1:
            x.append("?")
    # total outputs = prefix length + delimiter '|' + 2 tokens per query
    pad_len = len(s) + 2 * len(queries)
    y = ["_"] * pad_len

    # fill answer slots at end
    answers = []
    for pos in positions:
        answers.append(str(s[pos + 1]))  # successor
        answers.append("_")

    # remove last underscore
    answers = answers[:-1]
    y[-len(answers):] = answers
    return {"x": x, "y": y}

def gen_fuzzy_recall_sample(seq_len: int, vocab_size: int, window_size: int) -> Dict:
    s = make_recall_seq(seq_len, vocab_size)
    max_start = seq_len - window_size - 1
    if max_start < 0:
        raise ValueError("Sequence too short for given window_size")
    rand_size = random.randint(1, window_size)
    start = random.randint(0, max_start)
    pattern = s[start:start + rand_size]
    x = to_str(s) + ["|"] + to_str(pattern)
    answer = s[start + rand_size]
    y = ["_"] * (len(s) + window_size + 1) + [str(answer)]
    return {"x": x, "y": y}

def gen_noisy_recall_sample(seq_len: int, vocab_size: int, num_queries: int, noise_prob: float) -> Dict:
    # base sequence with occasional noise token
    # make sure to have two different vocabularies for key and value AND noise
    # make noise token vocab size scale with seq_len and noise_prob (but less than 1/3 of total vocab)
    noise_vocab_size = min(vocab_size // 3, int(seq_len * noise_prob))
    seq_vocab_size = vocab_size - noise_vocab_size
    noise_vocab_start = seq_vocab_size + 1
    s = make_recall_seq(seq_len, seq_vocab_size)
    noisy_prefix: List[int] = []
    for v in s:
        noisy_prefix.append(v)
        if random.random() < noise_prob:
            # insert random noise token from same vocab
            noisy_prefix.append(random.randint(noise_vocab_start, vocab_size))
    # choose query positions on the original sequence to define the labels
    positions = sorted(random.sample(range(len(s) - 1), num_queries))
    queries = [s[p] for p in positions]
    x = to_str(noisy_prefix) + ["|"]
    for i, q in enumerate(queries):
        x.append(str(q))
        if i != len(queries) - 1:
            x.append("?")
    answers = []
    for pos in positions:
        answers.append(str(s[pos + 1]))
        answers.append("_")
    answers = answers[:-1]  # remove last underscore

    y = ["_"] * (len(noisy_prefix) + 1 + len(answers))
    y[-len(answers):] = answers
    return {"x": x, "y": y}

# === Copy tasks ===

def gen_full_copy_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    x = to_str(s) + ["|"] + to_str(s[:-1])  # context + partial copy
    y = ["_"] * (len(s) + 1) + to_str(s)
    return {"x": x, "y": y}

def gen_full_copy_test_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    x = to_str(s) + ["|"]
    y = to_str(s)
    # we supervise all outputs
    m = [1] * len(y)
    return {"x": x, "y": y, "m": m}

def gen_reverse_copy_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    rev = list(reversed(s))
    x = to_str(s) + ["|"] + to_str(rev[1:])
    y = to_str(s[1:]) + ["|"] + to_str(rev)
    m = [0] * (len(s)) + [1] * len(s)
    return {"x": x, "y": y, "m": m}

def gen_reverse_copy_test_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    rev = list(reversed(s))
    x = to_str(s) + ["|"]
    y = to_str(rev)
    m = [1] * len(y)
    return {"x": x, "y": y, "m": m}

def gen_selective_copy_sample(seq_len: int, vocab_size: int, prob_n: float) -> Dict:
    seq: List[str] = []
    for _ in range(seq_len):
        if random.random() < prob_n:
            seq.append("n")
        else:
            seq.append(str(random.randint(1, vocab_size)))
    # ensure at least one numeric token
    if not any(t != "n" for t in seq):
        idx = random.randrange(seq_len)
        seq[idx] = "1"
    x = seq + ["|"]
    y = [t for t in seq if t != "n"]
    m = [1] * len(y)
    return {"x": x, "y": y, "m": m}

# === Memorization ===

def gen_memorization_dataset(vocab_size: int, num_pairs: int, num_test: int) -> Tuple[List[Dict], List[Dict]]:
    # create a permutation-based mapping
    keys = random.sample(range(1, vocab_size + 1), num_pairs)
    remaining = [v for v in range(1, vocab_size + 1) if v not in keys]
    if len(remaining) < num_pairs:
        # allow reuse with a simple shift
        values = [(k % vocab_size) + 1 for k in keys]
    else:
        values = random.sample(remaining, num_pairs)
    pairs = list(zip(keys, values))

    # training sample: single big sequence
    train_x: List[str] = []
    for i, (k, v) in enumerate(pairs):
        train_x.extend([str(k), str(v)])
        if i != len(pairs) - 1:
            train_x.append("|")
    train_y: List[str] = []
    # First pair: only value then '|'
    train_y.append(str(pairs[0][1]))
    train_y.append("|")
    for i in range(1, len(pairs)):
        k, v = pairs[i]
        train_y.extend([str(k), str(v), "|"])
    # mask: 1 on numbers, 0 on '|'
    m = [0 if t == "|" else 1 for t in train_y]
    train_samples = [{"x": train_x, "y": train_y, "m": m}]

    # test samples: query keys in random order
    test_samples: List[Dict] = []
    for _ in range(num_test):
        # pick a subset of keys to query
        q_keys = random.sample(keys, min(4, len(keys)))
        x: List[str] = []
        for i, k in enumerate(q_keys):
            x.append(str(k))
            x.append("m")
            if i != len(q_keys) - 1:
                x.append("|")
        y: List[str] = []
        for k in q_keys:
            v = dict(pairs)[k]
            y.extend([str(v), "_", "_"])
        m_y = [1 if t not in ["|", "_"] else 0 for t in y]
        test_samples.append({"x": x, "y": y, "m": m_y})
    return train_samples, test_samples


def make_memorization_sample_fns(
    vocab_size: int,
    num_pairs: int,
) -> Tuple[Callable[[], Dict], Callable[[], Dict]]:
    # create a permutation-based mapping once, then sample queries from it repeatedly
    keys = random.sample(range(1, vocab_size + 1), num_pairs)
    remaining = [v for v in range(1, vocab_size + 1) if v not in keys]
    if len(remaining) < num_pairs:
        values = [(k % vocab_size) + 1 for k in keys]
    else:
        values = random.sample(remaining, num_pairs)
    mapping = dict(zip(keys, values))

    train_x: List[str] = []
    for i, (k, v) in enumerate(mapping.items()):
        train_x.extend([str(k), str(v)])
        if i != len(mapping) - 1:
            train_x.append("|")
    train_y: List[str] = [str(mapping[keys[0]]), "|"]
    for i in range(1, len(keys)):
        k = keys[i]
        v = mapping[k]
        train_y.extend([str(k), str(v), "|"])
    train_m = [0 if t == "|" else 1 for t in train_y]
    fixed_train_sample = {"x": train_x, "y": train_y, "m": train_m}

    def train_sample_fn() -> Dict:
        return fixed_train_sample

    def test_sample_fn() -> Dict:
        q_keys = random.sample(keys, min(4, len(keys)))
        x: List[str] = []
        for i, k in enumerate(q_keys):
            x.append(str(k))
            x.append("m")
            if i != len(q_keys) - 1:
                x.append("|")
        y: List[str] = []
        for k in q_keys:
            y.extend([str(mapping[k]), "_", "_"])
        m_y = [1 if t not in ["|", "_"] else 0 for t in y]
        return {"x": x, "y": y, "m": m_y}

    return train_sample_fn, test_sample_fn

# === Reversal (predecessor lookup) ===

def gen_reversal_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    pos = random.randint(1, seq_len - 1)
    q = s[pos]
    prev_token = s[pos - 1]
    x = to_str(s) + ["|", str(q)]
    y_prefix = to_str(s[1:])
    y = y_prefix + ["|", str(q), str(prev_token)]
    m = [0] * len(y)
    m[-1] = 1
    return {"x": x, "y": y, "m": m}

def gen_reversal_test_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    pos = random.randint(1, seq_len - 1)
    q = s[pos]
    prev_token = s[pos - 1]
    x = to_str(s) + ["|", str(q)]
    y = [str(prev_token)]
    m = [1]
    return {"x": x, "y": y, "m": m}

# === Sorting (sort pairs by letter) ===

def gen_sorting_sample(num_pairs: int) -> Dict:
    letters = [chr(ord("a") + i) for i in range(26)]
    used_letters = random.sample(letters, num_pairs)
    nums = random.sample(range(1, 100), num_pairs)
    pairs = list(zip(nums, used_letters))
    random.shuffle(pairs)
    x: List[str] = []
    for n, ch in pairs:
        x.extend([str(n), ch])
    x.append("|")
    # sort by letter
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    y: List[str] = []
    for n, ch in sorted_pairs:
        y.extend([str(n), ch])
    m = [1] * len(y)
    return {"x": x, "y": y, "m": m}

# === Counting (count occurrences of a query token) ===

def gen_counting_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    q = random.choice(s)
    count_q = s.count(q)
    x = to_str(s) + ["|", str(q)]
    y_prefix = to_str(s[1:])
    y = y_prefix + ["|", str(q), str(count_q)]
    m = [0] * len(y)
    m[-1] = 1
    return {"x": x, "y": y, "m": m}

def gen_counting_test_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, vocab_size)
    q = random.choice(s)
    count_q = s.count(q)
    x = to_str(s) + ["|", str(q)]
    y = [str(count_q)]
    m = [1]
    return {"x": x, "y": y, "m": m}

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate synthetic sequence tasks.")
    parser.add_argument("--task", type=str, required=True,
                        choices=[
                            "single_recall",
                            "multi_recall",
                            "fuzzy_recall",
                            "noisy_recall",
                            "full_copy",
                            "reverse_copy",
                            "selective_copy",
                            "memorization",
                            "reversal",
                            "sorting",
                            "counting",
                        ],
                        help="Which task to generate.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to put train.txt and test.txt. If not set, uses ./data/{task}/")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=6,
                        help="Base sequence length for most tasks.")
    parser.add_argument("--vocab-size", type=int, default=10,
                        help="Numeric vocabulary size (1..vocab_size).")
    parser.add_argument("--seed", type=int, default=13)

    # task-specific knobs
    parser.add_argument("--num-queries", type=int, default=2,
                        help="Number of queries for multi/noisy recall.")
    parser.add_argument("--window-size", type=int, default=3,
                        help="Window size for fuzzy recall.")
    parser.add_argument("--noise-prob", type=float, default=0.2,
                        help="Noise probability for noisy recall.")
    parser.add_argument("--prob-n", type=float, default=0.5,
                        help="Probability of 'n' token in selective copy.")
    parser.add_argument("--num-pairs", type=int, default=4,
                        help="Number of key-value pairs / pairs in some tasks.")
    parser.add_argument("--upload-to-hf", action="store_true",
                        help="Upload generated train/test splits to the Hugging Face Hub.")
    parser.add_argument("--hf-repo-id", type=str, default=None,
                        help="Target dataset repo id on Hugging Face Hub, e.g. username/my-task.")
    parser.add_argument("--hf-private", action="store_true",
                        help="Create/upload as a private Hugging Face dataset repo.")
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    train_sample_fn: Callable[[], Dict]
    test_sample_fn: Callable[[], Dict]

    if args.task == "single_recall":
        train_sample_fn = lambda: gen_single_query_recall_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_single_query_recall_sample(args.seq_len, args.vocab_size)
    elif args.task == "multi_recall":
        train_sample_fn = lambda: gen_multi_query_recall_sample(args.seq_len, args.vocab_size, args.num_queries)
        test_sample_fn = lambda: gen_multi_query_recall_sample(args.seq_len, args.vocab_size, args.num_queries)
    elif args.task == "fuzzy_recall":
        train_sample_fn = lambda: gen_fuzzy_recall_sample(args.seq_len, args.vocab_size, args.window_size)
        test_sample_fn = lambda: gen_fuzzy_recall_sample(args.seq_len, args.vocab_size, args.window_size)
    elif args.task == "noisy_recall":
        train_sample_fn = lambda: gen_noisy_recall_sample(
            args.seq_len, args.vocab_size, args.num_queries, args.noise_prob
        )
        test_sample_fn = lambda: gen_noisy_recall_sample(
            args.seq_len, args.vocab_size, args.num_queries, args.noise_prob
        )
    elif args.task == "full_copy":
        train_sample_fn = lambda: gen_full_copy_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_full_copy_test_sample(args.seq_len, args.vocab_size)
    elif args.task == "reverse_copy":
        train_sample_fn = lambda: gen_reverse_copy_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_reverse_copy_test_sample(args.seq_len, args.vocab_size)
    elif args.task == "selective_copy":
        train_sample_fn = lambda: gen_selective_copy_sample(args.seq_len, args.vocab_size, args.prob_n)
        test_sample_fn = lambda: gen_selective_copy_sample(args.seq_len, args.vocab_size, args.prob_n)
    elif args.task == "memorization":
        train_sample_fn, test_sample_fn = make_memorization_sample_fns(
            vocab_size=args.vocab_size, num_pairs=args.num_pairs
        )
    elif args.task == "reversal":
        train_sample_fn = lambda: gen_reversal_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_reversal_test_sample(args.seq_len, args.vocab_size)
    elif args.task == "sorting":
        train_sample_fn = lambda: gen_sorting_sample(args.num_pairs)
        test_sample_fn = lambda: gen_sorting_sample(args.num_pairs)
    elif args.task == "counting":
        train_sample_fn = lambda: gen_counting_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_counting_test_sample(args.seq_len, args.vocab_size)
    else:
        raise ValueError(f"Unknown task {args.task}")

    output_dir = args.output_dir if args.output_dir else f"./data/{args.task}"
    train_path, test_path = write_dataset_streaming(
        output_dir=output_dir,
        num_train=args.num_train,
        train_sample_fn=train_sample_fn,
        num_test=args.num_test,
        test_sample_fn=test_sample_fn,
    )

    if args.upload_to_hf:
        if not args.hf_repo_id:
            parser.error("--hf-repo-id is required when --upload-to-hf is set.")
        upload_dataset_to_hf(
            train_path=train_path,
            test_path=test_path,
            repo_id=args.hf_repo_id,
            private=args.hf_private,
        )
        print(f"Uploaded dataset to Hugging Face Hub: {args.hf_repo_id}")

if __name__ == "__main__":
    main()
