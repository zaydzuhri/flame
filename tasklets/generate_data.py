import os
import csv
import argparse
import random
from typing import List, Dict, Tuple, Callable
from tqdm import tqdm

def make_int_seq(length: int, vocab: List[int]) -> List[int]:
    # return [random.choice(vocab) for _ in range(length)]
    # ensure the random choices are done WITHOUT replacement
    if length > len(vocab):
        raise ValueError(f"Length exceeds vocabulary size for unique sampling: {length} > {len(vocab)}")
    return random.sample(vocab, length)

def make_choices_int_seq(length: int, vocab: List[int]) -> List[int]:
    # return [random.choice(vocab) for _ in range(length)]
    return random.choices(vocab, k=length)

def make_recall_seq(seq_len: int, vocab_size: int, unique_pairs: bool = False) -> List[int]:
    # make sure to have two different vocabularies for key and value
    key_vocab_size = vocab_size // 2
    s: List[int] = []
    if unique_pairs:
        keys = make_int_seq(seq_len, list(range(1, key_vocab_size + 1)))
        values = make_int_seq(seq_len, list(range(key_vocab_size + 1, vocab_size + 1)))
        # interleave keys and values
        for k, v in zip(keys, values):
            s.append(k)
            s.append(v)
    else:
        key_vocab = list(range(1, key_vocab_size + 1))
        val_vocab = list(range(key_vocab_size + 1, vocab_size + 1))
        kv_map = {}
        s = []
        for _ in range(seq_len // 2):
            k = random.choice(key_vocab)
            if k not in kv_map:
                kv_map[k] = random.choice(val_vocab)
            v = kv_map[k]
            s.append(k)
            s.append(v)
    return s

# def make_recall_seq(seq_len: int, vocab_size: int) -> List[int]:
#     # make sure to have two different vocabularies for key and value
#     keys = make_int_seq(seq_len, list(range(1, vocab_size + 1)))
#     values = make_choices_int_seq(seq_len, list(range(1, vocab_size + 1)))
#     # interleave keys and values
#     s: List[int] = []
#     for k, v in zip(keys, values):
#         s.append(k)
#         s.append(v)
#     return s

# def make_fuzzy_recall_seq(seq_len: int, vocab_size: int, window_size: int, unique_pairs: bool = False) -> List[int]:
#     # make sure to have two different vocabularies for key and value
#     key_vocab_size = vocab_size // 2 
#     val_vocab_size = vocab_size - key_vocab_size
#     key_nums = list(range(1, key_vocab_size + 1))
#     val_nums = list(range(key_vocab_size + 1, vocab_size + 1))
#     # create random blobs of maximum length within both key and value vocab, then interleave them
#     key_blobs = []
#     for _ in range(key_vocab_size):
#         blob_len = random.randint(1, window_size)
#         blob = random.sample(key_nums, min(blob_len, len(key_nums)))
#         key_blobs.append(blob)
#     val_blobs = []
#     for _ in range(val_vocab_size):
#         blob_len = random.randint(1, window_size)
#         blob = random.sample(val_nums, min(blob_len, len(val_nums)))
#         val_blobs.append(blob)
#     # interleave keys and values
#     s: List[int] = []
#     if unique_pairs:
#         keys = make_int_seq(seq_len, key_blobs)
#         values = make_int_seq(seq_len, val_blobs)
#         for k, v in zip(keys, values):
#             s.append(k)
#             s.append(v)
#     else:
#         kv_map = {}
#         for _ in range(seq_len // 2):
#             k = random.choice(key_blobs)
#             k_tuple = tuple(k)
#             if k_tuple not in kv_map:
#                 v = random.choice(val_blobs)
#                 kv_map[k_tuple] = v
#             else:
#                 v = kv_map[k_tuple]
#             s.append(k)
#             s.append(v)
#     return s

# make key_blobs and val_blobs global so we can reuse them across samples for the fuzzy recall task, otherwise we can't guarantee the same key blob will have the same value blob across samples which breaks the task
def make_fuzzy_recall_seq(seq_len: int, vocab_size: int, window_size: int, key_blobs: List[List[int]], val_blobs: List[List[int]], unique_pairs: bool = False) -> List[int]:
    # interleave keys and values
    s: List[int] = []
    if unique_pairs:
        keys = make_int_seq(seq_len, key_blobs)
        values = make_int_seq(seq_len, val_blobs)
        for k, v in zip(keys, values):
            s.append(k)
            s.append(v)
    else:
        kv_map = {}
        for _ in range(seq_len // 2):
            k = random.choice(key_blobs)
            k_tuple = tuple(k)
            if k_tuple not in kv_map:
                v = random.choice(val_blobs)
                kv_map[k_tuple] = v
            else:
                v = kv_map[k_tuple]
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
    # output:
    # x: 1 2 3 4 5 6 | 3 4
    # y: _ _ _ _ _ _ _ _ 4
    # make sure to have two different vocabularies for key and value
    s = make_recall_seq(seq_len, vocab_size)
    # choose a query not in the last position and at an even index (to pick a key)
    pos = random.choice(range(0, len(s) - 2, 2))
    q = s[pos]
    v = s[pos + 1]
    x = to_str(s) + ["|", str(q), str(v)]
    y = ["_"] * (len(s) + 2) + [str(v)]
    return {"x": x, "y": y}


def gen_multi_query_recall_sample(seq_len: int, vocab_size: int, num_queries: int) -> Dict:
    # output:
    # x: 1 2 3 4 5 6 | 3 4 1 2
    # y: _ _ _ _ _ _ _ _ 4 _ 2
    # make sure to have two different vocabularies for key and value
    s = make_recall_seq(seq_len, vocab_size)
    # sample distinct positions at an even index, avoid last index for successor
    positions = sorted(random.sample(range(0, seq_len - 1, 2), num_queries))
    queries = [s[p] for p in positions]
    x = to_str(s) + ["|"]
    # queries separated by 'm'
    for i, q in enumerate(queries):
        x.append(str(q))
        x.append("?")
    # total outputs = prefix length + delimiter '|' + 2 tokens per query
    pad_len = len(s) + 1 + 2 * len(queries)
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

# def gen_fuzzy_recall_sample(seq_len: int, vocab_size: int, window_size: int) -> Dict:
#     # output:
#     # x: 1 2 3 4 5 6 | 2 3 4 5
#     # y: _ _ _ _ _ _ _ _ _ _ 5
#     s = make_fuzzy_recall_seq(seq_len, vocab_size, window_size) # this will be a list of lists (blobs)
#     # do something similar like the previous recall sample function but then flatten at the end
#     pos = random.choice(range(0, len(s) - 1, 2)) # choose a random position at an even index, but we can only guarantee the successor if it's not in the last blob
#     q = s[pos]
#     v = s[pos + 1]
#     s_flat = []
#     for blob in s:
#         s_flat.extend(blob)
#     x = s_flat + ["|"] + q + v
#     y = ["_"] * (len(s_flat) + 1 + len(q)) + v
#     # then we have to make them all the same length
#     # the maximum possible length is if all blobs are of maximum size, which is window_size * number of blobs + 1 for the delimiter + 2 * the maximum blob size for the query and value
#     max_len = window_size * len(s) + 1 + 2 * window_size
#     x = x + ["_"] * (max_len - len(x))
#     y = y + ["_"] * (max_len - len(y))
#     return {"x": x, "y": y}

# make the key blobs and value blobs global
def gen_fuzzy_recall_sample(seq_len: int, vocab_size: int, window_size: int, key_blobs: List[List[int]], val_blobs: List[List[int]]) -> Dict:
    # output:
    # x: 1 2 3 4 5 6 | 2 3 4 5
    # y: _ _ _ _ _ _ _ _ _ _ 5
    s = make_fuzzy_recall_seq(seq_len, vocab_size, window_size, key_blobs, val_blobs) # this will be a list of lists (blobs)
    # do something similar like the previous recall sample function but then flatten at the end
    pos = random.choice(range(0, len(s) - 1, 2)) # choose a random position at an even index, but we can only guarantee the successor if it's not in the last blob
    q = s[pos]
    v = s[pos + 1]
    s_flat = []
    for blob in s:
        s_flat.extend(blob)
    x = s_flat + ["|"] + q + v
    y = ["_"] * (len(s_flat) + 1 + len(q)) + v
    # then we have to make them all the same length
    # the maximum possible length is if all blobs are of maximum size, which is window_size * number of blobs + 1 for the delimiter + 2 * the maximum blob size for the query and value
    max_len = window_size * len(s) + 1 + 2 * window_size
    x = x + ["_"] * (max_len - len(x))
    y = y + ["_"] * (max_len - len(y))
    return {"x": x, "y": y}

def gen_noisy_recall_sample(seq_len: int, vocab_size: int, num_queries: int, noise_prob: float) -> Dict:
    # output:
    # x: 1 2 8 3 4 5 6 | 3 4 7 1 2
    # y: _ _ _ _ _ _ _ _ _ 4 _ _ 2
    # base sequence with occasional noise token
    # make sure to have two different vocabularies for key and value AND noise
    # make noise token vocab size scale with seq_len and noise_prob (but less than 1/3 of total vocab)
    noise_seq_len = int(seq_len * noise_prob)
    signal_seq_len = seq_len - noise_seq_len
    noise_vocab_size = min(vocab_size // 3, noise_seq_len)
    seq_vocab_size = vocab_size - noise_vocab_size
    noise_vocab_start = seq_vocab_size + 1
    s = make_recall_seq(signal_seq_len, seq_vocab_size)
    # insert noise_seq_len noise tokens at random positions in s
    noise_prefix = s.copy()
    for _ in range(noise_seq_len):
        noise_token = random.randint(noise_vocab_start, vocab_size)
        insert_pos = random.randint(0, len(s))
        noise_prefix.insert(insert_pos, noise_token)
    # choose query positions on even indices on the original sequence to define the labels
    positions = sorted(random.sample(range(0, len(s) - 1, 2), num_queries))
    queries = [s[p] for p in positions]
    x = to_str(noise_prefix) + ["|"]
    for i, q in enumerate(queries):
        x.append(str(q))
        x.append("?")

    # total outputs = prefix length + delimiter '|' + 2 tokens per query
    pad_len = len(noise_prefix) + 1 + 2 * len(queries)
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

# === Copy tasks ===

def gen_full_copy_sample(seq_len: int, vocab_size: int) -> Dict:
    # output:
    # x: 1 2 3 4 5 6 | 1 2 3 4 5 6
    # y: _ _ _ _ _ _ _ 1 2 3 4 5 6
    s = make_int_seq(seq_len, list(range(1, vocab_size + 1)))
    x = to_str(s) + ["|"] + to_str(s)  # context + partial copy
    y = ["_"] * (len(s) + 1) + to_str(s)
    return {"x": x, "y": y}

def gen_reverse_copy_sample(seq_len: int, vocab_size: int) -> Dict:
    s = make_int_seq(seq_len, list(range(1, vocab_size + 1)))
    rev = list(reversed(s))
    x = to_str(s) + ["|"] + to_str(rev)  # context + partial copy
    y = ["_"] * (len(s) + 1) + to_str(rev)
    return {"x": x, "y": y}

def gen_selective_copy_sample(seq_len: int, vocab_size: int, noise_prob: float) -> Dict:
    # output:
    # x: 1 2 n 3 4 n 5 6 | 1 2 3 4 5 6
    # y: _ _ _ _ _ _ _ _ _ 1 2 3 4 5 6
    # just make usual copy sample then insert noise tokens "n"
    signal_seq_len = int(seq_len * (1 - noise_prob))
    noise_seq_len = seq_len - signal_seq_len
    s = make_int_seq(signal_seq_len, list(range(1, vocab_size + 1)))
    noise_prefix = s.copy()
    for _ in range(noise_seq_len):
        insert_pos = random.randint(0, len(noise_prefix))
        noise_prefix.insert(insert_pos, "n")
    x = to_str(noise_prefix) + ["|"] + to_str(s)  # context + partial copy
    y = ["_"] * (len(noise_prefix) + 1) + to_str(s)
    return {"x": x, "y": y}

# === Memorization ===

def make_memorization_table(
    num_keys: int,
    key_len: int,
    vocab_size: int,
) -> Dict[Tuple[int, ...], int]:
    # create a list of num_keys unique keys, each key is a list of tokens of length key_len
    # to make sure the keys are unique, we can just incrementally generate them from the vocab
    vocab = list(range(1, vocab_size + 1))
    # doing it cheaply by keeping track of indices in the vocab for each position in the key
    key_indices = [0] * key_len
    keys = []
    for _ in range(num_keys):
        key = [vocab[i] for i in key_indices]
        keys.append(key)
        # increment the key indices, last one first like counting
        if key_indices[-1] < vocab_size - 1:
            key_indices[-1] += 1
        else:            # need to carry over
            for j in range(key_len - 1, -1, -1):
                if key_indices[j] < vocab_size - 1:
                    key_indices[j] += 1
                    break
                else:
                    key_indices[j] = 0
    # assign random values to each key, doesn't matter if they overlap
    table = {}    
    for key in keys:
        value = random.randint(1, vocab_size)
        table[tuple(key)] = value
    return table

def make_memorization_sample_fns(
    seq_len: int,
    key_len: int,
    table: Dict[Tuple[int, ...], int]
) -> Tuple[Callable[[], Dict], Callable[[], Dict]]:
    # output:
    # x: 1 2 3 | 3 4 5 | 6 7 8 |
    # y: _ _ 3 _ _ _ 5 _ _ _ 8 _
    # to create a sample, just get a number of key value pairs from the table and concatenate them with "|" in between
    num_pairs = seq_len // key_len + 2
    x = []
    y = []
    for _ in range(num_pairs):
        key = random.choice(list(table.keys()))
        value = table[key]
        x.extend(key)
        y.extend(["_"] * len(key))
        x.append(value)
        y.append(value)
        x.append("|")
        y.append("_")
    return {"x": x, "y": y}

# === Counting (count occurrences of a query token) ===

def gen_counting_sample(seq_len: int, vocab_size: int, max_count: int) -> Dict:
    # output:
    # x: 3 4 4 1 3 4 | 4 3
    # y: _ _ _ _ _ _ _ _ 3
    count = random.randint(1, min(seq_len - 1, vocab_size - 1, max_count))
    query_token = random.randint(1, vocab_size)
    # make sure the query token appears count times in the sequence
    vocab_other = [i for i in range(1, vocab_size + 1) if i != query_token]
    seq = make_choices_int_seq(seq_len - count, vocab_other)
    for _ in range(count):
        insert_pos = random.randint(0, len(seq))
        seq.insert(insert_pos, query_token)
    x = to_str(seq) + ["|", str(query_token), str(count)]
    y = ["_"] * (len(seq) + 2) + [str(count)]
    return {"x": x, "y": y}

# === Sorting (sort tokens by count) ===

def gen_sorting_sample(seq_len: int, vocab_size: int, num_unique_toks: int) -> Dict:
    # output:
    # x: 1 2 1 3 1 0 3 1 3 2 | 1 3 2 0
    # y: _ _ _ _ _ _ _ _ _ _ _ 1 3 2 0
    assert num_unique_toks <= seq_len, "Need at least 1 position per token"

    # Pick distinct tokens
    toks = random.sample(range(1, vocab_size + 1), num_unique_toks)

    # Pick strictly unique counts that sum to seq_len
    # Step 1: sample distinct positive integers
    counts = random.sample(range(1, seq_len + 1), num_unique_toks)

    # Step 2: rescale to sum to seq_len
    total = sum(counts)
    counts = [max(1, c * seq_len // total) for c in counts]

    # Fix rounding drift so total == seq_len
    diff = seq_len - sum(counts)
    counts[0] += diff

    # Ensure uniqueness (simple retry if scaling broke it)
    if len(set(counts)) != len(counts):
        return gen_sorting_sample(seq_len, vocab_size, num_unique_toks)

    # Build sequence
    seq = []
    for tok, count in zip(toks, counts):
        seq.extend([tok] * count)

    random.shuffle(seq)

    count_dict = dict(zip(toks, counts))
    sorted_toks = sorted(toks, key=lambda t: count_dict[t], reverse=True)

    x = to_str(seq) + ["|"] + to_str(sorted_toks)
    y = ["_"] * (len(seq) + 1) + to_str(sorted_toks)

    return {"x": x, "y": y}

# === State-tracking tasks ===

def gen_single_stack_ops_sample(seq_len: int, vocab_size: int) -> Dict:
    # output:
    # x: i 2 i 1 o 1 o 2 i 4 o 4
    # y: _ _ _ __  1 _ 2 _ _ _ 4
    x = []
    y = []
    stack = []
    for _ in range(seq_len//2):
        # sample operation (but only allow pop if stack is not empty)
        if len(stack) == 0:
            op = "i"
        else:
            op = random.choice(["i", "o"])
        if op == "i":
            token = random.randint(1, vocab_size)
            stack.append(token)
            x.extend(["i", str(token)])
            y.extend(["_", "_"])
        else:
            token = stack.pop()
            x.extend(["o", str(token)])
            y.extend(["_", str(token)])
    return {"x": x, "y": y}

def gen_multi_stack_ops_sample(seq_len: int, vocab_size: int, num_stacks: int) -> Dict:
    # output:
    # x: i 0 2 i 1 1 o 0 2 o 1 1 i 0 4 o 0 4
    # y: _ _ _ _ _ _ _ _ 2 _ _ 1 _ _ _ _ _ 4
    x = []
    y = []
    stacks = [[] for _ in range(num_stacks)]
    for _ in range(seq_len//3):
        # sample stack and operation (but only allow pop if stack is not empty)
        stack_idx = random.randint(0, num_stacks - 1)
        if len(stacks[stack_idx]) == 0:
            op = "i"
        else:
            op = random.choice(["i", "o"])
        if op == "i":
            token = random.randint(1, vocab_size)
            stacks[stack_idx].append(token)
            x.extend(["i", str(stack_idx), str(token)])
            y.extend(["_", "_", "_"])
        else:
            token = stacks[stack_idx].pop()
            x.extend(["o", str(stack_idx), str(token)])
            y.extend(["_", "_", str(token)])
    return {"x": x, "y": y}

def gen_flip_flop_language_sample(seq_len: int, vocab_size: int) -> Dict:
    # output:
    # x: w 0 i 1 r 0 i 0 w 1 r 1
    # y: _ _ _ _ _ 0 _ _ _ _ _ 1
    x = []
    y = []
    state = None
    for i in range(seq_len//2):
        op_choices = ["w"] if state is None else ["r", "i", "i"]
        # make sure that the last op is "r", so we also need to make sure there is a state written at the second to last step
        if i == seq_len//2 - 1 and state is not None:
            op = "r"
        elif i == seq_len//2 - 2 and state is None:
            op = "w"
        elif i == seq_len//2 - 2 and state is not None:
            op = "i"
        else:
            op = random.choice(op_choices)
        token = random.randint(1, vocab_size)
        if op == "w":
            x.extend([op, str(token)])
            y.extend(["_", "_"])
            state = token
        elif op == "r":
            x.extend([op, str(state)])
            y.extend(["_", str(state)])
            state = None
        else:
            x.extend([op, str(token)])
            y.extend(["_", "_"])
    return {"x": x, "y": y}

# === Formal language tasks ===

def gen_valid_dyck(seq_len: int) -> List[str]:
    """Generate a valid Dyck sequence of exact length seq_len (must be even)."""
    assert seq_len % 2 == 0, "Valid Dyck sequences must have even length"

    stack = 0
    seq = []

    for i in range(seq_len):
        remaining = seq_len - i

        # If we must close to finish properly
        if stack == remaining:
            seq.append(")")
            stack -= 1
        elif stack == 0:
            seq.append("(")
            stack += 1
        else:
            # Randomly choose open or close
            if random.random() < 0.5:
                seq.append("(")
                stack += 1
            else:
                seq.append(")")
                stack -= 1

    return seq


def gen_invalid_dyck(seq_len: int) -> List[str]:
    """Generate an invalid Dyck sequence."""
    # Start from a valid one, then corrupt it
    if seq_len % 2 != 0:
        # Odd length is automatically invalid
        return [random.choice(["(", ")"]) for _ in range(seq_len)]

    seq = gen_valid_dyck(seq_len)

    # Introduce an error
    error_type = random.choice(["flip", "prefix_break", "extra_open"])

    if error_type == "flip":
        # Flip one bracket
        i = random.randrange(seq_len)
        seq[i] = "(" if seq[i] == ")" else ")"

    elif error_type == "prefix_break":
        # Force invalid prefix (more closing than opening early)
        seq[0] = ")"

    elif error_type == "extra_open":
        # Replace a closing with opening → imbalance
        closes = [i for i, c in enumerate(seq) if c == ")"]
        if closes:
            i = random.choice(closes)
            seq[i] = "("

    return seq

def gen_dyck_language_sample(seq_len: int, vocab_size: int) -> Dict:
    # output: 
    # x: ( ( ) ) ( ) | 1 
    # y: _ _ _ _ _ _ _ 1 
    # we want an equal number of correct vs incorrect samples so sample the label first
    label = random.choice([0, 1])

    if label == 1:
        # Ensure valid sequence
        if seq_len % 2 != 0:
            seq_len += 1  # fix to even
        x = gen_valid_dyck(seq_len)
    else:
        x = gen_invalid_dyck(seq_len)

    y = ["_"] * (len(x) + 1) + [str(label)]
    x = x + ["|", str(label)]

    return {"x": x, "y": y}

def gen_anbncn_language_sample(seq_len: int, vocab_size: int) -> Dict:
    # output:
    # x: a a b b c c | 1
    # y: _ _ _ _ _ _ _ 1
    # we want an equal number of correct vs incorrect samples
    # so sample the label first
    label = random.choice([0, 1])
    if label == 1:
        n = random.randint(1, seq_len // 3)
        x = ["a"] * n + ["b"] * n + ["c"] * n
        # pad the left side
        x = ["_"] * (seq_len - len(x)) + x
        x = x + ["|", str(label)]
        y = ["_"] * (len(x) - 1) + [str(label)]
    else:
        an = random.randint(1, seq_len // 3)
        bn = random.randint(1, seq_len // 3)
        cn = random.randint(1, seq_len // 3)
        # make sure it's not the case that an == bn == cn
        while an == bn == cn:
            an = random.randint(1, seq_len // 3)
            bn = random.randint(1, seq_len // 3)
            cn = random.randint(1, seq_len // 3)
        x = ["a"] * an + ["b"] * bn + ["c"] * cn
        # pad the left side
        x = ["_"] * (seq_len - len(x)) + x
        x = x + ["|", str(label)]
        y = ["_"] * (len(x) - 1) + [str(label)]
    return {"x": x, "y": y}

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
                            "single_stack_ops",
                            "multi_stack_ops",
                            "flip_flop",
                            "dyck_language",
                            "anbncn_language",
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
    parser.add_argument("--num-pairs", type=int, default=4,
                        help="Number of key-value pairs / pairs in some tasks.")
    parser.add_argument("--num-keys", type=int, default=100,
                        help="Number of unique keys for memorization task.")
    parser.add_argument("--key-len", type=int, default=3, 
                        help="Length of each key sequence for memorization task.")
    parser.add_argument("--max-count", type=int, default=10,
                        help="Maximum count for counting task.")
    parser.add_argument("--num-unique-toks", type=int, default=5,
                        help="Number of unique tokens for sorting task.")
    parser.add_argument("--num-stacks", type=int, default=2,
                        help="Number of stacks for the multi-stack operations task.")
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
        # create random blobs of maximum length within both key and value vocab, then interleave them
        # make sure to have two different vocabularies for key and value
        key_vocab_size = args.vocab_size // 2 
        val_vocab_size = args.vocab_size - key_vocab_size
        # key_nums = list(range(1, key_vocab_size + 1))
        # val_nums = list(range(key_vocab_size + 1, args.vocab_size + 1))
        key_nums = list(range(1, args.vocab_size + 1))
        val_nums = list(range(1, args.vocab_size + 1))
        key_blobs = []
        for _ in range(key_vocab_size):
            blob_len = random.randint(1, args.window_size)
            blob = random.sample(key_nums, min(blob_len, len(key_nums)))
            key_blobs.append(blob)
        val_blobs = []
        for _ in range(val_vocab_size):
            blob_len = random.randint(1, args.window_size)
            blob = random.sample(val_nums, min(blob_len, len(val_nums)))
            val_blobs.append(blob)
        train_sample_fn = lambda: gen_fuzzy_recall_sample(args.seq_len, args.vocab_size, args.window_size, key_blobs, val_blobs)
        test_sample_fn = lambda: gen_fuzzy_recall_sample(args.seq_len, args.vocab_size, args.window_size, key_blobs, val_blobs)
    elif args.task == "noisy_recall":
        train_sample_fn = lambda: gen_noisy_recall_sample(
            args.seq_len, args.vocab_size, args.num_queries, args.noise_prob
        )
        test_sample_fn = lambda: gen_noisy_recall_sample(
            args.seq_len, args.vocab_size, args.num_queries, args.noise_prob
        )
    elif args.task == "full_copy":
        train_sample_fn = lambda: gen_full_copy_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_full_copy_sample(args.seq_len, args.vocab_size)
    elif args.task == "reverse_copy":
        train_sample_fn = lambda: gen_reverse_copy_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_reverse_copy_sample(args.seq_len, args.vocab_size)
    elif args.task == "selective_copy":
        train_sample_fn = lambda: gen_selective_copy_sample(args.seq_len, args.vocab_size, args.noise_prob)
        test_sample_fn = lambda: gen_selective_copy_sample(args.seq_len, args.vocab_size, args.noise_prob)
    elif args.task == "memorization":
        table = make_memorization_table(args.num_keys, args.key_len, args.vocab_size)
        train_sample_fn = lambda: make_memorization_sample_fns(args.seq_len, args.key_len, table)
        test_sample_fn = lambda: make_memorization_sample_fns(args.seq_len, args.key_len, table)
    elif args.task == "counting":
        train_sample_fn = lambda: gen_counting_sample(args.seq_len, args.vocab_size, args.max_count)
        test_sample_fn = lambda: gen_counting_sample(args.seq_len, args.vocab_size, args.max_count)
    elif args.task == "sorting":
        train_sample_fn = lambda: gen_sorting_sample(args.seq_len, args.vocab_size, args.num_unique_toks)
        test_sample_fn = lambda: gen_sorting_sample(args.seq_len, args.vocab_size, args.num_unique_toks)
    elif args.task == "single_stack_ops":
        train_sample_fn = lambda: gen_single_stack_ops_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_single_stack_ops_sample(args.seq_len, args.vocab_size)
    elif args.task == "multi_stack_ops":
        train_sample_fn = lambda: gen_multi_stack_ops_sample(args.seq_len, args.vocab_size, args.num_stacks)
        test_sample_fn = lambda: gen_multi_stack_ops_sample(args.seq_len, args.vocab_size, args.num_stacks)
    elif args.task == "flip_flop":
        train_sample_fn = lambda: gen_flip_flop_language_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_flip_flop_language_sample(args.seq_len, args.vocab_size)
    elif args.task == "dyck_language":
        train_sample_fn = lambda: gen_dyck_language_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_dyck_language_sample(args.seq_len, args.vocab_size)
    elif args.task == "anbncn_language":
        train_sample_fn = lambda: gen_anbncn_language_sample(args.seq_len, args.vocab_size)
        test_sample_fn = lambda: gen_anbncn_language_sample(args.seq_len, args.vocab_size)
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
