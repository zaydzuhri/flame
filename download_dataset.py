from datasets import load_dataset

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=64)
