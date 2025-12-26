# Attention Sink Analysis

This folder contains a decoupled Python script derived from
`resources/inferencerunpod.ipynb` for running inference and inspecting
attention/hidden-state behavior.

## Scripts

`analysis/attention_sink/inferencerunpod.py` provides:

- text generation for a prompt
- forward pass inspection with hidden states and attentions
- optional histograms and attention heatmaps

`analysis/attention_sink/scripts/download_hf_checkpoint.py` downloads a step
folder from a Hugging Face repo into a local `checkpoint/` directory. Use
`--checkpoint-subdir` when the repo stores checkpoints under a subfolder.

`analysis/attention_sink/scripts/convert_dcp_to_hf.py` converts a DCP
`checkpoint/step-<n>` folder into Hugging Face `save_pretrained` format.

## Usage

Download a DCP step folder from Hugging Face:

```bash
python analysis/attention_sink/scripts/download_hf_checkpoint.py \
  --repo-id zaydzuhri/vanilla-340M-4096-batch16-steps100000-20250409-210858 \
  --step 10000 \
  --allow-pattern config.json \
  --allow-pattern tokenizer.json \
  --allow-pattern tokenizer_config.json \
  --allow-pattern special_tokens_map.json \
  --output-root analysis/attention_sink/runs/vanilla-340M-4096
```

If the repo nests the checkpoint under a subfolder, provide the subdirectory
(the step is still saved under `output-root/checkpoint/step-<n>`):

```bash
python analysis/attention_sink/scripts/download_hf_checkpoint.py \
  --repo-id zaydzuhri/softpick-340M-4096-batch16-steps100000 \
  --checkpoint-subdir checkpoint \
  --step 1 \
  --allow-pattern config.json \
  --allow-pattern tokenizer.json \
  --allow-pattern tokenizer_config.json \
  --allow-pattern special_tokens_map.json \
  --output-root analysis/attention_sink/runs/softpick-340M-4096
```

Convert the DCP checkpoint to Hugging Face format:

```bash
python analysis/attention_sink/scripts/convert_dcp_to_hf.py \
  --checkpoint-root analysis/attention_sink/runs/vanilla-340M-4096 \
  --step 10000 \
  --base-model zaydzuhri/vanilla-340M-4096-model \
  --no-local-files-only \
  --output-dir analysis/attention_sink/hf_models/vanilla-340M-4096-step-10000
```

Generate text:

```bash
python analysis/attention_sink/inferencerunpod.py \
  --model analysis/attention_sink/hf_models/vanilla-340M-4096-step-10000 \
  --mode generate
```

Run a forward pass with attention and hidden-state plots:

```bash
python analysis/attention_sink/inferencerunpod.py \
  --model exp/vanilla.120M.batch4.seqlen2048.context2048.warmup1000.update2.steps5000.lr5e-4.cosine \
  --mode forward \
  --attn-impl naive_attn \
  --plot-attn-grid \
  --plot-hidden-hist \
  --save-dir analysis/attention_sink/outputs
```

Notes:

- Plots are saved under `analysis/attention_sink/outputs` by default.
- Use `--device cpu` if CUDA is unavailable.
- Set `--no-local-files-only` if you need `from_pretrained` to fetch remotely.
- If a model returns no attentions, the script raises an error.
- Use `--attn-impl naive_attn` to force attention outputs when the model defaults
  to `parallel_attn`.
- If you download config/tokenizer files locally, pass their paths via
  `--base-model` or `--config`/`--tokenizer` instead of repo ids.
- `download_checkpoint.py` downloads full repos; the script above supports
  single step folders for DCP runs.
