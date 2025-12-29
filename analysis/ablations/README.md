# Softpick Ablation Scripts

This folder contains inference-time ablations derived from
`resources/ablations/notebooklm_part2.md`:

- **Attention Knockout**: mask attention to the sink token and measure PPL impact.
- **Sum-of-Attention**: measure per-head sum of attention weights (no-op signal).
- **Sink-Logit Variance**: measure variance of attention logits pointing to the sink.

## Scripts

`analysis/ablations/attention_knockout.py`

- Runs perplexity once with normal attention and once with a masked sink key.
- Requires a `--attn-impl` that uses a `naive_*` attention path so the mask can
  be injected.
- Use `--mask-key-index 0` to mask the first token.

Example:

```bash
python analysis/ablations/attention_knockout.py \
  --model zaydzuhri/vanilla-340M-4096-model \
  --attn-impl naive_attn \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --n-samples 64 \
  --max-length 512
```

`analysis/ablations/sum_attention.py`

- Reports mean `sum(attention)` per head and per layer.
- Set `--attn-impl` to a `naive_*` attention implementation to ensure attention
  weights are returned.

Example:

```bash
python analysis/ablations/sum_attention.py \
  --model zaydzuhri/softpick-340M-4096-model \
  --attn-impl naive_softpick_attn \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --n-samples 64 \
  --max-length 512
```

`analysis/ablations/sink_logit_variance.py`

- Measures variance of attention logits targeting the sink token per layer.
- Uses the attention projections directly; no changes to the attention path
  required.

Example:

```bash
python analysis/ablations/sink_logit_variance.py \
  --model zaydzuhri/softpick-340M-4096-model \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --n-samples 64 \
  --max-length 512
```

## Notes

- Default settings use `--batch-size 1` and `--padding none` to avoid padding
  interactions with naive attention implementations.
- `--n-samples` counts non-empty text examples; if you request too few samples
  and the dataset starts with empty rows (as in Wikitext), the script will raise
  to prompt a larger sample count.
- When masking `--mask-key-index 0`, the attention knockout run skips the first
  prediction token to avoid the undefined all-masked attention case.
- The attention knockout script raises if masked attention produces non-finite
  values, so errors are surfaced immediately.
- Use `--no-skip-first-query-on-mask` to force masking key 0 for every query;
  this will likely error due to the causal all-masked first row.
- Use `--output` or `--output-dir` to save JSON/JSONL results.
- If you need remote downloads, pass `--no-local-files-only` and
  `--trust-remote-code` as needed.
