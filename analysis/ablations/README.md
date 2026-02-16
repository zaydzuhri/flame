# Softpick Ablation Scripts

This folder contains inference-time ablations derived from
`resources/ablations/notebooklm_part2.md`:

- **Attention Knockout**: mask attention to the sink token and measure PPL impact.
- **Sum-of-Attention**: measure per-head sum of attention weights (no-op signal).
- **Sink-Logit Variance**: measure variance of attention logits pointing to the sink.
- **Head Shutdown Dynamics**: measure per-head off-rates across checkpoints and track dead heads.
- **Dead-Head Zeroing PPL**: zero checkpoint-specific dead heads pre-o_proj and compare PPL vs baseline.
- **Dead-Head Zeroing LAMBADA**: compare baseline vs dead-head-zeroing exact-match accuracy on LAMBADA.
- **Head Dormant (Mean Sink)**: measure mean sink attention per head for softmax checkpoints.
- **Head Shutdown Parser**: summarize head-dead results into readable counts and deltas.
- **Head Shutdown Plots**: generate ACL-friendly plots from head-dead outputs.
- **Head Dormant (Guo)**: measure per-token sink dominance rates per head.
- **Dormant Threshold Scan**: post-hoc scan of multiple dormant thresholds.
- **Sink Dominance Sweep**: rerun Guo dormant analysis across sink thresholds.

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

`analysis/ablations/head_dead_analysis.py`

- Measures per-head off-rate using pre-output (pre-o_proj) head outputs.
- Iterates checkpoints using a model path template with `{step}`.
- Uses token budget (`--max-tokens`) instead of sample count to keep runs comparable.
- Outputs `dead_heads_by_step` and `dead_*_heads` lists to identify which heads stay dead.

Example:

```bash
python analysis/ablations/head_dead_analysis.py \
  --model-template analysis/attention_sink/hf_models/softpick-340M-4096-step-{step} \
  --steps 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 \
  --dataset DKYoon/SlimPajama-6B \
  --split train \
  --streaming \
  --batch-size 2 \
  --max-length 4096 \
  --max-tokens 5000000 \
  --output analysis/attention_sink/outputs/softpick-340M-head-dead.json
```

`analysis/ablations/dead_head_zeroing_ppl.py`

- Runs baseline and masked perplexity with the same dataset slice.
- Default behavior (`--auto-mask`) derives dead heads from baseline pass
  automatically, then replays the same batches for masked pass.
- If `--mask-json` is provided, the file must exist. Otherwise omit it to keep
  default auto-mask behavior.
- Optional `--mask-json` uses `head_dead_analysis.py` output and applies zeroing
  by setting per-layer head masks directly on attention modules
  (`set_dead_head_mask`).
- JSON parsing and step selection stay in the script layer, not model config.
- Baseline and masked runs share the same tokenized batches in one pipeline,
  including streaming datasets.
- For reproducibility, use the same checkpoint step in `--model` and `--step`.

Example:

```bash
python analysis/ablations/dead_head_zeroing_ppl.py \
  --model analysis/attention_sink/hf_models/softpick-1B-4096-step-100000 \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --n-samples 128 \
  --batch-size 1 \
  --padding none \
  --max-length 4096 \
  --output analysis/attention_sink/outputs/softpick-1B-dead-zeroing-ppl-step-100000.json
```

`analysis/ablations/dead_head_zeroing_lambada.py`

- Runs baseline and masked exact-match accuracy on LAMBADA-style last-word prediction.
- Uses the same external dead-head mask application path as the PPL script.
- Default behavior (`--auto-mask`) derives dead heads from baseline pass
  automatically, then replays the same samples for masked pass.
- Optional `--mask-json` + `--step` uses precomputed dead heads from
  `head_dead_analysis.py`.

Example:

```bash
python analysis/ablations/dead_head_zeroing_lambada.py \
  --model analysis/attention_sink/hf_models/softpick-1B-4096-step-100000 \
  --dataset lambada \
  --dataset-config plain_text \
  --split test \
  --max-length 4096 \
  --output analysis/attention_sink/outputs/softpick-1B-dead-zeroing-lambada-step-100000.json
```

`3rdparty/flash-linear-attention/evals/harness.py` (lm-eval integration)

- Registers `--model fla_deadmask` / `--model fla-deadmask` for lm-eval-harness.
- `fla_deadmask` applies dead-head masking during `loglikelihood` evaluation.
- Supports:
  - Auto-mask from the same evaluation requests (default when no JSON is provided).
  - JSON masks via `dead_head_mask_json` + `dead_head_mask_step`.
- Baseline is unchanged with `--model fla`.

Example (LAMBADA baseline):

```bash
python 3rdparty/flash-linear-attention/evals/harness.py \
  --model fla \
  --tasks lambada_openai \
  --model_args pretrained=analysis/attention_sink/hf_models/softpick-1B-4096-step-100000,trust_remote_code=true,dtype=bfloat16,device=cuda,batch_size=8
```

Example (LAMBADA dead-head masked, auto-mask default):

```bash
python 3rdparty/flash-linear-attention/evals/harness.py \
  --model fla_deadmask \
  --tasks lambada_openai \
  --model_args pretrained=analysis/attention_sink/hf_models/softpick-1B-4096-step-100000,trust_remote_code=true,dtype=bfloat16,device=cuda,batch_size=8,dead_head_auto_mask_eps=1e-2,dead_head_auto_mask_dead_threshold=0.95
```

`analysis/ablations/head_dormant_analysis.py`

- Measures per-head mean attention to the sink token (index 0 by default).
- Flags heads as dormant when the sink attention fraction exceeds `--dormant-threshold`.
- Requires `--attn-impl naive_attn` and `--batch-size 1` with `--padding none` to avoid padding artifacts.

Example:

```bash
python analysis/ablations/head_dormant_analysis.py \
  --model-template analysis/attention_sink/hf_models/softmax-340M-4096-step-{step} \
  --steps 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 \
  --attn-impl naive_attn \
  --dataset DKYoon/SlimPajama-6B \
  --split train \
  --streaming \
  --batch-size 1 \
  --padding none \
  --max-length 4096 \
  --max-tokens 5000000 \
  --output analysis/attention_sink/outputs/softmax-340M-head-dormant.json
```

`analysis/ablations/head_dormant_analysis_guo.py`

- Implements Guo et al. style dormancy: per-token sink-dominant attention plus value-state drain.
- Value-state drain is detected when the sink token value norm is below
  `--value-drain-threshold * mean(other token norms)`.
- Requires `--attn-impl naive_attn` and `--batch-size 1` with `--padding none`.

Example:

```bash
python analysis/ablations/head_dormant_analysis_guo.py \
  --model-template analysis/attention_sink/hf_models/vanilla-340M-4096-step-{step} \
  --steps 10000 \
  --attn-impl naive_attn \
  --dataset DKYoon/SlimPajama-6B \
  --split train \
  --streaming \
  --batch-size 1 \
  --padding none \
  --max-length 2048 \
  --max-tokens 200000 \
  --output analysis/attention_sink/outputs/vanilla-340M-head-dormant-guo-step10000.json
```

`analysis/ablations/parse_head_dead_analysis.py`

- Summarizes head-dead JSON output into counts, per-step deltas, and persistent heads.
- Use `--show-heads` to list specific head indices.

Example:

```bash
python analysis/ablations/parse_head_dead_analysis.py \
  --input analysis/attention_sink/outputs/softpick-340M-head-dead.json \
  --show-heads persistent
```

`analysis/ablations/plot_head_dead_analysis.py`

- Creates ACL 2-column-ready figures for dead-head trends and heatmaps.
- Uses `edd_utils.register_edd_style()` to match the notebook style.
- Set `--use-tex` if you want LaTeX text rendering and your environment supports it.
- Use `--head-bin-size`, `--layer-bin-size`, and `--layer-tick-step` to reduce clutter in the persistent-head map.

Example:

```bash
python analysis/ablations/plot_head_dead_analysis.py \
  --input analysis/attention_sink/outputs/softpick-340M-head-dead.json \
  --output-dir analysis/attention_sink/outputs/figures
```

`analysis/ablations/head_dormant_analysis_guo.py`

- Implements Guo et al.'s dormant-head idea for softmax models.
- A head is dormant for a token if sink attention > `--sink-dominance-threshold`
  (default 0.9) and optional entropy is below `--entropy-threshold`.
- `dormant_rate[step][layer][head]` is the fraction of tokens that are dormant.
- Heads are classified using `--dormant-threshold` (default 0.95) and
  `--mostly-dormant-threshold` (default 0.75).
- Requires `--attn-impl naive_attn` and `--batch-size 1`.

Example:

```bash
python analysis/ablations/head_dormant_analysis_guo.py \
  --model-template analysis/attention_sink/hf_models/softmax-340M-4096-step-{step} \
  --steps 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 \
  --attn-impl naive_attn \
  --dataset DKYoon/SlimPajama-6B \
  --split train \
  --streaming \
  --batch-size 1 \
  --padding none \
  --max-length 4096 \
  --max-tokens 5000000 \
  --output analysis/attention_sink/outputs/softmax-340M-head-dormant-guo.json
```

`analysis/ablations/scan_dormant_thresholds.py`

- Post-hoc scan over multiple dormant thresholds using `dormant_rate`.
- Avoids re-running attention; works directly on the JSON output.

Example:

```bash
python analysis/ablations/scan_dormant_thresholds.py \
  --input analysis/attention_sink/outputs/vanilla-340M-head-dormant-guo-step10000_new.json \
  --thresholds 0.95,0.9,0.8
```

`analysis/ablations/scan_sink_dominance_thresholds.py`

- Runs `head_dormant_analysis_guo.py` across sink-dominance thresholds.
- Defaults to thresholds 0.1..0.9 plus 0.95.

Example:

```bash
python analysis/ablations/scan_sink_dominance_thresholds.py \
  --model-template analysis/attention_sink/hf_models/vanilla-340M-4096-step-{step} \
  --steps 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 \
  --attn-impl naive_attn \
  --dataset DKYoon/SlimPajama-6B \
  --split train \
  --streaming \
  --batch-size 1 \
  --padding none \
  --max-length 2048 \
  --max-tokens 5000000 \
  --output-prefix vanilla-340M-head-dormant-guo \
  --output-dir analysis/attention_sink/outputs
```

## Notes

- Default settings use `--batch-size 1` and `--padding none` to avoid padding
  interactions with naive attention implementations.
- The head dormant analysis requires `--attn-impl naive_attn` to return attention
  weights and will error if attentions are unavailable.
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
- The head dead analysis uses pre-o_proj head outputs, so it does not require
  `--attn-impl naive_*` or attention weights.
