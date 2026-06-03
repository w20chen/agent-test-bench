# Recording Figure Scripts

These scripts read existing `attempt_*/recordings` artifacts and generate the
first three method-driving figures:

- Plot 1: pairwise iteration distance matrices for attention and MoE.
- Plot 2: layer specialization maps for attention roles and MoE experts.
- Plot 3: attention-vs-MoE layer specialization scatter.

Example:

```bash
conda run -n ML python scripts/recoding_figures/make_figures.py \
  traces/terminal-bench/Qwen_Qwen3-Coder-30B-A3B-Instruct/<run>/<task>/attempt_1 \
  --output-dir docs/recording_figures/<run>/<task>
```

The directory name intentionally follows the user-requested spelling
`recoding_figures`.
