Overwrote `README.md` with a project-specific, code- and artifact-grounded writeup based on:

- `artifacts/results.json` (exact metrics + per-utterance refs/hyps + bootstrap deltas)
- `artifacts/report.md` (the rendered results tables and timestamps)
- Implemented pipeline in `src/streaming_asr/*` plus `configs/smoke.yaml` and `configs/openslr_full.yaml`

It includes: problem statement, dataset provenance (OpenSLR resource IDs + exact URLs used in code), methodology (features/tokenizer/models/streaming mask/augmentation/decoding), baseline + ablation grid, exact results table references (and the numbers from the current artifacts), repro commands, limitations, and concrete next research steps.