# Submission Checklist (2026-05-09)

## Required by doScenes Challenge
- `submission.csv` (official columns)
- `ADE_instruction`
- `ADE_baseline`
- `Delta_ADE`
- Public code repository link
- Paper PDF link

## Local Files to Prepare
- Submission CSV:
  - `artifacts/submissions/submission.csv`
- Precheck report:
  - `artifacts/experiments/submission_precheck_report.json`
- Language effect report:
  - `artifacts/experiments/language_effect_report.json`
- Benchmark report:
  - `artifacts/experiments/benchmark_report.json`

## Expected CSV Header
`sample_token,instruction,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12`

## Submission Steps
1. Export CSV
2. Run precheck and verify `ok=true`
3. Fill in metrics from reports
4. Upload to official submission form

## Notes
- Export now deduplicates by `(sample_token, instruction)`.
- Keep command logs and reports for reproducibility.
