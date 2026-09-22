# 2026-09-09 retrospective evidence snapshot

Local source HEAD: `41343ba`. Server read-only status: 15:51–15:53 KST.
Copied through `bin/nx sh` tar stream. No server source uploads, training, cache mutation,
new test-label inspection, or cost measurement. Snapshot is not a pre-run attestation.

- `artifacts/kurosiwo`: three evaluation reports and input-exclusion IDs (no checkpoints).
- `artifacts/streaming_t1v`, `streaming_t1arch`, `streaming_t1xs`: 36/36/12 reports.
- `artifacts/italy_sealed`: four arms × three seeds plus completed size-stratified report.
- `logs`, `code`: observed logs and current source, not guaranteed executed source snapshots.
- `kurosiwo_s1_cache/meta.jsonl`: original 7,000-item manifest; eval excluded val/test one each.
- `olmo_streaming_dev/single_s1_dates`: acquisition assignments, not images or labels.
- `sen12_gp_contract/t1_manifest.json`: existing membership, not a changed split.
- `server_status.txt`: checkpoint mtimes and completion logs; no claim that mtime proves executed source.
- `summary.json`: independent report arithmetic; arithmetic pass is not clean-protocol certification.
- `architecture_summary.json`: existing audit tool on completed architecture reports.
- `input_sha256.json`: hashes of copied inputs (excludes generated summaries/this README).

Reproduce from `_work`:

```sh
python3 code/audit_streaming_external_reports.py artifacts/streaming_review_20260909 --out artifacts/streaming_review_20260909/summary.json
python3 code/audit_streaming_progress.py artifacts/streaming_review_20260909 --out artifacts/streaming_review_20260909/architecture_summary.json
python3 -m unittest discover -s tests -p 'test_streaming_external_report_audit.py' -v
```

Interpretation and unresolved validation-selection/causal-date issues are documented in
`docs/STREAMING_RESEARCH_UPDATE_2026_09_09.md`. The old 2026-09-08 invalid-run snapshot is retained.
