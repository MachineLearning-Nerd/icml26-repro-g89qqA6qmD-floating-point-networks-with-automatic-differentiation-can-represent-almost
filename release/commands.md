# Command provenance

All formal experiment nodes inherited this exact command:

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

## Startup and source audit

```bash
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx projects --json
orx runs ae512de4-00b5-43f2-91a6-4e98a389f26c
git branch -a
git status --short
git rev-parse HEAD
df -h .
curl -L -A 'Mozilla/5.0 (compatible; OpenResearch-Repro/1.0; +https://openresearch.ai)' https://ar5iv.labs.arxiv.org/html/2605.01702
orx paper 2605.01702 --full
```

The paper HTML, live verdict dataset, judged Space revision, public 12/12
comparison Space, and authors' code revision were downloaded into fresh
temporary directories. SHA-256 manifests were computed before candidate work.
The verdict filter was exact:

```text
space_id == "DineshAI/g89qqA6qmD"
```

## Experiment orchestration

Every child was created with:

```bash
orx create-experiment ae512de4-00b5-43f2-91a6-4e98a389f26c --title "<title>" --parent <parent-id>
git fetch origin
git checkout <experiment-branch>
git add <scoped-files>
git commit -m "<scoped-message>"
git push origin HEAD
```

Every CPU run used the same backend form and no GPU:

```bash
orx exp run <experiment-id> --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim --timeout 1h
orx exp wait <experiment-id> --timeout 480
orx logs <run-id> --bytes <bounded-byte-count>
```

Formal completed run IDs, oldest to newest:

```text
569fb7de-fe90-4196-b3d5-8c8a75c2d1a9  frozen baseline
07f978eb-2579-4e7f-b8ec-3db78c5cef54  evaluator-visible Claim 6
3653695a-6572-427a-91ca-3bee57dcd145  author Torch sweep
c9ae8d24-b7a5-49b2-a697-cde7fd52baed  paper-order interpreter
16e3bf71-a507-41a9-a3dc-6de58e19e5ef  derivative calibration
aa4badc0-9263-4559-aa03-4f7ac8deaa69  13-layer composition
96f08f7b-5568-4feb-9db6-4a7297372918  initial activation audit
e84e43c8-ba9f-48df-afa0-257442dbcd6d  complete activation audit
cadbe6ad-ea68-4097-be0d-554181b001f8  Theorem 3.2 four routes
4064ba4a-a790-48ab-ab16-e7c674c7151b  cumulative candidate
b348d98a-3474-4a1a-8fee-8dcd084e0b85  release-winning repeat
```

Failed diagnostic runs are retained in `orx runs`; none are used as evidence.

## Local short checks

These are single-core, sub-five-minute validation or presentation commands:

```bash
uv run --frozen --no-sync python space_candidate/code/verify_current.py
uv run --frozen --no-sync python -m py_compile run_reproduction.py
uv run --frozen --no-sync python -m json.tool space_candidate/logbook.json
uv run --frozen --no-sync python tools/materialize_evidence.py
marimo check --fix notebooks/floating_point_networks.py
marimo check --strict notebooks/floating_point_networks.py
xmllint --noout reports/floating-point-networks/images/<figure>.svg
rsvg-convert -w 1120 reports/floating-point-networks/images/<figure>.svg -o /tmp/<figure>.png
uv run --frozen --no-sync python tools/audit_candidate.py <fresh-candidate> --require-winning-run
```

All five rendered figures were visually inspected. No training or scientific
experiment was run directly in the local shell.
