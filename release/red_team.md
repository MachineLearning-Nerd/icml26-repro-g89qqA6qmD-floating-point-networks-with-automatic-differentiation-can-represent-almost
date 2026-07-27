# Evaluator-blind pre-publication red team

The reviewer was given only a fresh candidate directory and the campaign
rubric. It was not told where evidence lived and could not consult OpenResearch
logs, experiment descriptions, or repository source outside the candidate.

## Round 1

Entrypoints: `README.md`, `logbook.json`, `pages/index.md`.

Files opened:

```text
README.md
code/verify_claim6.py
code/verify_current.py
data/claim1_full_network.json
data/claim2_activations.json
data/claim3_theorem32.json
data/claim4_zero_gradient.json
data/claim5_zero_output_gradient.json
data/claim6_raw.json
logbook.json
pages/claim-1-current/page.md
pages/claim-2-current/page.md
pages/claim-3-current/page.md
pages/claim-4-current/page.md
pages/claim-5-current/page.md
pages/claim-6-current/page.md
pages/current-verification/page.md
pages/historical-index/page.md
pages/index.md
pages/overview/page.md
pages/verify/page.md
```

Conclusions not directly verifiable:

```text
claim 6 page missing token: Exact contract
claim 6 page missing token: Raw JSON
canonical cumulative page does not cite the winning run
```

Verdict: **FAIL**. The labels and final-run provenance were fixed without
changing scientific data.

## Round 2

The same traversal was repeated from a second fresh directory. It opened the
same evidence set and returned:

```json
{"problems": [], "verdict": "PASS"}
```

The candidate's standard-library verifier was then run from that fresh
directory:

```text
claim=1 evidence_check=PASS theorem_verdict=BLOCKED
claim=2 evidence_check=PASS theorem_verdict=BLOCKED
claim=3 evidence_check=PASS theorem_verdict=BLOCKED
claim=4 evidence_check=PASS theorem_verdict=BLOCKED
claim=5 evidence_check=PASS theorem_verdict=BLOCKED
claim=6 evidence_check=PASS theorem_verdict=VERIFIED
cumulative_regression=PASS controls=PASS
```

Verdict: **PASS**. A final identical traversal is required on the exact
committed candidate and again on the exact published Hugging Face revision.
