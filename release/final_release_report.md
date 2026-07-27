Previous live judged score: `5/12`

Conservative projected score range after the proposed change: `7–10/12`

Best-supported possible new score: `10/12` — **forecast, not a judge result**

# Final release report

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 0 | 1–2 | MEDIUM | BLOCKED | One 13-layer network gives `6/6` bit-exact arbitrary selected values and gradients, identity indicators, and `6/6` off-controls. Universal complete-domain/map quantifiers lack a proof certificate. |
| 2 | 0 | 1–2 | MEDIUM | BLOCKED | All six named activations pass all tested Conditions 2/3 witnesses and analytic/autograd checks; Condition 1 and full non-ReLU networks are uncertified. |
| 3 | 1 | 1–2 | LOW | BLOCKED | One 569-layer network gives `60/60` exact linear odd targets. Three verification routes plus the mandatory fourth falsification route do not certify arbitrary antisymmetric dependence. |
| 4 | 1 | 1–2 | MEDIUM | BLOCKED | The full released zero-gradient block passes `48/48` active and off cases under an independent paper-order interpreter; universal composition neutrality is uncertified. |
| 5 | 1 | 1–2 | MEDIUM | BLOCKED | The full calibrated gradient block passes `24/24` positive points and `6/6` composed targets; negative points and universal composition neutrality remain unresolved. |
| 6 | 2 | 2 | HIGH | VERIFIED | Concrete binary64 grouping difference, exact-rational associativity checker, and powers-of-two non-triggering control all pass. |

Current total score: **5/12**.

Conservative projected total score range: **7–10/12**.

Best-supported possible total: **10/12**, pending the live evaluator.

Claims 1–5 changed materially since the previous judge result: each now has an
exact source contract, full multilayer or activation-specific code, raw data,
independent checker, and a destructive control. Claim 6 is unchanged
scientifically and passes the cumulative regression.

Claims 1–5 remain BLOCKED under the campaign's universal-theorem gate. Claim 3
has LOW confidence after exactly four materially different routes:

1. historical scalar mechanism, rejected as toy;
2. executed theorem-depth network for a linear odd family;
3. exact source/construction quantifier audit;
4. assumption-satisfying nonlinear odd falsification attempt, which misses
   `48/60` for the candidate but cannot falsify an existential all-network
   claim.

## Winning experiment and compute

Winning branch:
`orx/release-ready-cumulative-verifier`

Git SHA:
`e412c472eb5adc6beb00bb311ce7f1fbf698170d`

Run:
`b348d98a-3474-4a1a-8fee-8dcd084e0b85`

Exact command:

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

Backend: Hugging Face `cpu-upgrade`; no GPU. Estimated demand before launch:
up to four Torch CPU threads. Actual allocation visible to the process: 64
logical CPUs and 64 affinity CPUs; Torch capped at four threads. Job runtime:
64 seconds. Scientific runtime: 24.20909078908153 seconds. Hugging Face did
not expose a monetary cost, so cost is reported as unavailable rather than
estimated.

## Experiment-tree summary

The tree is a downward sequence of focused bushes:

```text
frozen baseline
└── evaluator-visible Claim 6
    ├── author Torch contract sweep
    └── explicit paper-order interpreter  ← promoted
        └── derivative calibration
            └── 13-layer finite composition
                └── activation audit
                    └── complete activation audit
                        └── Theorem 3.2 four-route audit
                            └── cumulative candidate
                                └── release-ready cumulative verifier  ← winner
                                    └── publication surfaces/audit
```

## Historical subset and evaluator visibility

The exact judged Hugging Face revision is
`56de71e80b8e71a2928cdad2323cc7d7ad877d78`. Its 15-path file set remains a
subset of the planned final Space tree because publication is additive and the
upload performs no deletions. Existing binary assets are excluded from the
text-only upload and remain untouched. The judged `overview` and `verify`
pages are byte-identical; the original index is preserved byte-identically as
`pages/historical-index/page.md`. The immutable judged revision remains
addressable in repository history.

The final evaluator-blind traversal must report no problems from only
`README.md`, `logbook.json`, and `pages/index.md`. The visibility matrix on the
candidate current-verification page has no missing cells.

## Publication action

After the exact committed candidate passes validation, secret scan, allowlist
hash check, historical subset check, and final blind traversal, the action is:

1. upload only the allowlisted text files to the existing
   `DineshAI/g89qqA6qmD` Space through the Hugging Face API;
2. verify the exact returned Space revision by downloading it afresh;
3. fast-forward the public GitHub `main` branch to the publication commit and
   confirm it with `git ls-remote`;
4. mark the paper awaiting judge.

No second Space will be created. No score increase will be claimed before the
live judge records a new verdict.

## Post-publication verification

Publication completed additively to the existing Space at revision
`1bc79260f99dd7df0873360e53a85e04ebe2a5e8`. A fresh exact-revision download
confirmed:

- all 28 allowlisted upload hashes match;
- all 15 judged-revision paths remain present;
- the two judged evidence pages and preserved historical index retain their
  protected SHA-256 hashes;
- the blind canonical traversal opens all six current claim pages, both
  verifiers, every raw JSON file, and historical evidence with zero problems;
- the cumulative verifier reports six evidence checks PASS and all controls
  PASS;
- no credential pattern is present;
- the current Space head equals the published revision.

GitHub `main` was fast-forwarded with the same candidate text, report, and
notebook. The campaign is now **awaiting live judge**. The score remains the
previous recorded `5/12`.
