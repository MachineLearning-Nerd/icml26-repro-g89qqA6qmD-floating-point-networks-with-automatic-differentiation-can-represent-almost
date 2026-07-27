# Repro - Floating-Point Networks with AD

**Previous live judged score: 5/12.** This candidate does not claim a score
increase. It replaces toy/proxy evidence with full multilayer finite-domain
checks where possible, and it keeps universal theorem claims BLOCKED unless
proof-level evidence exists.

The current claim-by-claim verification is listed first. The original judged
pages remain reachable and unchanged, but their proxy verifiers are superseded.

## Current verification

| Claim | Current page | Honest verdict |
| --- | --- | --- |
| all | [Visibility matrix and cumulative command](#/current-verification) | — |
| 1 | [13-layer value/gradient network](#/claim-1-current) | BLOCKED |
| 2 | [six activation witnesses](#/claim-2-current) | BLOCKED |
| 3 | [569-layer loss-derivative network](#/claim-3-current) | BLOCKED |
| 4 | [full zero-gradient block](#/claim-4-current) | BLOCKED |
| 5 | [full zero-output/gradient block](#/claim-5-current) | BLOCKED |
| 6 | [non-associativity with exact checker](#/claim-6-current) | VERIFIED |

The current standalone verifier is
[`code/verify_current.py`](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py).
It exits nonzero on an evidence or control regression. `BLOCKED` is an honest
theorem-level result, not a hidden failed assertion.

## Historical rejected baseline

These pages are preserved exactly as judged at
`56de71e80b8e71a2928cdad2323cc7d7ad877d78`. Their apparent verification of
Claims 1–5 is rejected by the current campaign and must not be used as the
current verifier.

| Page |
| --- |
| [Historical rejected baseline — verify](#/verify) |
| [Historical rejected baseline — overview](#/overview) |
| [Historical rejected baseline — original index](#/historical-index) |
