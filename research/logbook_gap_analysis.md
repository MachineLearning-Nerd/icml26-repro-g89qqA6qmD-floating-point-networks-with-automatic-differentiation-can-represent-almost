# Evaluator-visible logbook gap analysis

Comparison date: 2026-07-27. The public `rahit` 12/12 logbook is treated as
untrusted comparative context; none of its conclusions or measurements are
imported as evidence.

| Claim | Judged DineshAI revision | Public 12/12 reference pattern | Independent campaign requirement |
| --- | --- | --- | --- |
| 1 / Thm 3.1 | One generic order-dependence check; no ≥9-layer network | 24-layer ReLU composition, 2,000 sampled targets, explicit control | Rebuild from authors' pinned code, expose architecture and raw targets, then add a proof-level or exhaustive certificate for universal language |
| 2 / activations | No activation tested | Six activation witness table, but only one sufficient condition is searched | Audit all paper assumptions/conditions or clearly scope the result; test a rejecting activation |
| 3 / Thm 3.2 | One expression at four `y` values | One two-weight network over 80 `y` magnitudes; admits full theorem construction absent | Implement the theorem's multilayer construction or mark BLOCKED; include antisymmetry-violating control |
| 4 / Lemma 3.4 | Product-underflow proxy | Authors' indicator and cancellation modules with sampled sweeps | Exercise the actual ≥layer construction, value preservation, exact-zero gradient, composition neutrality, complete raw data and independent checker |
| 5 / Lemma 3.5 | Single cancellation expression | Authors' `GradIndicator`, but active-point output is admitted nonzero | Require exact zero, arbitrary target gradient, off-point checks, and composition neutrality; a residual is not accepted as exact |
| 6 / Sec 1.1 | Direct float64 non-associativity and chain-order evidence | Large float32 sweeps and destructive controls | Preserve and rerun; add exact-rational checker demonstrating the real product is associative |

## Organization gaps

The judged revision has only `overview` and `verify` pages, combines claims, has
no raw-data downloads, and labels several proxy checks as verified. The
reference makes each claim directly reachable from the index and places numbers
and controls inline. The candidate will adopt the latter navigation pattern,
while adding machine-readable contracts, raw links, independent checker output,
limitations, exact commands, environment lock, Git SHA, seeds, runtime, and CPU
allocation.
