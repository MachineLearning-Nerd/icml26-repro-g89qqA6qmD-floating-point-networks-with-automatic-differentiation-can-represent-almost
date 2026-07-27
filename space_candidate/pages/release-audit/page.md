# Release audit and evaluator-blind review

## Blind review round 1 — FAIL, fixed

The candidate was copied into a fresh empty directory. The reviewer began only
from `README.md`, `logbook.json`, and `pages/index.md`; no OpenResearch logs,
repository knowledge, or hidden artifacts were available.

It opened 21 distinct files and found three discoverability defects:

```text
claim 6 page missing token: Exact contract
claim 6 page missing token: Raw JSON
canonical cumulative page does not cite the winning run
```

The science was unchanged. Claim 6's headings/link label were made canonical,
and the winning repeat's run ID, Git SHA, allocation, and runtime were added to
the cumulative page.

## Blind review round 2 — PASS

The review was repeated from a second fresh directory and opened:

```text
README.md
logbook.json
pages/index.md
pages/current-verification/page.md
pages/claim-1-current/page.md ... pages/claim-6-current/page.md
code/verify_current.py
code/verify_claim6.py
data/claim1_full_network.json ... data/claim6_raw.json
pages/historical-index/page.md
pages/overview/page.md
pages/verify/page.md
```

Result:

```text
problems=[]
verdict=PASS
cumulative_regression=PASS controls=PASS
```

## Historical evidence safety

The judged revision
`56de71e80b8e71a2928cdad2323cc7d7ad877d78` remains immutable in Hugging Face
history. `pages/overview/page.md` and `pages/verify/page.md` are byte-identical
to that revision. Its original `pages/index.md` is preserved byte-identically
as `pages/historical-index/page.md`. Existing binary assets are not part of
the text-only upload and therefore are not overwritten or deleted.

Current verification is first in navigation. All historical navigation labels
begin exactly with **Historical rejected baseline**.

## Release forecast, not a judge result

Previous live judged score: **5/12**. Conservative projected range after this
candidate: **7–10/12**. Best-supported possible score: **10/12**, explicitly a
forecast. Only a new live judge verdict can change the score.
