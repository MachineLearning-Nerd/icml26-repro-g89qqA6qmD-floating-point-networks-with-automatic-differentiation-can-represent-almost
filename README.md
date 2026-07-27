# Reproduction campaign: floating-point networks and AD

This branch is the frozen OpenResearch baseline for arXiv 2605.01702,
“Floating-Point Networks with Automatic Differentiation Can Represent Almost
All Floating-Point Functions and Their Gradients.”

It seeds the authors' public implementation
[`yechanp/fp-grad-rep`](https://github.com/yechanp/fp-grad-rep) at
`3cf61240748f09af29084556b1876eddc1e462fb`, adds a locked Python 3.12 `uv`
environment, and converts the interactive author demo into a deterministic,
headless verifier. The baseline also reruns the previously judged
non-associativity evidence. It does **not** claim that the five construction
claims are verified: those remain `BLOCKED` here pending faithful child
experiments.

| Branch / experiment | Purpose | Exact run command | Assessment | Compute |
| --- | --- | --- | --- | --- |
| `orx/frozen-author-code-baseline` | Author-code smoke test plus historical Claim 6 regression | `uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py` | Claim 6 VERIFIED; Claims 1–5 BLOCKED at baseline | Hugging Face `cpu-upgrade` |

Source and theorem anchors are recorded in
[`research/source_audit.md`](research/source_audit.md), and the comparison with
the public 12/12 reference logbook is in
[`research/logbook_gap_analysis.md`](research/logbook_gap_analysis.md).

## Authors' original README

# Floating-Point Networks with Automatic Differentiation Can Represent Almost All Floating-Point Functions and Their Gradients

## Paper Abstract

Theoretical studies show that for any differentiable function on a compact
domain, there exists a neural network that approximates both the function
values and gradients. However, such a result cannot be used in practice since
it assumes real parameters and exact internal operations. In contrast, real
implementations only use a finite subset of reals and machine operations with
round-off errors. In this work, we investigate whether a similar result holds
for neural networks under floating-point arithmetic, when the gradient with
respect to the input is computed by the automatic differentiation algorithm.

## Authors' original command

```bash
python plot_indicators.py
```
