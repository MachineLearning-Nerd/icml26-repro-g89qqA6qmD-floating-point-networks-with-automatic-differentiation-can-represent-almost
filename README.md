# Reproduction: exact values and independently chosen AD gradients

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/blob/main/notebooks/floating_point_networks.py)

This repository reproduces the six judged claims of arXiv
[2605.01702](https://arxiv.org/abs/2605.01702), *Floating-Point Networks with
Automatic Differentiation Can Represent Almost All Floating-Point Functions
and Their Gradients*. The previous logbook demonstrated arithmetic mechanisms
but not the paper's multilayer constructions.

The new reproduction executes the full released building blocks with an
independent paper-order float32 interpreter. Its headline finite result is one
13-layer ReLU network with bit-exact values and independently selected signed
gradients on all six points of its declared domain:

| metric | paper claim | observed |
| --- | --- | ---: |
| minimum depth | every \(L\ge9\) | tested depth `13` |
| selected-domain values | exact for every input | `6/6` bit-exact |
| selected-domain gradients | exact independent targets | `6/6` bit-exact |
| neighboring off-points | zero | `6/6` value/gradient zero |
| loss-dependent network | every \(L\ge569\) in float32 | depth `569`, `60/60` linear-odd gradients exact |

Assessment: **substantial faithful finite corroboration, but Claims 1–5 remain
BLOCKED at universal theorem scope; Claim 6 is VERIFIED.** Finite experiments
cannot prove the theorem's quantifiers over the complete bounded
floating-point domain, every target map, dimension, and admissible activation.
No result is promoted from toy evidence or presented as a universal proof.

All formal jobs used Hugging Face `cpu-upgrade`, no GPU. The final cumulative
job ran in 64 seconds (24.209091 seconds scientific runtime), with 64
logical/affinity CPUs visible and Torch capped at four threads. Python 3.12,
NumPy 2.3.1, and Torch 2.7.1+cpu are locked with `uv`.

- [Illustrated technical report](reports/floating-point-networks/report.md)
- [Self-contained marimo tutorial](notebooks/floating_point_networks.py)
- [Current evaluator-visible candidate](space_candidate/pages/index.md)
- [Exact source and quantifier audit](research/source_audit.md)
- [Gap analysis against the public 12/12 reference](research/logbook_gap_analysis.md)

| Branch / experiment | Purpose | Exact run command | Assessment | Compute |
| --- | --- | --- | --- | --- |
| `main` | Publication surface | Not run as an experiment (publication surface) | Report, notebook, and exact published logbook text | — |
| [`orx/frozen-author-code-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/tree/orx/frozen-author-code-baseline) | Frozen author-code baseline and Claim 6 regression | `uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py` | Claim 6 VERIFIED; other claims BLOCKED at baseline | HF `cpu-upgrade`, CPU |
| [`orx/finite-domain-full-network-composition`](https://github.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/tree/orx/finite-domain-full-network-composition) | 13-layer exact value/gradient network | `uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py` | `6/6` values and gradients exact; universal scope BLOCKED | HF `cpu-upgrade`, CPU |
| [`orx/complete-disjunctive-activation-witness-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/tree/orx/complete-disjunctive-activation-witness-audit) | Six activation Conditions 2/3 | `uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py` | `6/6` activations pass tested witnesses; universal scope BLOCKED | HF `cpu-upgrade`, CPU |
| [`orx/theorem-3-2-four-route-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/tree/orx/theorem-3-2-four-route-audit) | Full 569-layer network and falsification route | `uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py` | `60/60` linear-odd gradients exact; arbitrary odd scope BLOCKED | HF `cpu-upgrade`, CPU |
| [`orx/release-ready-cumulative-verifier`](https://github.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/tree/orx/release-ready-cumulative-verifier) | Rerun every accepted check and control | `uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py` | cumulative verifier PASS; Claims 1–5 BLOCKED, Claim 6 VERIFIED | HF `cpu-upgrade`, CPU |

Run the tutorial locally with `marimo edit
notebooks/floating_point_networks.py` or `marimo run
notebooks/floating_point_networks.py`. The notebook embeds the formal results;
it does not require rerunning the experiment.

## Reproduction internals

The campaign seeds the authors' public implementation
[`yechanp/fp-grad-rep`](https://github.com/yechanp/fp-grad-rep) at
`3cf61240748f09af29084556b1876eddc1e462fb`, pins the environment, and converts
the interactive author demo into a deterministic headless verifier. The
explicit interpreter and experiment routes are under `reproduction/`.

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
