# Claim 6 — current VERIFIED evidence

## Exact contract and source

Section 1.1 of arXiv 2605.01702 states that the construction exploits
floating-point non-associativity to decouple the AD-computed gradient from the
classical exact-real chain-rule proportionality. Source:
[`#S1.SS1.SSS1`](https://ar5iv.labs.arxiv.org/html/2605.01702#S1.SS1.SSS1).
The retrieved paper HTML SHA-256 was
`5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d`
on 2026-07-27.

This page tests the arithmetic mechanism, not the universal representation
theorems.

## Fixed command and environment

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

Evidence-producing Git SHA:
`ac11dbd077d7c7f3a52ab3f53bea48e866ca4ffa`. Deterministic seed:
`260501702`. Python 3.12.12, NumPy 2.3.1, Torch 2.7.1+cpu. Hugging Face
`cpu-upgrade`; 64 logical CPUs were visible, Torch was capped at 4 threads,
formal run duration was 26 seconds, and the scientific verifier took 0.144128
seconds.

## Raw result

For the finite binary64 operands

```text
a = -3.390181137426782e-49
b =  1.9001924576251575e-36
c =  2.9840622466904965e-46
```

the two explicitly rounded products were:

| grouping | binary64 result |
| --- | ---: |
| `(a*b)*c` | `-1.9223318928897442e-130` |
| `a*(b*c)` | `-1.9223318928897446e-130` |

They differ by one or more binary64 bits; the relative gap is
`2.0829733955809995e-16`.

Download the complete [Raw JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim6_raw.json)
and the standalone [verifier
source](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_claim6.py).
It is also included in the [cumulative regression
verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py).
The verifier exits nonzero on any failed assertion.

## Independent checker

`fractions.Fraction.from_float` reconstructs each binary64 operand exactly.
The exact rational products
`(a*b)*c` and `a*(b*c)` are equal. Thus the discrepancy is introduced by
intermediate floating-point rounding, not by different real products.

Checker output:

```text
float_bitwise_difference=true
exact_rational_products_equal=true
```

## Negative control

For the exactly representable powers of two `(2.0, 0.5, 8.0)`, both groupings
equal `8.0`; the non-associativity detector does not fire.

```text
nonassociativity_detector_fires=false
expected=false
control_pass=true
```

## Verdict and limitation

**VERIFIED.** This directly verifies the non-associative arithmetic mechanism
and preserves the previous full-credit claim. It does not by itself verify
Theorem 3.1, Theorem 3.2, Lemma 3.4, or Lemma 3.5.
