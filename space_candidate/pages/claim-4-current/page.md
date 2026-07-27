# Claim 4 — full zero-gradient block; theorem verdict BLOCKED

## Exact contract

[Lemma 3.4](https://ar5iv.labs.arxiv.org/html/2605.01702#S3.Thmtheorem4.1.1.1)
constructs, under Conditions 1–2 and distinguishability, a network with the
target function values, exactly zero AD gradient for every stated input
gradient, and neutrality under the paper's `boxplus` composition. Paper HTML
SHA-256:
`5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d`.

## Full architecture evidence

The released `ZeroGradIndicator`—not the historical product-underflow proxy—
was evaluated with an independent paper-order float32 affine interpreter.
Across 48 active cases it produced the target value with exact zero gradient
in `48/48`; all 48 neighboring off-points gave exact zero value and gradient.
The same blocks are used in the 13-layer composed network, where all six
target values and all six independent target gradients are exact.

The destructive control zeros the companion branch's target-gradient weight;
the intended gradient is lost, so the control fires for its intended reason.

[Raw JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim4_zero_gradient.json)
· [standalone cumulative verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py)

## Provenance and verdict

Fixed command:

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

Seed `260501704`; Git `75df430674d839cbf996f17a4ca180c897efc8a1`;
run `c9ae8d24-b7a5-49b2-a697-cde7fd52baed`; HF `cpu-upgrade`; estimated
1 core; 64 logical/affinity CPUs visible; Torch 4 threads; 37 s job,
3.446874 s scientific runtime.

**BLOCKED.** This is faithful finite evidence for the actual multilayer
architecture. It does not certify every target function on the complete
domain or universal `boxplus` neutrality, so it remains below a proof-level
lemma verdict.
