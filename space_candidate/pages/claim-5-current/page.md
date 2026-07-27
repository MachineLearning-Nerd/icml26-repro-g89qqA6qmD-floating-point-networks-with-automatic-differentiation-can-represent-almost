# Claim 5 — full zero-output/gradient block; theorem verdict BLOCKED

## Exact contract

[Lemma 3.5](https://ar5iv.labs.arxiv.org/html/2605.01702#S3.Thmtheorem5.1.1.1)
constructs, under Conditions 1–3 and distinguishability, a network that is
exactly zero on the complete domain, realizes an arbitrary target AD gradient,
and is neutral under `boxplus`. Paper HTML SHA-256:
`5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d`.

## Architecture, calibration, and controls

An explicit paper-order audit found a reproducible released-code discrepancy:
the stored indicator derivative was exactly half the executed derivative at
24 positive active points. Released normalization therefore passed `0/48`
active gradient contracts. A bit-exact correction factor of `0.5` makes the
complete `GradIndicator` output zero with its target gradient in `24/24`
calibratable cases and gives zero value/gradient at `24/24` off-points.

The corrected block is then used in the 13-layer finite-domain composition:
all six arbitrary signed gradients, all six independent values, and all six
next-float off-point controls are exact. The uncalibrated released
normalization is the negative control and fails `48/48` active contracts.

[Raw JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim5_zero_output_gradient.json)
· [standalone cumulative verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py)

## Provenance and verdict

Fixed command:

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

Seeds `260501705`–`260501706`; calibration run
`16e3bf71-a507-41a9-a3dc-6de58e19e5ef`; composition run
`aa4badc0-9263-4559-aa03-4f7ac8deaa69`; winning evidence Git
`abe50406cafc2de150f06a9761e4c8aa2d64054a`; HF `cpu-upgrade`;
estimated 1 core; 64 logical/affinity CPUs visible; Torch 4 threads; 42 s
composition job, 6.337863 s scientific runtime.

**BLOCKED.** This replaces the historical scalar expression with the complete
multilayer branch and documents the correction transparently. Negative active
points were not calibratable, and no complete-domain or `boxplus` proof
certificate is supplied.
