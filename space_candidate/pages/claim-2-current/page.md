# Claim 2 — six activation audit; theorem verdict BLOCKED

## Exact contract

[Theorem 3.1](https://ar5iv.labs.arxiv.org/html/2605.01702#S3.Thmtheorem1.1.1.1)
names ReLU, ELU, GELU, Swish, Sigmoid, and tanh, subject to the paper's
floating-point and activation assumptions. The exact universal network
quantifiers are the same as Claim 1. Paper HTML SHA-256:
`5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d`.

## Witness audit

For float32 (`p=23`, `q=8`, `e_min=-126`, `e_max=127`), the verifier checks
all four numerical bullets of activation Condition 2, one complete
disjunctive route through Condition 3, and agreement between independent
analytic derivatives and Torch autograd.

| activation | Condition 2 | Condition 3 route | analytic/autograd |
| --- | --- | --- | --- |
| ReLU | 4/4 | first, pass | agree |
| ELU | 4/4 | second, pass | agree |
| GELU | 4/4 | second, pass | agree |
| Swish | 4/4 | second, pass | agree |
| Sigmoid | 4/4 | second, pass | agree |
| tanh | 4/4 | second, pass | agree |

The rejecting control \(\sigma(x)=1\) has neither value separation nor the
required derivative lower bound and is rejected.

[Raw JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim2_activations.json)
· [standalone cumulative verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py)

## Provenance and verdict

Fixed command:

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

Seed `260501707`; Git `f176ad13f0b84471ba46fae69c614558f9acbec3`;
run `e84e43c8-ba9f-48df-afa0-257442dbcd6d`; HF `cpu-upgrade`; estimated
1 core; 64 logical/affinity CPUs visible; Torch 4 threads; 42 s job,
6.466838 s scientific runtime.

**BLOCKED.** The named activations are no longer skipped, and their directly
testable sufficient-condition witnesses pass. Condition 1 and a complete
non-ReLU construction for every activation were not independently certified,
so these results are not promoted to a universal theorem verdict.
