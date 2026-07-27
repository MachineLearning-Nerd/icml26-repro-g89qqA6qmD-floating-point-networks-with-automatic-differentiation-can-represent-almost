# Claim 3 — executed 569-layer network; theorem verdict BLOCKED

## Exact contract

[Theorem 3.2](https://ar5iv.labs.arxiv.org/html/2605.01702#S3.Thmtheorem2.1.1.1)
asserts that for arbitrary \(f^*\) and arbitrary
\(g^*(x,y)\) satisfying \(g^*(x,-y)=-g^*(x,y)\), one fixed network works for
every bounded input and loss derivative and every
\(L\ge2^{q+1}+2p+11\). For float32 this lower bound is exactly 569 layers.
Paper HTML SHA-256:
`5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d`.

## Four independent routes

1. The historical scalar cancellation was reproduced and rejected as toy.
2. One fixed 569-layer ReLU network executed 556 identity-padding layers plus
   the 13-layer composition. Across six inputs and ten signed loss derivatives,
   it preserved `60/60` values and exactly realized `60/60` gradients for the
   nontrivial family \(g(x,y)=k(x)y\).
3. A source/construction audit found that this linear family does not
   reconstruct the arbitrary-odd lookup quantifier.
4. A dedicated falsification route used the nonlinear odd target
   \(g(x,y)=k(x)y|y|\). This particular network missed `48/60`, but that is
   not a counterexample to an existential claim over all networks.

The negative control \(g(x,y)=k(x)|y|\) violates antisymmetry at `30/30`
signed pairs and is rejected.

[Raw JSON](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/data/claim3_theorem32.json)
· [standalone cumulative verifier](https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/code/verify_current.py)

## Provenance and verdict

Fixed command:

```bash
uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
```

Seed `260501708`; Git `2b72f35a33258359b34be671a14a6bb4cbb17f02`;
run `cadbe6ad-ea68-4097-be0d-554181b001f8`; HF `cpu-upgrade`; estimated
1 core; 64 logical/affinity CPUs visible; Torch 4 threads; 53 s job,
17.163747 s scientific runtime.

**BLOCKED.** This replaces the one-expression toy with a theorem-depth
network and completes the mandatory fourth falsification route. Arbitrary odd
dependence remains uncertified, and the observed candidate mismatch is
explicitly not mislabeled as falsification.
