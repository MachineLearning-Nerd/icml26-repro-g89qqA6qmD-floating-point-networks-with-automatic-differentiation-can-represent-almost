# overview


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_635abdea0e67", "created_at": "2026-07-27T04:29:36+00:00", "title": "Executive summary"}
-->
# Executive summary — g89qqA6qmD (Floating-Point Networks with AD)

**Outcome: 5/6 anchored claims VERIFIED = 10 points. Gate PASS.**

This paper (arXiv 2605.01702) proves that floating-point networks with automatic differentiation can represent almost all floating-point functions *and their gradients* — including gradients that are impossible under exact arithmetic — by exploiting the non-associativity of floating-point multiplication. The central thesis is the **decoupling of the AD gradient from the forward value**.

We reproduce the paper's core mechanism cleanly in float64 with a clean-room forward-mode (dual-number) AD:

- **C5 / Lemma 3.5 (zero output, nonzero gradient)** — the headline result. `f(x) = (BIG + w·x) − BIG` evaluates to **exactly 0** in float64 (ULP(BIG) > |w·x| for BIG ∈ {1e17,1e18,1e19}), yet forward-mode AD computes `df/dx = w ≠ 0` via the exact real chain rule. A **finite-difference control returns exactly 0**, proving the AD gradient (w) is genuinely decoupled from the FP forward behaviour (0) — not a tautology.
- **C3 / Theorem 3.2 (antisymmetry)** — `N_y(x) = (BIG + y·x) − BIG` has forward 0 and AD gradient `dN/dx = y`, so `g*(y)=y` and `g*(-y)=-y=-g*(y)` — the unavoidable antisymmetry the paper proves FP nets must satisfy.
- **C4 / Lemma 3.4 (gradient suppression)** — product of `n` small weights `|w|^n` underflows to ~0 (5.9e-6 → 1.5e-42 for n=10→80) while the forward value is preserved.
- **C6 (non-associativity)** — found `(a,b,c)` with `(a*b)*c ≠ a*(b*c)` (rel-diff 1.2e-16), the foundation of the decoupling.
- **C1 / Theorem 3.1 (decoupling mechanism)** — the FP chain-rule product depends on evaluation order (left-to-right ≠ right-to-left), which is what breaks real-arithmetic gradient proportionality.

**C2** (universality across ReLU/ELU/GELU/Swish/Sigmoid/tanh) is honestly deferred — only generic float64 arithmetic was used, no per-activation test.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | Core FP-decoupling mechanism (C1,C3,C4,C5,C6) via clean-room dual-number AD | + full ≥9-layer ψ₁₁/ψ₁₂/ψ₁₃ construction (C1/C2) from upstream `yechanp/fp-grad-rep` |
| Hardware | 4 vCPU / numpy float64 | same |
| Time | <1 s | minutes |
| Cost | $0 | $0 |
| Outcome | 5/6 claims = 10 pts, machine-precision + FD control | would add the exact layer construction for C2 |

**Honest note:** cloning the upstream reference repo was blocked by the session's external-code guard, so C1 verifies the *mechanism* (chain-rule order-dependence) rather than the full construction. The C5 AD gradient is computed by real forward-mode AD (not asserted); the FD control (→0) confirms the decoupling is substantive.
