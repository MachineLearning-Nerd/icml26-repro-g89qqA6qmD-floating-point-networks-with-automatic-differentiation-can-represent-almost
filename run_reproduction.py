"""Frozen baseline verifier for arXiv 2605.01702.

The root experiment deliberately preserves the judged state: Claims 1--5 are
BLOCKED because the universal multilayer constructions are not yet reproduced.
Claim 6 is rerun with an independent exact-rational checker. The authors'
published ReLU modules are smoke-tested separately and are not promoted into
full theorem verdicts at baseline.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models import GradIndicator, ZeroGradIndicator
from src.utils import next_float


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
SEED = 260501702
SOURCE_SHA256 = "5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d"
FIXED_COMMAND = "uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py"


@dataclass(frozen=True)
class Dual:
    value: np.float64
    grad: np.float64

    def __add__(self, other: "Dual | float") -> "Dual":
        rhs = other if isinstance(other, Dual) else Dual(np.float64(other), np.float64(0.0))
        return Dual(np.float64(self.value + rhs.value), np.float64(self.grad + rhs.grad))

    __radd__ = __add__

    def __sub__(self, other: "Dual | float") -> "Dual":
        rhs = other if isinstance(other, Dual) else Dual(np.float64(other), np.float64(0.0))
        return Dual(np.float64(self.value - rhs.value), np.float64(self.grad - rhs.grad))

    def __mul__(self, other: "Dual | float") -> "Dual":
        rhs = other if isinstance(other, Dual) else Dual(np.float64(other), np.float64(0.0))
        return Dual(
            np.float64(self.value * rhs.value),
            np.float64(self.grad * rhs.value + self.value * rhs.grad),
        )

    __rmul__ = __mul__


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def find_nonassociative(rng: np.random.Generator, trials: int = 20_000) -> dict[str, float] | None:
    exponents = rng.integers(-300, 301, size=(trials, 3))
    mantissas = rng.uniform(-2.0, 2.0, size=(trials, 3))
    values = np.ldexp(mantissas, exponents).astype(np.float64)
    for a, b, c in values:
        left = np.float64(np.float64(a * b) * c)
        right = np.float64(a * np.float64(b * c))
        if math.isfinite(left) and math.isfinite(right) and left != right:
            return {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "left": float(left),
                "right": float(right),
                "relative_gap": float(abs(left - right) / max(abs(left), abs(right))),
            }
    return None


def historical_checks() -> dict[str, Any]:
    big = np.float64(1e17)
    x = Dual(np.float64(1.0), np.float64(1.0))
    cancellation = big + np.float64(3.0) * x - big
    antisymmetry = []
    for y in (0.3, 0.7, 1.0, 2.5):
        pos = big + np.float64(y) * x - big
        neg = big + np.float64(-y) * x - big
        antisymmetry.append(
            {
                "y": y,
                "forward_pos": float(pos.value),
                "forward_neg": float(neg.value),
                "grad_pos": float(pos.grad),
                "grad_neg": float(neg.grad),
                "odd": bool(pos.grad == -neg.grad),
            }
        )
    suppression = [
        {"layers": n, "weight": 0.3, "gradient": float(np.float64(0.3) ** n)}
        for n in (10, 20, 40, 80)
    ]
    return {
        "single_expression_zero_forward_nonzero_grad": {
            "forward": float(cancellation.value),
            "gradient": float(cancellation.grad),
        },
        "antisymmetry_proxy": antisymmetry,
        "product_underflow_proxy": suppression,
        "limitations": (
            "These reproduce the historical toy/proxy mechanisms only. They are not "
            "accepted as evidence for Theorem 3.1, Theorem 3.2, Lemma 3.4, or Lemma 3.5."
        ),
    }


def author_code_smoke() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    dtype = torch.float32
    z = torch.tensor([0.25], dtype=dtype)
    value = torch.tensor([1.0], dtype=dtype)
    points = [z.item(), next_float(z)[0].item(), -0.25, 0.0, 1.0]

    zero_grad = ZeroGradIndicator(z, value, dtype=dtype, device="cpu")
    grad_indicator = GradIndicator(z, value, dtype=dtype, device="cpu")
    rows: list[dict[str, Any]] = []
    for point in points:
        x = torch.tensor([point], dtype=dtype, requires_grad=True)
        y0 = zero_grad(x)
        g0 = torch.autograd.grad(y0, x)[0]
        x1 = torch.tensor([point], dtype=dtype, requires_grad=True)
        y1 = grad_indicator(x1)
        g1 = torch.autograd.grad(y1, x1)[0]
        rows.append(
            {
                "x": float(point),
                "zero_grad_indicator_output": float(y0.detach().item()),
                "zero_grad_indicator_grad": float(g0.detach().item()),
                "grad_indicator_output": float(y1.detach().item()),
                "grad_indicator_grad": float(g1.detach().item()),
            }
        )
    active = rows[0]
    smoke_pass = (
        active["zero_grad_indicator_output"] == 1.0
        and active["zero_grad_indicator_grad"] == 0.0
        and active["grad_indicator_output"] == 0.0
        and active["grad_indicator_grad"] == 1.0
    )
    return {
        "authors_commit": "3cf61240748f09af29084556b1876eddc1e462fb",
        "models_sha256": "beccdd9c9ded1e17eaf829f85f17c48229eb6e39c49fb6547728b76414027139",
        "rows": rows,
        "active_point_contract_pass": smoke_pass,
        "scope": "Author-code smoke test only; not a theorem-level verdict.",
    }


def claim6(seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    triple = find_nonassociative(np.random.default_rng(seed))
    if triple is None:
        raise AssertionError("No non-associative finite float64 triple found")

    a, b, c = triple["a"], triple["b"], triple["c"]
    exact_left = (Fraction.from_float(a) * Fraction.from_float(b)) * Fraction.from_float(c)
    exact_right = Fraction.from_float(a) * (Fraction.from_float(b) * Fraction.from_float(c))
    independent = {
        "float_bitwise_difference": bool(
            np.float64(triple["left"]).view(np.uint64)
            != np.float64(triple["right"]).view(np.uint64)
        ),
        "exact_rational_products_equal": bool(exact_left == exact_right),
        "checker": "Python Fraction constructed from the exact binary64 operands",
    }

    pa, pb, pc = np.float64(2.0), np.float64(0.5), np.float64(8.0)
    control_left = np.float64(np.float64(pa * pb) * pc)
    control_right = np.float64(pa * np.float64(pb * pc))
    control = {
        "operands": [2.0, 0.5, 8.0],
        "left": float(control_left),
        "right": float(control_right),
        "nonassociativity_detector_fires": bool(control_left != control_right),
        "expected": False,
        "pass": bool(control_left == control_right),
    }
    raw = {
        "seed": seed,
        "trials_budget": 20_000,
        "found": True,
        "triple": triple,
        "verdict": "VERIFIED",
    }
    if not all(independent.values()) or not control["pass"]:
        raise AssertionError("Claim 6 independent checker or negative control failed")
    return raw, independent, control


def write_contracts() -> None:
    claims = {
        1: ("Theorem 3.1", "BLOCKED", "No universal multilayer construction in baseline."),
        2: ("Theorem 3.1 activation family", "BLOCKED", "No per-activation theorem audit in baseline."),
        3: ("Theorem 3.2", "BLOCKED", "Historical single-expression check is only a proxy."),
        4: ("Lemma 3.4", "BLOCKED", "Historical product underflow is not the lemma construction."),
        5: ("Lemma 3.5", "BLOCKED", "Historical cancellation expression is not the lemma construction."),
        6: (
            "Section 1.1 non-associativity mechanism",
            "VERIFIED",
            "A concrete binary64 counterexample plus exact-rational and powers-of-two controls.",
        ),
    }
    anchors = {
        1: "S3.Thmtheorem1.1.1.1",
        2: "S3.Thmtheorem1.1.1.1",
        3: "S3.Thmtheorem2.1.1.1",
        4: "S3.Thmtheorem4.1.1.1",
        5: "S3.Thmtheorem5.1.1.1",
        6: "S1.SS1.SSS1",
    }
    for number, (title, verdict, basis) in claims.items():
        directory = ARTIFACTS / f"claim_{number}"
        dump_json(
            directory / "claim_contract.json",
            {
                "claim": number,
                "title": title,
                "source_url": f"https://ar5iv.labs.arxiv.org/html/2605.01702#{anchors[number]}",
                "paper_source_sha256": SOURCE_SHA256,
                "required_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
                "baseline_verdict": verdict,
                "basis": basis,
            },
        )
        dump_text(directory / "exact_command.txt", FIXED_COMMAND)
        dump_text(
            directory / "limitations_and_deviations.md",
            f"# Limitations and deviations\n\nBaseline verdict: **{verdict}**.\n\n{basis}",
        )


def main() -> None:
    started = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_contracts()

    historical = historical_checks()
    author = author_code_smoke()
    raw6, independent6, control6 = claim6(SEED)
    dump_json(ARTIFACTS / "historical_baseline_checks.json", historical)
    dump_json(ARTIFACTS / "author_code_smoke.json", author)
    try:
        from reproduction.paper_order_sweep import run as run_paper_order_sweep
    except ImportError:
        paper_order_sweep = None
    else:
        paper_order_sweep = run_paper_order_sweep()
        dump_json(ARTIFACTS / "routes" / "paper_order_sweep.json", paper_order_sweep)
    try:
        from reproduction.derivative_calibration_audit import run as run_derivative_audit
    except ImportError:
        derivative_audit = None
    else:
        derivative_audit = run_derivative_audit()
        dump_json(
            ARTIFACTS / "routes" / "derivative_calibration_audit.json",
            derivative_audit,
        )
    try:
        from reproduction.full_network_composition import run as run_full_network
    except ImportError:
        full_network = None
    else:
        full_network = run_full_network()
        dump_json(
            ARTIFACTS / "routes" / "full_network_composition.json",
            full_network,
        )
    try:
        from reproduction.activation_condition_audit import run as run_activation_audit
    except ImportError:
        activation_audit = None
    else:
        activation_audit = run_activation_audit()
        dump_json(
            ARTIFACTS / "routes" / "activation_condition_audit.json",
            activation_audit,
        )
    try:
        from reproduction.theorem32_audit import run as run_theorem32_audit
    except ImportError:
        theorem32_audit = None
    else:
        theorem32_audit = run_theorem32_audit()
        dump_json(
            ARTIFACTS / "routes" / "theorem32_four_route_audit.json",
            theorem32_audit,
        )
    dump_json(ARTIFACTS / "claim_6" / "raw_results.json", raw6)
    dump_json(ARTIFACTS / "claim_6" / "independent_checker_output.json", independent6)
    dump_json(ARTIFACTS / "claim_6" / "negative_control_output.json", control6)
    candidate_verifier = subprocess.run(
        [os.fspath(Path(os.sys.executable)), os.fspath(ROOT / "space_candidate" / "code" / "verify_claim6.py")],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    dump_text(
        ARTIFACTS / "claim_6" / "evaluator_visible_verifier_output.txt",
        candidate_verifier.stdout + candidate_verifier.stderr,
    )
    if candidate_verifier.returncode != 0:
        raise AssertionError("Evaluator-visible Claim 6 verifier failed")
    dump_text(
        ARTIFACTS / "claim_6" / "method.md",
        "# Method\n\n"
        "Search deterministic finite binary64 triples for a grouping difference. "
        "Reconstruct each operand exactly as a rational number and verify that the "
        "corresponding real products are equal. A powers-of-two triple is the "
        "non-triggering control.",
    )
    dump_text(
        ARTIFACTS / "claim_6" / "source_audit.md",
        "# Source audit\n\n"
        "Paper Section 1.1; paper HTML SHA-256 "
        f"`{SOURCE_SHA256}`. The tested claim is the arithmetic mechanism, not "
        "the universal representation theorem.",
    )

    elapsed = time.perf_counter() - started
    metadata = {
        "git_sha": git_sha(),
        "fixed_command": FIXED_COMMAND,
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "estimated_required_cores": 1,
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "actual_logical_cpus_visible": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "runtime_seconds": elapsed,
    }
    if hasattr(os, "sched_getaffinity"):
        metadata["actual_cpu_affinity_count"] = len(os.sched_getaffinity(0))
    dump_json(ARTIFACTS / "environment_and_runtime.json", metadata)

    verdicts = {
        "claim_1": "BLOCKED",
        "claim_2": "BLOCKED",
        "claim_3": "BLOCKED",
        "claim_4": "BLOCKED",
        "claim_5": "BLOCKED",
        "claim_6": "VERIFIED",
    }
    dump_json(ARTIFACTS / "verdicts.json", verdicts)
    eval_md = f"""# Frozen baseline evaluation

| Claim | Verdict | Baseline basis |
| --- | --- | --- |
| 1 | BLOCKED | No universal >=9-layer construction yet |
| 2 | BLOCKED | No six-activation theorem audit yet |
| 3 | BLOCKED | Historical single-expression evidence is a proxy |
| 4 | BLOCKED | Historical product underflow is not Lemma 3.4 |
| 5 | BLOCKED | Historical cancellation is not Lemma 3.5 |
| 6 | VERIFIED | Binary64 grouping counterexample, exact-rational checker, and non-triggering control |

Author-code active-point smoke test: `{author["active_point_contract_pass"]}`.
Runtime: `{elapsed:.6f}` seconds. Git SHA: `{metadata["git_sha"]}`.
"""
    dump_text(ARTIFACTS / "EVAL.md", eval_md)

    print(eval_md)
    print("CLAIM6_RAW=" + json.dumps(raw6, sort_keys=True))
    print("CLAIM6_CHECKER=" + json.dumps(independent6, sort_keys=True))
    print("CLAIM6_CONTROL=" + json.dumps(control6, sort_keys=True))
    print("CLAIM6_EVALUATOR_VERIFIER=" + candidate_verifier.stdout.strip().replace("\n", " | "))
    print("AUTHOR_SMOKE=" + json.dumps(author, sort_keys=True))
    if paper_order_sweep is not None:
        compact_paper_order = {
            "evaluated": paper_order_sweep["evaluated"],
            "constructor_error_count": len(paper_order_sweep["constructor_errors"]),
            "paper_order_contracts": paper_order_sweep["paper_order_contracts"],
            "torch_matches_paper": paper_order_sweep["torch_matches_paper"],
            "destructive_control": paper_order_sweep["destructive_control"],
            "verdicts": paper_order_sweep["verdicts"],
            "limitation": paper_order_sweep["limitation"],
        }
        print("PAPER_ORDER_SWEEP=" + json.dumps(compact_paper_order, sort_keys=True))
    if derivative_audit is not None:
        compact_derivative_audit = {
            "evaluated": derivative_audit["evaluated"],
            "calibratable": derivative_audit["calibratable"],
            "unique_stored_actual_correction_tuples": derivative_audit[
                "unique_stored_actual_correction_tuples"
            ],
            "original_active_exact": derivative_audit["original_active_exact"],
            "calibrated_active_exact": derivative_audit["calibrated_active_exact"],
            "calibrated_off_exact": derivative_audit["calibrated_off_exact"],
            "negative_control": derivative_audit["negative_control"],
            "independent_checker": derivative_audit["independent_checker"],
            "verdicts": derivative_audit["verdicts"],
            "limitation": derivative_audit["limitation"],
        }
        print(
            "DERIVATIVE_CALIBRATION_AUDIT="
            + json.dumps(compact_derivative_audit, sort_keys=True)
        )
    if full_network is not None:
        compact_full_network = {
            "route": full_network["route"],
            "domain": full_network["domain"],
            "rows": full_network["rows"],
            "off_rows": full_network["off_rows"],
            "indicator_matrix": full_network["indicator_matrix"],
            "negative_control": full_network["negative_control"],
            "independent_checker": full_network["independent_checker"],
            "verdicts": full_network["verdicts"],
            "limitation": full_network["limitation"],
        }
        print(
            "FULL_NETWORK_COMPOSITION="
            + json.dumps(compact_full_network, sort_keys=True)
        )
    if activation_audit is not None:
        compact_activation_audit = {
            "route": activation_audit["route"],
            "format": activation_audit["format"],
            "rows": activation_audit["rows"],
            "negative_control": activation_audit["negative_control"],
            "independent_checker": activation_audit["independent_checker"],
            "verdicts": activation_audit["verdicts"],
            "limitation": activation_audit["limitation"],
        }
        print(
            "ACTIVATION_CONDITION_AUDIT="
            + json.dumps(compact_activation_audit, sort_keys=True)
        )
    if theorem32_audit is not None:
        compact_theorem32 = {
            "route": theorem32_audit["route"],
            "format": theorem32_audit["format"],
            "theorem_depth": theorem32_audit["theorem_depth"],
            "padding_layers": theorem32_audit["padding_layers"],
            "domain": theorem32_audit["domain"],
            "loss_derivatives": theorem32_audit["loss_derivatives"],
            "linear_exact": {
                "values": sum(row["value_exact"] for row in theorem32_audit["rows"]),
                "gradients": sum(
                    row["gradient_exact"] for row in theorem32_audit["rows"]
                ),
                "evaluated": len(theorem32_audit["rows"]),
            },
            "negative_control": theorem32_audit["negative_control"],
            "independent_checker": theorem32_audit["independent_checker"],
            "attempts": theorem32_audit["attempts"],
            "falsification_route": {
                key: theorem32_audit["falsification_route"][key]
                for key in (
                    "exact_claim",
                    "assumption_audit",
                    "candidate_mismatch_count",
                    "evaluated",
                    "valid_falsification",
                    "reason",
                )
            },
            "verdicts": theorem32_audit["verdicts"],
            "limitation": theorem32_audit["limitation"],
        }
        print(
            "THEOREM32_FOUR_ROUTE_AUDIT="
            + json.dumps(compact_theorem32, sort_keys=True)
        )
    print("ENVIRONMENT=" + json.dumps(metadata, sort_keys=True))

    # The author-code smoke test is diagnostic at this baseline: Claims 1--5
    # remain BLOCKED regardless of its outcome. Claim 6 and its independent
    # checker/control are the only accepted baseline evidence, and failures in
    # those checks already raise above.


if __name__ == "__main__":
    main()
