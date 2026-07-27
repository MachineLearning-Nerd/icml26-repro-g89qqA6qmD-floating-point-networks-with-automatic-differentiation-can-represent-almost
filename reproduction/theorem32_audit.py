"""Four-route audit of Theorem 3.2's loss-derivative quantifier."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

from reproduction.full_network_composition import FiniteDomainNetwork


SEED = 260501708
P = 23
Q = 8
THEOREM_DEPTH = 2 ** (Q + 1) + 2 * P + 11


def _f32(value: float) -> float:
    return float(np.float32(value))


def _bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def _exact(value: float, expected: float) -> bool:
    return _bits(value) == _bits(expected)


class PaddedLossDerivativeNetwork:
    """A fixed 569-layer ReLU network on the declared positive input domain."""

    def __init__(
        self,
        domain: list[float],
        function_values: list[float],
        slopes: list[float],
    ) -> None:
        self.base = FiniteDomainNetwork(
            domain=domain,
            function_values=function_values,
            upstream_values=[1.0] * len(domain),
            target_gradients=slopes,
        )
        self.padding_layers = THEOREM_DEPTH - self.base.depth
        if self.padding_layers < 0:
            raise AssertionError("base network exceeds theorem depth")
        self.depth = self.base.depth + self.padding_layers

    def evaluate(self, x: float, loss_derivative: float) -> tuple[float, float]:
        value = np.float32(x)
        masks = []
        for _ in range(self.padding_layers):
            masks.append(bool(value > np.float32(0.0)))
            value = np.maximum(value, np.float32(0.0)).astype(np.float32)
        output, gradient = self.base.evaluate(float(value), loss_derivative)
        backward = np.float32(gradient)
        for active in reversed(masks):
            backward = np.float32(backward if active else 0.0)
        return output, float(backward)


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    domain = [2.0**-20, 0.25, 0.5, 1.0, 2.0, 8.0]
    function_values = [0.25, -0.5, 1.0, -2.0, 4.0, -8.0]
    slopes = [1.0, -2.0, 0.5, -1.0, 4.0, -0.25]
    magnitudes = [0.25, 0.5, 1.0, 2.0, 4.0]
    loss_derivatives = [-value for value in reversed(magnitudes)] + magnitudes
    network = PaddedLossDerivativeNetwork(domain, function_values, slopes)
    rows = []
    for x, target_value, slope in zip(
        domain, function_values, slopes, strict=True
    ):
        for loss_derivative in loss_derivatives:
            output, gradient = network.evaluate(x, loss_derivative)
            target_gradient = _f32(slope * loss_derivative)
            rows.append(
                {
                    "x": _f32(x),
                    "loss_derivative": _f32(loss_derivative),
                    "target_value": _f32(target_value),
                    "target_gradient": target_gradient,
                    "observed_value": output,
                    "observed_gradient": gradient,
                    "value_exact": _exact(output, target_value),
                    "gradient_exact": _exact(gradient, target_gradient),
                }
            )

    antisymmetry_rows = []
    for x, slope in zip(domain, slopes, strict=True):
        for magnitude in magnitudes:
            positive = _f32(slope * magnitude)
            negative = _f32(slope * -magnitude)
            antisymmetry_rows.append(
                {
                    "x": _f32(x),
                    "magnitude": magnitude,
                    "g_positive": positive,
                    "g_negative": negative,
                    "bit_exact": _exact(negative, -positive),
                }
            )

    even_control_violations = 0
    for slope in slopes:
        for magnitude in magnitudes:
            positive = _f32(slope * magnitude)
            negative = _f32(slope * magnitude)
            even_control_violations += not _exact(negative, -positive)
    negative_control = {
        "name": "even target g(x,y)=k(x)|y|",
        "expected": "antisymmetry rejection",
        "antisymmetry_violation_count": even_control_violations,
        "rejected": even_control_violations > 0,
    }

    nonlinear_rows = []
    nonlinear_mismatches = 0
    for x, slope in zip(domain, slopes, strict=True):
        for loss_derivative in loss_derivatives:
            _, observed = network.evaluate(x, loss_derivative)
            target = _f32(
                math.copysign(
                    abs(slope) * abs(loss_derivative) * abs(loss_derivative),
                    slope * loss_derivative,
                )
            )
            mismatch = not _exact(observed, target)
            nonlinear_mismatches += mismatch
            nonlinear_rows.append(
                {
                    "x": _f32(x),
                    "loss_derivative": _f32(loss_derivative),
                    "target_nonlinear_odd": target,
                    "observed_candidate": observed,
                    "mismatch": mismatch,
                }
            )
    nonlinear_antisymmetric = all(
        _exact(
            _f32(
                math.copysign(
                    abs(slope) * magnitude * magnitude,
                    -slope,
                )
            ),
            -_f32(
                math.copysign(
                    abs(slope) * magnitude * magnitude,
                    slope,
                )
            ),
        )
        for slope in slopes
        for magnitude in magnitudes
    )
    falsification_route = {
        "exact_claim": (
            "For every antisymmetric g*(x,y), Theorem 3.2 asserts existence of an "
            "L-layer network for every L>=569 in float32 (q=8,p=23)."
        ),
        "assumption_audit": {
            "q_at_least_6": Q >= 6,
            "bounded_positive_x_domain": True,
            "bounded_loss_derivatives": True,
            "nonlinear_target_antisymmetric": nonlinear_antisymmetric,
        },
        "candidate_mismatch_count": nonlinear_mismatches,
        "evaluated": len(nonlinear_rows),
        "valid_falsification": False,
        "reason": (
            "The nonlinear odd target satisfies the tested assumptions and this particular "
            "569-layer candidate misses it, but the theorem is existential over all "
            "networks. No exhaustive network-space or proof-level impossibility argument "
            "was available, so this is not a counterexample to Theorem 3.2."
        ),
        "rows": nonlinear_rows,
    }
    checker = {
        "exact_theorem_depth": network.depth == THEOREM_DEPTH,
        "padding_layers_executed": network.padding_layers,
        "all_linear_target_values_exact": all(row["value_exact"] for row in rows),
        "all_linear_target_gradients_exact": all(
            row["gradient_exact"] for row in rows
        ),
        "all_linear_targets_antisymmetric": all(
            row["bit_exact"] for row in antisymmetry_rows
        ),
        "even_target_control_rejected": negative_control["rejected"],
        "nonlinear_falsification_target_antisymmetric": nonlinear_antisymmetric,
        "nonlinear_candidate_misses_target": nonlinear_mismatches > 0,
    }
    if not all(
        value if isinstance(value, bool) else value > 0 for value in checker.values()
    ):
        raise AssertionError("Theorem 3.2 route checker or control failed")
    attempts = [
        {
            "route": 1,
            "method": "historical BIG-cancellation scalar expression",
            "result": "mechanism reproduced but rejected as a one-expression toy",
        },
        {
            "route": 2,
            "method": "single executed 569-layer ReLU network",
            "result": (
                f"{sum(row['gradient_exact'] for row in rows)}/{len(rows)} exact "
                "linear antisymmetric gradients with preserved values"
            ),
        },
        {
            "route": 3,
            "method": "exact source quantifier and construction-gap audit",
            "result": (
                "the arbitrary-odd lookup component is not present in the released code; "
                "the linear family cannot establish the universal target quantifier"
            ),
        },
        {
            "route": 4,
            "method": "dedicated nonlinear-odd falsification attempt",
            "result": falsification_route["reason"],
        },
    ]
    return {
        "route": "Theorem 3.2 four-route audit",
        "seed": SEED,
        "format": {"p": P, "q": Q},
        "theorem_depth": THEOREM_DEPTH,
        "padding_layers": network.padding_layers,
        "domain": [_f32(value) for value in domain],
        "loss_derivatives": loss_derivatives,
        "rows": rows,
        "antisymmetry_rows": antisymmetry_rows,
        "negative_control": negative_control,
        "independent_checker": checker,
        "attempts": attempts,
        "falsification_route": falsification_route,
        "verdicts": {"claim_3": "BLOCKED"},
        "limitation": (
            "A full-depth fixed network exactly realizes a nontrivial linear antisymmetric "
            "family, but arbitrary antisymmetric dependence is not certified. Four distinct "
            "routes, including falsification, therefore end in BLOCKED."
        ),
    }
