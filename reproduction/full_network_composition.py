"""Finite-domain composition of the full Lemma 3.4/3.5 ReLU blocks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from reproduction.paper_order import (
    evaluate_exact_indicator,
    evaluate_grad,
    evaluate_zero_grad,
    f32,
    left_sum,
)
from src.models import GradIndicator, ZeroGradIndicator


SEED = 260501706


def _bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def _exact(value: float, expected: float) -> bool:
    return _bits(value) == _bits(expected)


def _calibrate_positive(module: GradIndicator, active_x: float) -> dict[str, float]:
    indicator_output, actual_derivative = evaluate_exact_indicator(module.indicator, active_x)
    stored_derivative = float(module.indicator.diff.detach().item())
    if not _exact(indicator_output, 1.0) or actual_derivative == 0.0:
        raise AssertionError("selected domain point is not calibratable")
    correction = float(f32(stored_derivative / actual_derivative))
    with torch.no_grad():
        module.linear.weight.mul_(correction)
        module.zerograd_indicator.linear.weight.mul_(correction)
    return {
        "indicator_output": indicator_output,
        "stored_derivative": stored_derivative,
        "actual_derivative": actual_derivative,
        "correction": correction,
    }


@dataclass
class ValueBranch:
    module: ZeroGradIndicator


@dataclass
class GradientBranch:
    module: GradIndicator
    sign: float


class FiniteDomainNetwork:
    """Parallel 11/12-layer branches followed by one affine sum (depth 13)."""

    depth = 13

    def __init__(
        self,
        domain: list[float],
        function_values: list[float],
        upstream_values: list[float],
        target_gradients: list[float],
    ) -> None:
        self.domain = [float(f32(value)) for value in domain]
        self.value_branches: list[ValueBranch] = []
        self.gradient_branches: list[GradientBranch] = []
        self.calibrations: list[dict[str, float]] = []
        for x, target_value, upstream, target_gradient in zip(
            self.domain,
            function_values,
            upstream_values,
            target_gradients,
            strict=True,
        ):
            z = torch.tensor([x], dtype=torch.float32)
            value = torch.tensor([target_value], dtype=torch.float32)
            self.value_branches.append(
                ValueBranch(ZeroGradIndicator(z, value, dtype=torch.float32, device="cpu"))
            )
            magnitude = abs(float(f32(target_gradient / upstream)))
            gradient_module = GradIndicator(
                z,
                torch.tensor([magnitude], dtype=torch.float32),
                dtype=torch.float32,
                device="cpu",
            )
            self.calibrations.append(_calibrate_positive(gradient_module, x))
            self.gradient_branches.append(
                GradientBranch(
                    module=gradient_module,
                    sign=-1.0 if target_gradient < 0.0 else 1.0,
                )
            )

    def evaluate(self, x: float, upstream: float) -> tuple[float, float]:
        outputs: list[np.float32] = []
        gradients: list[np.float32] = []
        for branch in self.value_branches:
            output, gradient = evaluate_zero_grad(branch.module, x, upstream)
            outputs.append(f32(output))
            gradients.append(f32(gradient))
        for branch in self.gradient_branches:
            output, gradient = evaluate_grad(branch.module, x, upstream)
            outputs.append(f32(branch.sign * output))
            gradients.append(f32(branch.sign * gradient))
        return float(left_sum(np.asarray(outputs))), float(left_sum(np.asarray(gradients)))

    def indicator_matrix(self) -> list[list[float]]:
        matrix: list[list[float]] = []
        for x in self.domain:
            row = []
            for branch in self.value_branches:
                indicator = branch.module.indicator
                output, _ = evaluate_exact_indicator(indicator, x)
                row.append(output)
            matrix.append(row)
        return matrix


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    domain = [2.0**-20, 0.25, 0.5, 1.0, 2.0, 8.0]
    function_values = [0.25, -0.5, 1.0, -2.0, 4.0, -8.0]
    upstream_values = [0.5, 1.0, 2.0, 4.0, 0.25, 8.0]
    target_gradients = [1.0, -2.0, 0.5, -1.0, 4.0, -0.25]
    network = FiniteDomainNetwork(
        domain, function_values, upstream_values, target_gradients
    )
    rows = []
    for x, target_value, upstream, target_gradient in zip(
        domain,
        function_values,
        upstream_values,
        target_gradients,
        strict=True,
    ):
        observed_value, observed_gradient = network.evaluate(x, upstream)
        rows.append(
            {
                "x": float(f32(x)),
                "target_value": float(f32(target_value)),
                "upstream": float(f32(upstream)),
                "target_gradient": float(f32(target_gradient)),
                "observed_value": observed_value,
                "observed_gradient": observed_gradient,
                "value_exact": _exact(observed_value, target_value),
                "gradient_exact": _exact(observed_gradient, target_gradient),
            }
        )
    off_rows = []
    for x in domain:
        off = float(np.nextafter(f32(x), f32(math.inf), dtype=np.float32))
        value, gradient = network.evaluate(off, 1.0)
        off_rows.append(
            {
                "source_x": float(f32(x)),
                "off_x": off,
                "observed_value": value,
                "observed_gradient": gradient,
                "zero_zero_exact": _exact(value, 0.0) and _exact(gradient, 0.0),
            }
        )
    indicator_matrix = network.indicator_matrix()
    identity_exact = all(
        _exact(value, 1.0 if row == column else 0.0)
        for row, values in enumerate(indicator_matrix)
        for column, value in enumerate(values)
    )

    shuffled = FiniteDomainNetwork(
        domain,
        function_values,
        upstream_values,
        target_gradients[1:] + target_gradients[:1],
    )
    shuffled_rows = [
        shuffled.evaluate(x, upstream)[1]
        for x, upstream in zip(domain, upstream_values, strict=True)
    ]
    negative_control = {
        "name": "rotate target-gradient assignments between indicator branches",
        "expected": "at least one exact target-gradient mismatch",
        "observed_gradients": shuffled_rows,
        "mismatch_count": sum(
            not _exact(observed, expected)
            for observed, expected in zip(
                shuffled_rows, target_gradients, strict=True
            )
        ),
    }
    direct_checker = {
        "network_depth": network.depth,
        "depth_at_least_9": network.depth >= 9,
        "domain_size": len(domain),
        "all_values_exact": all(row["value_exact"] for row in rows),
        "all_gradients_exact": all(row["gradient_exact"] for row in rows),
        "all_off_domain_controls_zero": all(row["zero_zero_exact"] for row in off_rows),
        "indicator_matrix_identity_exact": identity_exact,
        "negative_control_fires": negative_control["mismatch_count"] > 0,
    }
    if not all(direct_checker.values()):
        raise AssertionError("finite-domain full-network contract or control failed")
    return {
        "route": "full finite-domain 13-layer ReLU network composition",
        "seed": SEED,
        "domain": [float(f32(value)) for value in domain],
        "function_values": function_values,
        "upstream_values": upstream_values,
        "target_gradients": target_gradients,
        "rows": rows,
        "off_rows": off_rows,
        "indicator_matrix": indicator_matrix,
        "calibrations": network.calibrations,
        "negative_control": negative_control,
        "independent_checker": direct_checker,
        "verdicts": {
            "claim_1": "BLOCKED",
            "claim_4": "BLOCKED",
            "claim_5": "BLOCKED",
        },
        "limitation": (
            "This directly reconstructs a multi-layer network for arbitrary selected maps "
            "and exhausts its declared six-point domain, but it is not a universal proof "
            "over every float32 point, dimension, map, and admissible activation."
        ),
    }
