"""Deterministic contract audit of the authors' pinned Torch modules."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models import GradIndicator, ZeroGradIndicator


SEED = 260501703


def _evaluate(module: nn.Module, point: float) -> tuple[float, float]:
    x = torch.tensor([point], dtype=torch.float32, requires_grad=True)
    output = module(x)
    gradient = torch.autograd.grad(output, x)[0]
    return float(output.detach().item()), float(gradient.detach().item())


def _bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def _active_equal(output: float, expected_output: float, gradient: float, expected_gradient: float) -> bool:
    return _bits(output) == _bits(expected_output) and _bits(gradient) == _bits(expected_gradient)


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    rng = np.random.default_rng(SEED)

    structured = [
        -8.0,
        -2.0,
        -1.0,
        -0.5,
        -0.25,
        -2.0**-20,
        2.0**-20,
        0.25,
        0.5,
        1.0,
        2.0,
        8.0,
    ]
    random_points = rng.uniform(-8.0, 8.0, size=52).astype(np.float32).tolist()
    points = [float(np.float32(x)) for x in structured + random_points if x != 0.0]
    targets = [0.25, 0.5, 1.0, 2.0]

    zero_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    constructor_errors: list[dict[str, Any]] = []
    for point in points:
        z = torch.tensor([point], dtype=torch.float32)
        off = float(torch.nextafter(z, torch.tensor([math.inf], dtype=torch.float32))[0].item())
        for target in targets:
            value = torch.tensor([target], dtype=torch.float32)
            try:
                zero_module = ZeroGradIndicator(z, value, dtype=torch.float32, device="cpu")
                active_output, active_gradient = _evaluate(zero_module, point)
                off_output, off_gradient = _evaluate(zero_module, off)
                zero_rows.append(
                    {
                        "z": point,
                        "target_value": target,
                        "active_output": active_output,
                        "active_gradient": active_gradient,
                        "off_x": off,
                        "off_output": off_output,
                        "off_gradient": off_gradient,
                        "active_contract": _active_equal(active_output, target, active_gradient, 0.0),
                        "off_contract": _active_equal(off_output, 0.0, off_gradient, 0.0),
                        "affine_module_count": sum(
                            isinstance(module, nn.Linear) for module in zero_module.modules()
                        ),
                        "documented_max_path_layers": 11,
                    }
                )

                grad_module = GradIndicator(z, value, dtype=torch.float32, device="cpu")
                active_output, active_gradient = _evaluate(grad_module, point)
                off_output, off_gradient = _evaluate(grad_module, off)
                grad_rows.append(
                    {
                        "z": point,
                        "target_gradient": target,
                        "active_output": active_output,
                        "active_gradient": active_gradient,
                        "off_x": off,
                        "off_output": off_output,
                        "off_gradient": off_gradient,
                        "active_contract": _active_equal(active_output, 0.0, active_gradient, target),
                        "off_contract": _active_equal(off_output, 0.0, off_gradient, 0.0),
                        "affine_module_count": sum(
                            isinstance(module, nn.Linear) for module in grad_module.modules()
                        ),
                    }
                )
            except (RuntimeError, ValueError, OverflowError) as error:
                constructor_errors.append(
                    {"z": point, "target": target, "type": type(error).__name__, "message": str(error)}
                )

    control_z = torch.tensor([0.25], dtype=torch.float32)
    control_target = torch.tensor([1.0], dtype=torch.float32)
    control_module = GradIndicator(control_z, control_target, dtype=torch.float32, device="cpu")
    original_output, original_gradient = _evaluate(control_module, 0.25)
    with torch.no_grad():
        control_module.linear2.weight.zero_()
    perturbed_output, perturbed_gradient = _evaluate(control_module, 0.25)
    destructive_control = {
        "change": "set GradIndicator target-gradient linear weight to exactly zero",
        "original_output": original_output,
        "original_gradient": original_gradient,
        "perturbed_output": perturbed_output,
        "perturbed_gradient": perturbed_gradient,
        "target_gradient": 1.0,
        "target_lost_after_perturbation": _bits(perturbed_gradient) != _bits(1.0),
    }
    if not destructive_control["target_lost_after_perturbation"]:
        raise AssertionError("destructive gradient-path control did not lose the target")

    summary = {
        "route": "pinned author Torch implementation contract sweep",
        "seed": SEED,
        "dtype": "float32",
        "points": len(points),
        "targets_per_point": len(targets),
        "attempted_constructions_per_module": len(points) * len(targets),
        "zero_grad_indicator": {
            "active_exact": sum(row["active_contract"] for row in zero_rows),
            "off_exact": sum(row["off_contract"] for row in zero_rows),
            "evaluated": len(zero_rows),
        },
        "grad_indicator": {
            "active_exact": sum(row["active_contract"] for row in grad_rows),
            "off_exact": sum(row["off_contract"] for row in grad_rows),
            "evaluated": len(grad_rows),
        },
        "constructor_errors": constructor_errors,
        "destructive_control": destructive_control,
        "zero_grad_rows": zero_rows,
        "grad_rows": grad_rows,
        "verdicts": {
            "claim_4": "BLOCKED",
            "claim_5": "BLOCKED",
        },
        "limitation": (
            "A deterministic finite sweep diagnoses the released implementation but cannot "
            "establish the paper's universal quantifiers. Exact failures are evidence about "
            "this code/environment pair, not a falsification of the theorem."
        ),
    }
    return summary
