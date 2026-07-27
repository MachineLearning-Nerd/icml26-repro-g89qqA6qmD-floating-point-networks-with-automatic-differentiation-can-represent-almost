"""Compare Torch execution with the paper-order interpreter."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

from reproduction.paper_order import evaluate_grad, evaluate_zero_grad
from src.models import GradIndicator, ZeroGradIndicator


SEED = 260501704


def _torch_evaluate(module: torch.nn.Module, point: float) -> tuple[float, float]:
    x = torch.tensor([point], dtype=torch.float32, requires_grad=True)
    y = module(x)
    gradient = torch.autograd.grad(y, x)[0]
    return float(y.detach().item()), float(gradient.detach().item())


def _bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def _exact_pair(pair: tuple[float, float], expected: tuple[float, float]) -> bool:
    return _bits(pair[0]) == _bits(expected[0]) and _bits(pair[1]) == _bits(expected[1])


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    points = [
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
    targets = [0.25, 0.5, 1.0, 2.0]
    rows: list[dict[str, Any]] = []
    constructor_errors: list[dict[str, Any]] = []
    for point in points:
        z = torch.tensor([point], dtype=torch.float32)
        off = float(torch.nextafter(z, torch.tensor([math.inf], dtype=torch.float32))[0].item())
        for target in targets:
            value = torch.tensor([target], dtype=torch.float32)
            try:
                zero = ZeroGradIndicator(z, value, dtype=torch.float32, device="cpu")
                grad = GradIndicator(z, value, dtype=torch.float32, device="cpu")
                paper_zero_active = evaluate_zero_grad(zero, point)
                paper_zero_off = evaluate_zero_grad(zero, off)
                paper_grad_active = evaluate_grad(grad, point)
                paper_grad_off = evaluate_grad(grad, off)
                torch_zero_active = _torch_evaluate(zero, point)
                torch_grad_active = _torch_evaluate(grad, point)
                rows.append(
                    {
                        "z": float(np.float32(point)),
                        "target": target,
                        "off_x": off,
                        "paper_zero_active": paper_zero_active,
                        "paper_zero_off": paper_zero_off,
                        "paper_grad_active": paper_grad_active,
                        "paper_grad_off": paper_grad_off,
                        "torch_zero_active": torch_zero_active,
                        "torch_grad_active": torch_grad_active,
                        "paper_zero_active_contract": _exact_pair(paper_zero_active, (target, 0.0)),
                        "paper_zero_off_contract": _exact_pair(paper_zero_off, (0.0, 0.0)),
                        "paper_grad_active_contract": _exact_pair(paper_grad_active, (0.0, target)),
                        "paper_grad_off_contract": _exact_pair(paper_grad_off, (0.0, 0.0)),
                        "torch_and_paper_zero_match": _exact_pair(
                            torch_zero_active, paper_zero_active
                        ),
                        "torch_and_paper_grad_match": _exact_pair(
                            torch_grad_active, paper_grad_active
                        ),
                    }
                )
            except (RuntimeError, ValueError, OverflowError) as error:
                constructor_errors.append(
                    {"z": point, "target": target, "type": type(error).__name__, "message": str(error)}
                )

    control_z = torch.tensor([0.25], dtype=torch.float32)
    control_target = torch.tensor([1.0], dtype=torch.float32)
    control = GradIndicator(control_z, control_target, dtype=torch.float32, device="cpu")
    original = evaluate_grad(control, 0.25)
    with torch.no_grad():
        control.linear2.weight.zero_()
    perturbed = evaluate_grad(control, 0.25)
    destructive_control = {
        "change": "zero target-gradient linear weight",
        "original": original,
        "perturbed": perturbed,
        "target_gradient_lost": _bits(perturbed[1]) != _bits(1.0),
    }
    if not destructive_control["target_gradient_lost"]:
        raise AssertionError("paper-order destructive control did not lose the target")

    evaluated = len(rows)
    summary = {
        "route": "explicit paper-order float32 interpreter over pinned author weights",
        "seed": SEED,
        "points": len(points),
        "targets_per_point": len(targets),
        "evaluated": evaluated,
        "constructor_errors": constructor_errors,
        "paper_order_contracts": {
            "zero_active_exact": sum(row["paper_zero_active_contract"] for row in rows),
            "zero_off_exact": sum(row["paper_zero_off_contract"] for row in rows),
            "grad_active_exact": sum(row["paper_grad_active_contract"] for row in rows),
            "grad_off_exact": sum(row["paper_grad_off_contract"] for row in rows),
        },
        "torch_matches_paper": {
            "zero_active_exact": sum(row["torch_and_paper_zero_match"] for row in rows),
            "grad_active_exact": sum(row["torch_and_paper_grad_match"] for row in rows),
        },
        "destructive_control": destructive_control,
        "rows": rows,
        "verdicts": {"claim_4": "BLOCKED", "claim_5": "BLOCKED"},
        "limitation": (
            "This is a finite architecture audit, not a certificate for arbitrary functions "
            "on the theorem's complete domain. Failures of released weights are not theorem "
            "falsifications unless all paper assumptions and the full construction are met."
        ),
    }
    return summary
