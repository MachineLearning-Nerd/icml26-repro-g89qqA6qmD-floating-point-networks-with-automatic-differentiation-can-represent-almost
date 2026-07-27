"""Audit and independently correct the released exact-indicator calibration."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

from reproduction.paper_order import evaluate_exact_indicator, evaluate_grad
from src.models import GradIndicator


SEED = 260501705


def _bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def _exact_pair(pair: tuple[float, float], expected: tuple[float, float]) -> bool:
    return _bits(pair[0]) == _bits(expected[0]) and _bits(pair[1]) == _bits(expected[1])


def _calibrate(module: GradIndicator, active_x: float) -> dict[str, float | int | bool]:
    indicator_output, actual_derivative = evaluate_exact_indicator(module.indicator, active_x)
    stored_derivative = float(module.indicator.diff.detach().item())
    calibratable = actual_derivative != 0.0 and math.isfinite(actual_derivative)
    if not calibratable:
        return {
            "indicator_output": indicator_output,
            "stored_derivative": stored_derivative,
            "actual_derivative": actual_derivative,
            "stored_bits": _bits(stored_derivative),
            "actual_bits": _bits(actual_derivative),
            "normalization_correction": 0.0,
            "correction_bits": _bits(0.0),
            "calibratable": False,
        }
    correction = float(np.float32(stored_derivative / actual_derivative))
    with torch.no_grad():
        module.linear.weight.mul_(correction)
        module.zerograd_indicator.linear.weight.mul_(correction)
    return {
        "indicator_output": indicator_output,
        "stored_derivative": stored_derivative,
        "actual_derivative": actual_derivative,
        "stored_bits": _bits(stored_derivative),
        "actual_bits": _bits(actual_derivative),
        "normalization_correction": correction,
        "correction_bits": _bits(correction),
        "calibratable": True,
    }


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
    for point in points:
        z = torch.tensor([point], dtype=torch.float32)
        off = float(
            torch.nextafter(z, torch.tensor([math.inf], dtype=torch.float32))[0].item()
        )
        for target in targets:
            value = torch.tensor([target], dtype=torch.float32)
            original = GradIndicator(z, value, dtype=torch.float32, device="cpu")
            original_active = evaluate_grad(original, point)
            calibrated = GradIndicator(z, value, dtype=torch.float32, device="cpu")
            calibration = _calibrate(calibrated, point)
            if calibration["calibratable"]:
                calibrated_active = evaluate_grad(calibrated, point)
                calibrated_off = evaluate_grad(calibrated, off)
            else:
                calibrated_active = None
                calibrated_off = None
            rows.append(
                {
                    "z": float(np.float32(point)),
                    "target": target,
                    "off_x": off,
                    "calibration": calibration,
                    "original_active": original_active,
                    "calibrated_active": calibrated_active,
                    "calibrated_off": calibrated_off,
                    "original_active_contract": _exact_pair(original_active, (0.0, target)),
                    "calibrated_active_contract": calibrated_active is not None
                    and _exact_pair(calibrated_active, (0.0, target)),
                    "calibrated_off_contract": calibrated_off is not None
                    and _exact_pair(calibrated_off, (0.0, 0.0)),
                }
            )

    evaluated = len(rows)
    original_hits = sum(row["original_active_contract"] for row in rows)
    active_hits = sum(row["calibrated_active_contract"] for row in rows)
    off_hits = sum(row["calibrated_off_contract"] for row in rows)
    ratios = sorted(
        {
            (
                row["calibration"]["stored_derivative"],
                row["calibration"]["actual_derivative"],
                row["calibration"]["normalization_correction"],
            )
            for row in rows
            if row["calibration"]["calibratable"]
        }
    )
    calibratable = sum(row["calibration"]["calibratable"] for row in rows)
    negative_control = {
        "name": "released uncalibrated normalization",
        "expected": "at least one active target-gradient contract failure",
        "active_exact": original_hits,
        "evaluated": evaluated,
        "fires": original_hits < evaluated,
    }
    independent_checker = {
        "all_indicator_outputs_one": all(
            _bits(row["calibration"]["indicator_output"]) == _bits(1.0) for row in rows
        ),
        "all_calibratable_active_exact": active_hits == calibratable,
        "all_calibratable_off_exact": off_hits == calibratable,
        "normalization_identity_bitwise": all(
            _bits(
                float(
                    np.float32(
                        row["calibration"]["actual_derivative"]
                        * row["calibration"]["normalization_correction"]
                    )
                )
            )
            == _bits(row["calibration"]["stored_derivative"])
            for row in rows
            if row["calibration"]["calibratable"]
        ),
    }
    if not negative_control["fires"]:
        raise AssertionError("wrong-calibration negative control did not fail")
    return {
        "route": "clean-room derivative calibration of released GradIndicator algebra",
        "seed": SEED,
        "evaluated": evaluated,
        "calibratable": calibratable,
        "unique_stored_actual_correction_tuples": ratios,
        "original_active_exact": original_hits,
        "calibrated_active_exact": active_hits,
        "calibrated_off_exact": off_hits,
        "negative_control": negative_control,
        "independent_checker": independent_checker,
        "rows": rows,
        "verdicts": {"claim_5": "BLOCKED"},
        "limitation": (
            "The calibration directly exercises the full released ReLU architecture at "
            "finite points. It repairs an implementation constant but does not by itself "
            "certify the universally quantified Lemma 3.5 construction."
        ),
    }
