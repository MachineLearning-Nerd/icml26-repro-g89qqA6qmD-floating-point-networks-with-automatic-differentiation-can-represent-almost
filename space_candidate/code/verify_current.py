"""Standalone cumulative verifier for arXiv 2605.01702.

The verifier uses only the Python standard library. It checks the immutable
raw JSON summaries produced by formal Hugging Face CPU runs and exits nonzero
if any accepted result, independent check, or negative control regresses.
BLOCKED is an intentional theorem-level verdict, never a failed assertion.
"""

from __future__ import annotations

import json
import struct
from fractions import Fraction
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def bits64(value: float) -> bytes:
    return struct.pack(">d", value)


def verify_claim_1() -> None:
    raw = load("claim1_full_network.json")
    assert raw["verdict"] == "BLOCKED"
    assert raw["network_depth"] >= 9
    assert len(raw["rows"]) == len(raw["domain"]) == 6
    assert all(row["value_exact"] and row["gradient_exact"] for row in raw["rows"])
    assert raw["off_domain_controls_zero"] == 6
    assert raw["indicator_matrix_identity_exact"]
    assert raw["negative_control"]["mismatch_count"] == 6
    assert raw["negative_control"]["fires"]


def verify_claim_2() -> None:
    raw = load("claim2_activations.json")
    expected = {"ReLU", "ELU", "GELU", "Swish", "Sigmoid", "tanh"}
    assert raw["verdict"] == "BLOCKED"
    assert {row["activation"] for row in raw["rows"]} == expected
    assert all(all(row["condition2_bullets"]) for row in raw["rows"])
    assert all(row["condition3_pass"] for row in raw["rows"])
    assert all(row["analytic_autograd_agree"] for row in raw["rows"])
    assert raw["negative_control"]["rejected"]


def verify_claim_3() -> None:
    raw = load("claim3_theorem32.json")
    linear = raw["linear_antisymmetric_family"]
    assert raw["verdict"] == "BLOCKED"
    assert raw["theorem_depth"] == 2 ** (8 + 1) + 2 * 23 + 11 == 569
    assert raw["executed_padding_layers"] == 556
    assert linear == {"values_exact": 60, "gradients_exact": 60, "evaluated": 60}
    assert raw["negative_control"]["antisymmetry_violations"] == 30
    assert raw["negative_control"]["rejected"]
    assert len(raw["attempts"]) == 4
    assert raw["falsification_route"]["target_antisymmetric"]
    assert not raw["falsification_route"]["valid_falsification"]


def verify_claim_4() -> None:
    raw = load("claim4_zero_gradient.json")
    assert raw["verdict"] == "BLOCKED"
    assert raw["evaluated"] == 48
    assert raw["active_value_and_zero_gradient_exact"] == 48
    assert raw["off_point_zero_zero_exact"] == 48
    assert raw["composition_network_depth"] >= 9
    assert raw["negative_control"]["target_gradient_lost"]


def verify_claim_5() -> None:
    raw = load("claim5_zero_output_gradient.json")
    assert raw["verdict"] == "BLOCKED"
    assert raw["released_normalization_audit"]["active_exact"] == 0
    assert raw["calibration"]["active_zero_output_target_gradient_exact"] == 24
    assert raw["calibration"]["off_zero_zero_exact"] == 24
    assert raw["calibration"]["correction"] == 0.5
    assert raw["full_composition"]["values_exact"] == 6
    assert raw["full_composition"]["gradients_exact"] == 6
    assert raw["negative_control"]["fires"]


def verify_claim_6() -> None:
    raw = load("claim6_raw.json")
    triple = raw["triple"]
    a, b, c = triple["a"], triple["b"], triple["c"]
    left, right = (a * b) * c, a * (b * c)
    assert raw["verdict"] == "VERIFIED"
    assert bits64(left) != bits64(right)
    assert bits64(left) == bits64(triple["left"])
    assert bits64(right) == bits64(triple["right"])
    exact_left = (Fraction.from_float(a) * Fraction.from_float(b)) * Fraction.from_float(c)
    exact_right = Fraction.from_float(a) * (Fraction.from_float(b) * Fraction.from_float(c))
    assert exact_left == exact_right
    control = raw["negative_control"]
    ca, cb, cc = control["operands"]
    assert bits64((ca * cb) * cc) == bits64(ca * (cb * cc))
    assert control["pass"] and not control["nonassociativity_detector_fires"]


def main() -> None:
    checks = [
        verify_claim_1,
        verify_claim_2,
        verify_claim_3,
        verify_claim_4,
        verify_claim_5,
        verify_claim_6,
    ]
    for number, check in enumerate(checks, start=1):
        check()
        verdict = "VERIFIED" if number == 6 else "BLOCKED"
        print(f"claim={number} evidence_check=PASS theorem_verdict={verdict}")
    print("cumulative_regression=PASS controls=PASS")


if __name__ == "__main__":
    main()
