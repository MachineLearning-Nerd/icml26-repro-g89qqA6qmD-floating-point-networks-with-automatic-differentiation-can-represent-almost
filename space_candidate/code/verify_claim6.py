"""Standalone verifier for Claim 6 of arXiv 2605.01702.

Uses only the Python standard library and exits nonzero if the raw evidence,
exact-rational checker, or negative control fails.
"""

from __future__ import annotations

import json
import struct
from fractions import Fraction
from pathlib import Path


RAW = Path(__file__).resolve().parents[1] / "data" / "claim6_raw.json"


def bits(value: float) -> bytes:
    return struct.pack(">d", value)


def main() -> None:
    evidence = json.loads(RAW.read_text(encoding="utf-8"))
    triple = evidence["triple"]
    a, b, c = triple["a"], triple["b"], triple["c"]
    left = (a * b) * c
    right = a * (b * c)
    assert bits(left) != bits(right), "binary64 groupings unexpectedly match"
    assert bits(left) == bits(triple["left"]), "left raw value is not reproducible"
    assert bits(right) == bits(triple["right"]), "right raw value is not reproducible"

    exact_left = (Fraction.from_float(a) * Fraction.from_float(b)) * Fraction.from_float(c)
    exact_right = Fraction.from_float(a) * (Fraction.from_float(b) * Fraction.from_float(c))
    assert exact_left == exact_right, "exact-real associativity checker failed"

    control = evidence["negative_control"]
    ca, cb, cc = control["operands"]
    control_left = (ca * cb) * cc
    control_right = ca * (cb * cc)
    assert bits(control_left) == bits(control_right), "negative control fired"
    assert control_left == control["left"] == control["right"], "control raw values differ"

    print("claim=6 verdict=VERIFIED")
    print("float_bitwise_difference=true")
    print("exact_rational_products_equal=true")
    print("nonassociativity_detector_fires=false expected=false control_pass=true")


if __name__ == "__main__":
    main()
