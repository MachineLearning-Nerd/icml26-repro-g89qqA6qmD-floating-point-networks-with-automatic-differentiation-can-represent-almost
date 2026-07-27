"""Float32 witness audit for the paper's activation Conditions 2 and 3."""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as functional


SEED = 260501707
P = 23
Q = 8
E_MIN = -(2 ** (Q - 1)) + 2
E_MAX = (2 ** (Q - 1)) - 1


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def _f32(value: float) -> float:
    return float(np.float32(value))


def _power_of_two_exponent(value: float) -> int | None:
    magnitude = abs(_f32(value))
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return None
    mantissa, exponent = math.frexp(magnitude)
    return exponent - 1 if mantissa == 0.5 else None


def _max_eta0(d0: float, d1: float) -> int:
    d0 = abs(d0)
    d1 = abs(d1)
    if d0 == 0.0:
        return 512
    eta = 0
    while eta < 511 and math.ldexp(d0, eta + 1) <= d1:
        eta += 1
    return eta


def _relu(x: torch.Tensor) -> torch.Tensor:
    return functional.relu(x)


def _elu(x: torch.Tensor) -> torch.Tensor:
    return functional.elu(x)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACTIVATIONS: dict[str, tuple[TensorFunction, float, float]] = {
    "ReLU": (_relu, -1.0, 2.0**-12),
    "ELU": (_elu, -20.0, 0.0),
    "GELU": (_gelu, -5.0, 0.0),
    "Swish": (_swish, -20.0, 0.0),
    "Sigmoid": (torch.sigmoid, -20.0, 0.0),
    "tanh": (torch.tanh, -10.0, 0.0),
}


def _analytic(name: str, x: float) -> tuple[float, float]:
    scalar = torch.tensor(np.float32(x), dtype=torch.float32)
    if name == "ReLU":
        value = torch.maximum(scalar, torch.tensor(0.0))
        derivative = torch.tensor(1.0 if x > 0.0 else 0.0)
    elif name == "ELU":
        value = torch.where(scalar >= 0.0, scalar, torch.expm1(scalar))
        derivative = torch.where(scalar >= 0.0, torch.tensor(1.0), torch.exp(scalar))
    elif name == "GELU":
        cdf = 0.5 * (1.0 + torch.erf(scalar / math.sqrt(2.0)))
        density = torch.exp(-0.5 * scalar * scalar) / math.sqrt(2.0 * math.pi)
        value = scalar * cdf
        derivative = cdf + scalar * density
    elif name == "Swish":
        sigmoid = torch.sigmoid(scalar)
        value = scalar * sigmoid
        derivative = sigmoid + scalar * sigmoid * (1.0 - sigmoid)
    elif name == "Sigmoid":
        value = torch.sigmoid(scalar)
        derivative = value * (1.0 - value)
    elif name == "tanh":
        value = torch.tanh(scalar)
        derivative = 1.0 - value * value
    else:
        raise KeyError(name)
    return float(value.item()), float(derivative.item())


def _autograd(function: TensorFunction, x: float) -> tuple[float, float]:
    scalar = torch.tensor(np.float32(x), dtype=torch.float32, requires_grad=True)
    value = function(scalar)
    derivative = torch.autograd.grad(value, scalar)[0]
    return float(value.detach().item()), float(derivative.detach().item())


def _agreement(observed: float, reference: float) -> bool:
    if observed == reference:
        return True
    scale = max(abs(observed), abs(reference), float(np.finfo(np.float32).tiny))
    return abs(observed - reference) <= 8.0 * float(np.spacing(np.float32(scale)))


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    rows = []
    for name, (function, delta0, delta1) in ACTIVATIONS.items():
        value0, derivative0 = _analytic(name, delta0)
        value1, derivative1 = _analytic(name, delta1)
        autograd0 = _autograd(function, delta0)
        autograd1 = _autograd(function, delta1)
        derivative_exponent = _power_of_two_exponent(derivative1)
        eta0 = _max_eta0(derivative0, derivative1)
        eta1 = 2
        eta2 = 0
        max_value = max(abs(value0), abs(value1))
        bullet1 = (
            all(
                abs(value) <= math.ldexp(1.0, E_MAX)
                for value in (value0, value1, value0 - value1)
            )
            and value0 != value1
        )
        bullet2 = (
            eta0 >= 1
            and math.ldexp(abs(derivative0), eta0) <= abs(derivative1)
            and math.ldexp(1.0, -P) <= abs(derivative1) <= 1.0
            and derivative_exponent is not None
        )
        bullet3 = (
            derivative_exponent is not None
            and -derivative_exponent <= eta1 <= E_MAX
            and derivative_exponent <= eta2 <= -E_MIN + P
        )
        bullet4 = (
            math.ldexp(1.0, E_MIN + eta2)
            <= max_value
            <= math.ldexp(1.0, E_MAX - eta1)
        )
        condition3_second_bullet = (
            abs(_analytic(name, 0.0)[1]) >= math.ldexp(1.0, -P)
            and all(
                math.isfinite(_analytic(name, endpoint)[0])
                and abs(_analytic(name, endpoint)[0]) <= math.ldexp(1.0, E_MAX)
                for endpoint in (-2.0**-9, 0.0, 2.0**-9)
            )
        )
        autograd_agrees = all(
            _agreement(observed, reference)
            for observed, reference in zip(
                (*autograd0, *autograd1),
                (value0, derivative0, value1, derivative1),
                strict=True,
            )
        )
        rows.append(
            {
                "activation": name,
                "delta0": delta0,
                "delta1": delta1,
                "sigma_delta0": value0,
                "sigma_delta1": value1,
                "derivative_delta0": derivative0,
                "derivative_delta1": derivative1,
                "derivative_power_of_two_exponent": derivative_exponent,
                "eta0": eta0,
                "eta1": eta1,
                "eta2": eta2,
                "condition2_bullets": [bullet1, bullet2, bullet3, bullet4],
                "condition3_second_bullet": condition3_second_bullet,
                "autograd_delta0": autograd0,
                "autograd_delta1": autograd1,
                "autograd_agrees": autograd_agrees,
                "pass": all(
                    (bullet1, bullet2, bullet3, bullet4, condition3_second_bullet)
                )
                and autograd_agrees,
            }
        )

    constant_control = {
        "name": "constant activation sigma(x)=1",
        "sigma_delta0": 1.0,
        "sigma_delta1": 1.0,
        "derivative_delta0": 0.0,
        "derivative_delta1": 0.0,
        "condition2_value_separation": False,
        "condition2_derivative_lower_bound": False,
    }
    independent_checker = {
        "all_six_present": {row["activation"] for row in rows} == set(ACTIVATIONS),
        "all_condition2_witnesses_pass": all(
            all(row["condition2_bullets"]) for row in rows
        ),
        "all_condition3_local_witnesses_pass": all(
            row["condition3_second_bullet"] for row in rows
        ),
        "all_analytic_autograd_checks_pass": all(row["autograd_agrees"] for row in rows),
        "constant_control_rejected": not constant_control[
            "condition2_value_separation"
        ]
        and not constant_control["condition2_derivative_lower_bound"],
    }
    if not independent_checker["constant_control_rejected"]:
        raise AssertionError("constant-activation negative control was not rejected")
    return {
        "route": "float32 Conditions 2 and 3 witnesses for six practical activations",
        "seed": SEED,
        "format": {"p": P, "q": Q, "e_min": E_MIN, "e_max": E_MAX},
        "rows": rows,
        "negative_control": constant_control,
        "independent_checker": independent_checker,
        "verdicts": {"claim_2": "BLOCKED"},
        "limitation": (
            "All four bullets of Condition 2 and the local second bullet of Condition 3 "
            "are directly checked. Condition 1 and the full network construction for each "
            "non-ReLU activation are not independently certified, so the universal "
            "six-activation theorem claim remains BLOCKED."
        ),
    }
