"""Explicit float32 interpreter for the paper's affine and AD order.

The paper defines matrix products as left-to-right rounded sums and reverse-mode
AD as left-associative products. Framework BLAS kernels need not expose that
reduction order. This module evaluates the pinned authors' weights with the
paper's operations directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.nn as nn

from src.models import CancellationLinear, ExactIndicator, GradIndicator, ZeroGradIndicator


Array = np.ndarray


def f32(value: Any) -> np.float32:
    return np.float32(value)


def left_sum(values: Array) -> np.float32:
    total = f32(0.0)
    for value in np.asarray(values, dtype=np.float32).reshape(-1):
        total = f32(total + f32(value))
    return total


def affine_forward(layer: nn.Linear, inputs: Array) -> Array:
    x = np.asarray(inputs, dtype=np.float32).reshape(-1)
    weights = layer.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    bias = (
        layer.bias.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if layer.bias is not None
        else None
    )
    outputs = np.empty(weights.shape[0], dtype=np.float32)
    for row in range(weights.shape[0]):
        products = np.asarray([f32(weights[row, col] * x[col]) for col in range(weights.shape[1])])
        value = left_sum(products)
        if bias is not None:
            value = f32(value + bias[row])
        outputs[row] = value
    return outputs


def affine_backward(layer: nn.Linear, upstream: Array) -> Array:
    gradient = np.asarray(upstream, dtype=np.float32).reshape(-1)
    weights = layer.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    outputs = np.empty(weights.shape[1], dtype=np.float32)
    for col in range(weights.shape[1]):
        products = np.asarray(
            [f32(gradient[row] * weights[row, col]) for row in range(weights.shape[0])]
        )
        outputs[col] = left_sum(products)
    return outputs


def relu_forward(inputs: Array) -> tuple[Array, Array]:
    x = np.asarray(inputs, dtype=np.float32).reshape(-1)
    return np.maximum(x, f32(0.0)).astype(np.float32), x


def relu_backward(preactivation: Array, upstream: Array) -> Array:
    mask = (np.asarray(preactivation, dtype=np.float32) > f32(0.0)).astype(np.float32)
    return np.asarray(
        [f32(g * m) for g, m in zip(np.asarray(upstream).reshape(-1), mask.reshape(-1))],
        dtype=np.float32,
    )


@dataclass
class BasicIndicatorCache:
    pre1: Array
    pre2: Array
    pre3: Array


def basic_indicator_forward(module: nn.Module, x: np.float32) -> tuple[np.float32, BasicIndicatorCache]:
    first, pre1 = relu_forward(affine_forward(module.lin1, np.asarray([x], dtype=np.float32)))
    second, pre2 = relu_forward(affine_forward(module.lin2, first))
    third, pre3 = relu_forward(affine_forward(module.lin3, second))
    return f32(third[0]), BasicIndicatorCache(pre1=pre1, pre2=pre2, pre3=pre3)


def basic_indicator_backward(
    module: nn.Module, cache: BasicIndicatorCache, upstream: np.float32
) -> np.float32:
    grad = relu_backward(cache.pre3, np.asarray([upstream], dtype=np.float32))
    grad = affine_backward(module.lin3, grad)
    grad = relu_backward(cache.pre2, grad)
    grad = affine_backward(module.lin2, grad)
    grad = relu_backward(cache.pre1, grad)
    grad = affine_backward(module.lin1, grad)
    return f32(grad[0])


@dataclass
class ExactIndicatorCache:
    left: BasicIndicatorCache
    right: BasicIndicatorCache
    pre_out: Array


def exact_indicator_forward(
    module: ExactIndicator, x: np.float32
) -> tuple[np.float32, ExactIndicatorCache]:
    left_module = module.left.model
    right_module = module.right.model
    left, left_cache = basic_indicator_forward(left_module, x)
    right, right_cache = basic_indicator_forward(right_module, x)
    output, pre_out = relu_forward(
        affine_forward(module.lin_out, np.asarray([left, right], dtype=np.float32))
    )
    return f32(output[0]), ExactIndicatorCache(left=left_cache, right=right_cache, pre_out=pre_out)


def exact_indicator_backward(
    module: ExactIndicator, cache: ExactIndicatorCache, upstream: np.float32
) -> np.float32:
    grad = relu_backward(cache.pre_out, np.asarray([upstream], dtype=np.float32))
    branch_gradients = affine_backward(module.lin_out, grad)
    left = basic_indicator_backward(module.left.model, cache.left, f32(branch_gradients[0]))
    right = basic_indicator_backward(module.right.model, cache.right, f32(branch_gradients[1]))
    return f32(left + right)


@dataclass
class CancellationCache:
    pre1: Array
    pre2: Array


def cancellation_input(x: np.float32, p: int, e_max: int) -> Array:
    values = [f32(x)]
    for exponent in range(p, e_max + 1):
        power = f32(np.ldexp(f32(1.0), exponent))
        successor = np.nextafter(power, f32(np.inf), dtype=np.float32)
        values.extend([successor, f32(-successor)])
    return np.asarray(values, dtype=np.float32)


def cancellation_forward(
    module: CancellationLinear, x: np.float32
) -> tuple[np.float32, CancellationCache]:
    vector = cancellation_input(x, module.p, module.e_max)
    first, pre1 = relu_forward(affine_forward(module.linear, vector))
    second, pre2 = relu_forward(affine_forward(module.linear2, first))
    return f32(second[0]), CancellationCache(pre1=pre1, pre2=pre2)


def cancellation_backward(
    module: CancellationLinear, cache: CancellationCache, upstream: np.float32
) -> np.float32:
    grad = relu_backward(cache.pre2, np.asarray([upstream], dtype=np.float32))
    grad = affine_backward(module.linear2, grad)
    grad = relu_backward(cache.pre1, grad)
    grad = affine_backward(module.linear, grad)
    return f32(grad[0])


@dataclass
class TripleCache:
    caches: tuple[CancellationCache, CancellationCache, CancellationCache]


def triple_forward(module: nn.Module, x: np.float32) -> tuple[np.float32, TripleCache]:
    y1, c1 = cancellation_forward(module.c1, x)
    y2, c2 = cancellation_forward(module.c2, y1)
    y3, c3 = cancellation_forward(module.c3, y2)
    return y3, TripleCache((c1, c2, c3))


def triple_backward(module: nn.Module, cache: TripleCache, upstream: np.float32) -> np.float32:
    grad = cancellation_backward(module.c3, cache.caches[2], upstream)
    grad = cancellation_backward(module.c2, cache.caches[1], grad)
    grad = cancellation_backward(module.c1, cache.caches[0], grad)
    return grad


@dataclass
class ZeroGradCache:
    indicator: ExactIndicatorCache
    triple: TripleCache


def zero_grad_forward(
    module: ZeroGradIndicator, x: np.float32
) -> tuple[np.float32, ZeroGradCache]:
    indicated, indicator_cache = exact_indicator_forward(module.indicator, x)
    cancelled, triple_cache = triple_forward(module.cancel, indicated)
    output = affine_forward(module.linear, np.asarray([cancelled], dtype=np.float32))
    return f32(output[0]), ZeroGradCache(indicator=indicator_cache, triple=triple_cache)


def zero_grad_backward(
    module: ZeroGradIndicator, cache: ZeroGradCache, upstream: np.float32
) -> np.float32:
    grad = affine_backward(module.linear, np.asarray([upstream], dtype=np.float32))
    grad_scalar = triple_backward(module.cancel, cache.triple, f32(grad[0]))
    return exact_indicator_backward(module.indicator, cache.indicator, grad_scalar)


@dataclass
class GradIndicatorCache:
    indicator: ExactIndicatorCache
    pre1: Array
    pre2: Array
    zero_grad: ZeroGradCache


def grad_indicator_forward(
    module: GradIndicator, x: np.float32
) -> tuple[np.float32, GradIndicatorCache]:
    indicated, indicator_cache = exact_indicator_forward(module.indicator, x)
    first, pre1 = relu_forward(
        affine_forward(module.linear, np.asarray([indicated], dtype=np.float32))
    )
    second, pre2 = relu_forward(affine_forward(module.linear2, first))
    zero, zero_cache = zero_grad_forward(module.zerograd_indicator, x)
    output = affine_forward(module.linear3, np.asarray([second[0], zero], dtype=np.float32))
    return f32(output[0]), GradIndicatorCache(
        indicator=indicator_cache, pre1=pre1, pre2=pre2, zero_grad=zero_cache
    )


def grad_indicator_backward(
    module: GradIndicator, cache: GradIndicatorCache, upstream: np.float32
) -> np.float32:
    branches = affine_backward(module.linear3, np.asarray([upstream], dtype=np.float32))
    grad_main = relu_backward(cache.pre2, np.asarray([branches[0]], dtype=np.float32))
    grad_main = affine_backward(module.linear2, grad_main)
    grad_main = relu_backward(cache.pre1, grad_main)
    grad_main = affine_backward(module.linear, grad_main)
    grad_main_scalar = exact_indicator_backward(module.indicator, cache.indicator, f32(grad_main[0]))
    grad_zero = zero_grad_backward(module.zerograd_indicator, cache.zero_grad, f32(branches[1]))
    return f32(grad_main_scalar + grad_zero)


def evaluate_zero_grad(module: ZeroGradIndicator, x: float) -> tuple[float, float]:
    output, cache = zero_grad_forward(module, f32(x))
    gradient = zero_grad_backward(module, cache, f32(1.0))
    return float(output), float(gradient)


def evaluate_grad(module: GradIndicator, x: float) -> tuple[float, float]:
    output, cache = grad_indicator_forward(module, f32(x))
    gradient = grad_indicator_backward(module, cache, f32(1.0))
    return float(output), float(gradient)
