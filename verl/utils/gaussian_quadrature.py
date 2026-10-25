"""Gaussian quadrature helpers.

The nodes and weights are generated at runtime instead of being limited to a
small, hard-coded lookup table.  Results are cached per process because a
training run reuses the same quadrature rule for every batch.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def compute_gauss_legendre_quadrature(
    num_points: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a ``num_points`` Gauss-Legendre rule on the interval ``[0, 1]``.

    ``numpy.polynomial.legendre.leggauss`` returns a rule on ``[-1, 1]``;
    both nodes and weights are transformed here so callers cannot accidentally
    omit the interval scaling.  The immutable result is cached, making this a
    one-time CPU calculation for each quadrature size in a worker process.
    """
    if isinstance(num_points, bool) or not isinstance(num_points, int):
        raise TypeError(f"num_points must be an integer, got {type(num_points).__name__}")
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return tuple(float(x) for x in nodes), tuple(float(w) for w in weights)
