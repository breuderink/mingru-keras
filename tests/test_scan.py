# %%
from keras import ops
import numpy as np
import pytest


def Blelloch_operator(prev, curr):
    prev_keep, prev_hidden = prev
    curr_keep, curr_hidden = curr
    keep = prev_keep * curr_keep
    hidden = prev_hidden * curr_keep + curr_hidden
    return keep, hidden


def Blellochs_method(X, Z, axis=-2):
    _, H = ops.associative_scan(Blelloch_operator, ((1 - Z), Z * X), axis=axis)
    return H


def test_associativity():
    # f(a, f(b, c)) == f(f(a, b), c).

    X = np.random.randn(3)
    G = np.random.rand(3)
    a, b, c = zip(X, G)

    fn = Blelloch_operator
    foldl = fn(fn(a, b), c)
    foldr = fn(a, fn(b, c))

    assert foldl == pytest.approx(foldr)


@pytest.mark.parametrize("b", [1, 32])
@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("d", [1, 16])
def test_unroll(b, n, d):
    X = np.random.randn(b, n, d)
    Z = np.random.rand(b, n, d)

    H_desired = np.zeros_like(X)
    h = np.zeros(d, dtype=np.float32)
    for i in range(n):
        h = h + Z[:, i, :] * (X[:, i, :] - h)
        H_desired[:, i, :] = h

    H_actual = Blellochs_method(X, Z)
    np.testing.assert_allclose(
        H_actual.squeeze(), H_desired.squeeze(), rtol=1e-4, atol=1e-4
    )
