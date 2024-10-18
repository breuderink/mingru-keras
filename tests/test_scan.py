# %%
import keras
from operator import add, mul
from keras import ops
import numpy as np
import pytest


def recurrence_op(i, j):
    c_ia, c_ib = i
    c_ja, c_jb = j

    star = mul
    companion = mul
    plus = add
    return (companion(c_ia, c_ja), plus(star(c_ib, c_ja), c_jb))


merge = recurrence_op


def test_associative():
    # f(a, f(b, c)) == f( f(a, b), c).

    X = np.random.randn(3)
    G = np.random.rand(3)
    a, b, c = zip(X, G)

    fn = merge
    foldl = fn(fn(a, b), c)
    foldr = fn(a, fn(b, c))

    assert foldl == pytest.approx(foldr)


def unroll(X, Z, axis=-2):
    A, B = (1 - Z), Z * X
    _, H = ops.associative_scan(merge, (A, B), axis=axis)
    return H


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("d", [1])
def test_unroll(b, n, d):
    X = np.arange(n, dtype=float)[None, :, None]
    Z = np.full_like(X, fill_value=0.1)

    print(X.squeeze())
    print(Z.squeeze())

    H_desired = np.zeros_like(X)
    h = np.zeros(d, dtype=np.float32)
    for i in range(n):
        h = h + Z[:, i, :] * (X[:, i, :] - h)
        H_desired[:, i, :] = h

    H_actual = unroll(X, Z, axis=1)
    np.testing.assert_allclose(H_actual.squeeze(), H_desired.squeeze(), rtol=1e-4)
