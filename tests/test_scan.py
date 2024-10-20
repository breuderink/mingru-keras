# %%
from keras import ops
import numpy as np
import pytest
from mingru.mingru import sequential_method, Blelloch_operator, Blellochs_method


def test_associativity():
    # f(a, f(b, c)) == f(f(a, b), c).

    X = np.random.randn(3)
    G = np.random.rand(3)
    a, b, c = zip(X, G)

    fn = Blelloch_operator
    foldl = fn(fn(a, b), c)
    foldr = fn(a, fn(b, c))

    assert foldl == pytest.approx(foldr)


@pytest.mark.parametrize("b,n,d", [(32, 10, 8), (1, 10000, 1)])
def test_Blellochs_method(b, n, d):
    X = np.random.randn(b, n, d)
    Z = np.random.rand(b, n, d)
    H_desired = sequential_method(X, Z)
    H_actual = Blellochs_method(ops.convert_to_tensor(X), ops.convert_to_tensor(Z))
    np.testing.assert_allclose(
        H_actual.squeeze(), H_desired.squeeze(), rtol=1e-4, atol=1e-4
    )
