import keras
from keras import ops
import pytest
from mingru import MinGRU
from mingru.core import sequential_method, Blellochs_method


@pytest.mark.parametrize("b,n,d", [(32, 10, 8), (1, 1000, 1)])
def test_Blellochs_method(b, n, d):
    X = keras.random.normal((b, n, d))
    Z = keras.random.uniform((b, n, d))
    H_desired = sequential_method(X, Z)
    H_actual = Blellochs_method(X, Z)

    assert ops.max(ops.abs(H_actual - H_desired)) < 1e-6


@pytest.mark.parametrize("b,n,d,d2", [(16, 10, 8, 16), (32, 100, 2, 64)])
def test_MinGRU(b, n, d, d2):

    layer = MinGRU(d2)

    X = keras.random.normal((b, n, d))
    Y = layer(X)
    assert Y.shape == (b, n, d2)

    # Count parameters.
    assert layer.gate.count_params() == d * d2 + d2
    assert layer.candidate.count_params() == d * d2 + d2
