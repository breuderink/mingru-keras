from tempfile import NamedTemporaryFile

import keras
import pytest
from keras import ops

from mingru_keras import MinGRU
from mingru_keras.core import Blellochs_method, sequential_method


@pytest.mark.parametrize("b,n,d", [(32, 10, 8), (1, 1000, 1)])
def test_Blellochs_method(b, n, d):
    X = keras.random.normal((b, n, d))
    Z = keras.random.uniform((b, n, d))
    H_desired = sequential_method(X, Z)
    H_actual = Blellochs_method(X, Z)

    assert ops.max(ops.abs(H_actual - H_desired)) < 1e-6


@pytest.mark.parametrize("b,n,i,d", [(16, 10, 8, 16), (32, 100, 2, 64)])
def test_MinGRU(b, n, i, d):
    layer = MinGRU(d)
    X = keras.random.normal((b, n, i))
    Y = layer(X)
    assert Y.shape == (b, n, d)

    # Count parameters.
    assert layer.gate.count_params() == i * d + d
    assert layer.candidate.count_params() == i * d + d


@pytest.mark.parametrize("b,n,i,d", [(32, 100, 8, 16)])
def test_saving(b, n, i, d):
    model = keras.Sequential([MinGRU(d)])
    model.build((None, None, i))
    model.summary()

    with NamedTemporaryFile("wb", suffix=".keras") as f:
        keras.saving.save_model(model, f.name)
        model2 = keras.saving.load_model(f.name)
        pass

    X = keras.random.normal((b, n, i))
    Y1 = model(X)
    Y2 = model2(X)

    assert ops.max(ops.abs(Y1 - Y2)) < 1e-4
