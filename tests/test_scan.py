import keras
from keras import ops
import pytest
from mingru.core import sequential_method, Blelloch_operator, Blellochs_method


def test_associativity():
    # f(a, f(b, c)) == f(f(a, b), c).

    X = keras.random.normal(3)
    G = keras.random.uniform(3)
    a, b, c = zip(X, G)

    fn = Blelloch_operator
    foldl = fn(fn(a, b), c)
    foldr = fn(a, fn(b, c))

    assert foldl == pytest.approx(foldr)


@pytest.mark.parametrize("b,n,d", [(32, 10, 8), (1, 1000, 1)])
def test_Blellochs_method(b, n, d):
    X = keras.random.normal((b, n, d))
    Z = keras.random.uniform((b, n, d))
    H_desired = sequential_method(X, Z)
    H_actual = Blellochs_method(X, Z)

    assert ops.max(ops.abs(H_actual - H_desired)) < 1e-6
