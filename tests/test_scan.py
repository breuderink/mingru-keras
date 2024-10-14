from keras import ops
import pytest

def merge(prev, curr):
    H_prev, Z_prev = prev
    H_curr, Z_curr = curr
    H = H_prev + Z_curr * (H_curr - H_prev)
    Z = ops.maximum(Z_prev, Z_curr)
    # print(f"{H_prev=}\n{H_curr=}\n{Z_curr=}\n=>\n{H=}\n{Z=},\n")
    return H, Z


def unroll(X, Z):
    H, _ = ops.associative_scan(merge, (X, Z))
    return H

@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("position", range(10))
def test_sparse_update(n, position):
    X = ops.arange(n)
    Z = ops.where(X == position, 1, 0)
    Y_actual = unroll(X, Z)
    Y_desired = ops.where(X < position, 0, position)
    assert (Y_actual == Y_desired).all()

