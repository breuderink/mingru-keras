from keras import ops


def sequential_method(X, Z):
    _, n, _ = X.shape
    h = ops.zeros_like(X[:, 0])
    H = []
    for i in range(n):
        h = h + Z[:, i, :] * (X[:, i, :] - h)
        H.append(h)
    return ops.stack(H, axis=1)


def Blelloch_operator(prev, curr):
    prev_keep, prev_hidden = prev
    curr_keep, curr_hidden = curr
    keep = prev_keep * curr_keep
    hidden = prev_hidden * curr_keep + curr_hidden
    return keep, hidden


def Blellochs_method(X, Z, axis=-2):
    _, H = ops.associative_scan(Blelloch_operator, ((1 - Z), Z * X), axis=axis)
    return H
