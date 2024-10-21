import keras
from keras import ops


def sequential_method(H_tilde, Z):
    _, n, _ = H_tilde.shape
    h = ops.zeros_like(H_tilde[:, 0])
    H = []
    for i in range(n):
        h = h + Z[:, i, :] * (H_tilde[:, i, :] - h)
        H.append(h)
    return ops.stack(H, axis=1)


def Blelloch_operator(prev, curr):
    prev_keep, prev_hidden = prev
    curr_keep, curr_hidden = curr
    keep = prev_keep * curr_keep
    hidden = prev_hidden * curr_keep + curr_hidden
    return keep, hidden


def Blellochs_method(H_tilde, Z, axis=-2):
    _, H = ops.associative_scan(Blelloch_operator, ((1 - Z), Z * H_tilde), axis=axis)
    return H


class MinGRU(keras.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gate = keras.layers.Dense(units, activation="sigmoid")
        self.candidate = keras.layers.Dense(units)

    def build(self, input_shape):
        super().build(input_shape)
        self.gate.build(input_shape)
        self.candidate.build(input_shape)

    def call(self, X, method="blellochs"):
        Z = self.gate(X)
        H_tilde = self.candidate(X)
        if method == "blellochs":
            H = Blellochs_method(H_tilde, Z)
        elif method == "sequential":
            H = sequential_method(H_tilde, Z)
        else:
            raise NotImplementedError(f"Method {method} is not implemented!")

        return H

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
