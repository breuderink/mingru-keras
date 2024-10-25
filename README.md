# MinGRU Implementation in Keras

This repository contains a Keras implementation of the minGRU model, a minimal
and parallelizable version of the traditional Gated Recurrent Unit (GRU)
architecture. The minGRU model is based on the research paper ["Were RNNs All We
Needed?"](https://arxiv.org/abs/2410.01201) that revisits traditional recurrent
neural networks and modifies them to be efficiently trained in parallel.

## Features

* Minimal GRU architecture with significantly fewer parameters than traditional GRUs.
* Fully parallelizable during training, achieving faster training times.
* Compatible with Keras 3.

## Installation

This project uses uv to manage dependencies. To install the required
dependencies, run:

```bash
pip install --upgrade mingru-keras
```

## Usage

To use the MinGRU model in your own project, simply import the `MinGRU` class
and use it as you would any other Keras layer.

## Example

```python
>>> import keras
>>> from mingru_keras import MinGRU
>>> layer = MinGRU(units=64)
>>> X = keras.random.normal((32, 1000, 8))
>>> layer(X).shape
(32, 1000, 64)

```

## Contributing

Contributions are welcome! If you'd like to report a bug or suggest a feature,
please open an issue or submit a pull request.