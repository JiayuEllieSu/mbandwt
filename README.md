# mbandwt

A small Python experiment with decomposing a signal into several components and putting it back together.

The code splits a one-dimensional signal into blocks of `M` samples, then applies an orthonormal cosine basis to each block. This produces `M` rows of coefficients, from the constant (lowest-frequency) pattern to more rapidly varying patterns. Multiplying by the basis again reconstructs the signal, up to floating-point error.

This is a **one-level cosine block transform**, not a complete M-band wavelet transform. The coefficients describe local cosine patterns rather than an overlapping wavelet filter bank.

## Run the example

Requires Python 3.9+ and NumPy.

```bash
python -m pip install -e .
python example.py
```

## Using it

```python
import numpy as np
from mbandwt import decompose, reconstruct

signal = np.arange(12, dtype=float)
coefficients = decompose(signal, bands=3)
recovered = reconstruct(coefficients)

print(coefficients.shape)  # (3, 4)
print(np.allclose(signal, recovered))  # True
```

The input must be a nonempty, one-dimensional array of finite real numbers. `bands` must be an integer of at least 2, and the signal length must be divisible by `bands`.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

## Scope

This project is meant to demonstrate decomposition and reconstruction with a simple orthogonal basis. It does not include overlapping FIR wavelet filters, boundary extension, multilevel decomposition, or polyphase methods.

[Project page](https://jiayuelliesu.github.io/projects/mbandwt/)

MIT licensed.
