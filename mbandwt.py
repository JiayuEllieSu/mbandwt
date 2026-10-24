"""A one-level M-channel cosine block transform."""

import numpy as np


def _basis(bands):
    """Build an orthonormal cosine basis for a block of M samples."""
    if isinstance(bands, (bool, np.bool_)) or not isinstance(bands, (int, np.integer)) or bands < 2:
        raise ValueError("bands must be an integer >= 2")

    k = np.arange(bands)[:, None]
    n = np.arange(bands)[None, :]
    basis = np.sqrt(2.0 / bands) * np.cos(np.pi * k * (n + 0.5) / bands)
    basis[0] /= np.sqrt(2.0)
    return basis


def decompose(signal, bands=3):
    """Split a 1D signal into blocks and return one coefficient row per band."""
    basis = _basis(bands)
    x = np.asarray(signal)
    if x.ndim != 1 or x.size == 0 or x.size % bands:
        raise ValueError("signal must be nonempty, 1D, and divisible by bands")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("signal must contain real numbers")
    x = x.astype(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("signal must contain finite numbers")

    blocks = x.reshape(-1, bands)
    return (blocks @ basis.T).T


def reconstruct(coefficients):
    """Reconstruct the signal from the coefficient rows."""
    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 2 or coeffs.shape[0] < 2 or coeffs.shape[1] == 0:
        raise ValueError("coefficients must have shape (bands, blocks), with bands >= 2")
    if not np.issubdtype(coeffs.dtype, np.number) or np.iscomplexobj(coeffs):
        raise ValueError("coefficients must contain real numbers")
    coeffs = coeffs.astype(float)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must contain finite numbers")

    basis = _basis(coeffs.shape[0])
    return (coeffs.T @ basis).reshape(-1)
