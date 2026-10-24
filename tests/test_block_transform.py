import numpy as np
import pytest
from mbandwt import decompose, reconstruct


@pytest.mark.parametrize("bands", [2, 3, 4, 5, 8])
def test_block_round_trip(bands):
    signal = np.random.default_rng(bands).normal(size=15 * bands)
    coefficients = decompose(signal, bands=bands)
    assert coefficients.shape == (bands, 15)
    np.testing.assert_allclose(reconstruct(coefficients), signal, atol=1e-12)


def test_constant_signal_has_zero_detail():
    np.testing.assert_allclose(decompose(np.ones(15), bands=3)[1:], 0, atol=1e-12)


def test_energy():
    x = np.random.default_rng(42).normal(size=24)
    np.testing.assert_allclose(np.sum(decompose(x, 4) ** 2), np.sum(x ** 2))


@pytest.mark.parametrize("bands", [0, 1, -2, 2.5, True])
def test_invalid_bands(bands):
    with pytest.raises(ValueError):
        decompose(np.ones(12), bands)


@pytest.mark.parametrize("x", [[], [[1, 2], [3, 4]], [1, 2, 3, 4], [1, 2, np.inf], [1+2j, 3, 4]])
def test_invalid_signal(x):
    with pytest.raises(ValueError):
        decompose(x, bands=3)


def test_invalid_coefficients():
    with pytest.raises(ValueError):
        reconstruct(np.ones((1, 2)))
