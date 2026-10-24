import numpy as np
import pytest
from mbandwt import decompose, reconstruct


@pytest.mark.parametrize("bands", [2, 3, 4, 5, 8])
def test_round_trip(bands):
    signal = np.random.default_rng(bands).normal(size=bands * 15)
    coefficients = decompose(signal, bands)
    assert coefficients.shape == (bands, 15)
    np.testing.assert_allclose(reconstruct(coefficients), signal, atol=1e-12)


def test_constant_signal_has_no_detail_coefficients():
    coefficients = decompose(np.ones(15), bands=3)
    np.testing.assert_allclose(coefficients[1:], 0, atol=1e-12)


def test_energy_is_preserved():
    signal = np.random.default_rng(12).normal(size=24)
    coefficients = decompose(signal, bands=4)
    np.testing.assert_allclose(np.sum(coefficients**2), np.sum(signal**2))


@pytest.mark.parametrize("bands", [0, 1, -3, 2.5, True])
def test_invalid_band_count(bands):
    with pytest.raises(ValueError):
        decompose(np.ones(12), bands)


@pytest.mark.parametrize("signal", [[], [[1, 2], [3, 4]], [1, 2, 3, 4]])
def test_invalid_signal_shape_or_length(signal):
    with pytest.raises(ValueError):
        decompose(signal, bands=3)


@pytest.mark.parametrize("signal", [[1, 2, float("nan")], [1, float("inf"), 3], [1 + 1j, 2, 3]])
def test_invalid_signal_values(signal):
    with pytest.raises(ValueError):
        decompose(signal, bands=3)


@pytest.mark.parametrize("coefficients", [
    np.ones((1, 4)),
    np.ones((3, 0)),
    np.ones(6),
    [[1, 2], [3, float("nan")]],
    [[1 + 1j, 2], [3, 4]],
])
def test_invalid_coefficients(coefficients):
    with pytest.raises(ValueError):
        reconstruct(coefficients)
