# mbandwt

Python code for **M-band signal transforms**, FIR filter banks, and reconstruction, with an arbitrary number of channels `M >= 2`.

The package includes a cosine block transform, overlapping FIR filter banks, and multilevel decomposition.

## Install

Requires Python 3.9+, NumPy, and SciPy.

```bash
python -m pip install -e .
```

## Block transform

This example applies an orthonormal cosine transform to consecutive blocks of a one-dimensional signal. The blocks do not overlap.

```python
import numpy as np
from mbandwt import decompose, reconstruct

signal = np.arange(12.0)
coeffs = decompose(signal, bands=3)
recovered = reconstruct(coeffs)
print(coeffs.shape)                       # (3, 4)
print(np.allclose(signal, recovered))     # True
```

Run `python example.py` to see the same example.

## FIR filter banks

An FIR filter bank uses one analysis and one synthesis filter for each channel. The analysis stage filters and downsamples the signal. The inverse stage upsamples and filters the coefficients, then combines the results.

```python
import numpy as np
from mbandwt import dwt, idwt
from mbandwt.families import mlt

bank = mlt(3)                             # 3-channel lapped filter bank
signal = np.random.default_rng(0).normal(size=90)
coeffs = dwt(signal, bank, mode="periodic")
recovered = idwt(coeffs)
print(bank.validate().perfect_reconstruction)
print(np.allclose(signal, recovered))
```

`FilterBank(analysis, synthesis)` also accepts custom FIR filters. Arbitrary custom coefficients do **not** automatically give perfect reconstruction. Check a bank with `bank.validate()` before using it for an inverse.

## Multilevel decomposition

At each level, `wavedec` splits the approximation (channel 0) into another `M` channels. `waverec` reverses the levels using stored lengths and boundary information.

```python
import numpy as np
from mbandwt import wavedec, waverec
from mbandwt.families import mlt

bank = mlt(3)
signal = np.random.default_rng(4).normal(size=486)
coeffs = wavedec(signal, bank, level=4, mode="periodic")
recovered = waverec(coeffs)
print("levels:", coeffs.level)
print("error:", np.max(np.abs(signal - recovered)))
```

See [`examples/multilevel_analysis.py`](examples/multilevel_analysis.py) for subband energies and [`examples/custom_bank_validation.py`](examples/custom_bank_validation.py) for validating custom filters.

## Additional features

- `FilterBank`: analysis/synthesis FIR filter coefficients and diagnostic methods
- `polyphase.py`: polyphase matrices and numerical perfect-reconstruction checks
- `families/`: Haar and Daubechies (two channels), M-channel block cosine, lapped cosine, lazy, and paraunitary constructions
- `frequency_response()`: sample each filter's frequency response, with optional plotting
- Boundary modes: `periodic`, `zero`, `symmetric`, `reflect`, `constant`
- Complex-valued and batched signals with a selectable transform axis

Finite-extension boundary modes can produce more coefficients than the input length, and some lengths require padding. Multilevel decomposition splits only the approximation channel. Haar and Daubechies filters are implemented for `M = 2`; lapped and paraunitary constructions support larger `M`.

For the filtering phase, gain/delay compensation, and polyphase conventions, see [`docs/conventions.md`](docs/conventions.md).

## Mathematical exposition

The paper *From Fourier Analysis to M-Band Wavelets: Multiresolution and Perfect-Reconstruction Filter Banks* covers the mathematics behind the implementation: multiresolution analysis, vanishing moments, polyphase matrices, filter design, and perfect reconstruction.

- [Read the exposition (PDF)](docs/paper/mband_wavelets.pdf)
- [LaTeX source](docs/paper/mband_wavelets.tex)
- [Figure and numerical experiment code](docs/paper/generate_figures.py)
- [Project page](https://jiayuelliesu.github.io/projects/mbandwt/)

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Optional cross-checks use PyWavelets and externally generated LTFAT reference data; absent optional data may cause a skip.

Passing the tests does not establish perfect reconstruction for an arbitrary user-defined bank. Use `bank.validate()` to check custom coefficients.

MIT License. Jiayu Ellie Su.
