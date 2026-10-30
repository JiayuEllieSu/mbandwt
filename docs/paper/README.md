# Wavelet exposition

**From Fourier Analysis to M-Band Wavelets: Multiresolution and Perfect-Reconstruction Filter Banks**  
Jiayu Ellie Su, October 2026

- [PDF](mband_wavelets.pdf)
- [LaTeX source](mband_wavelets.tex)
- [Figure generation and numerical checks](generate_figures.py)

The paper covers discrete wavelet transforms, M-band filter banks, vanishing moments, polyphase matrices, and reconstruction. `generate_figures.py` builds the plots and computes numerical checks.

To build locally (Python with NumPy, SciPy, Matplotlib, and a LaTeX installation):

```bash
python generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error mband_wavelets.tex
pdflatex -interaction=nonstopmode -halt-on-error mband_wavelets.tex
```

The GitHub Actions workflow builds the PDF when the LaTeX source changes.