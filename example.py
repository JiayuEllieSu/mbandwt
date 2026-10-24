import numpy as np
from mbandwt import decompose, reconstruct

signal = np.arange(12, dtype=float)
coefficients = decompose(signal, bands=3)
recovered = reconstruct(coefficients)

print("Coefficients shape:", coefficients.shape)
print("Reconstruction error:", np.max(np.abs(signal - recovered)))
