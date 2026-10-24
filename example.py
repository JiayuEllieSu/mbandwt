import numpy as np
from mbandwt import decompose, reconstruct

signal = np.arange(12, dtype=float)
coefficients = decompose(signal, bands=3)
recovered = reconstruct(coefficients)

print("Coefficient shape:", coefficients.shape)
print("Maximum reconstruction error:", np.max(np.abs(signal - recovered)))
