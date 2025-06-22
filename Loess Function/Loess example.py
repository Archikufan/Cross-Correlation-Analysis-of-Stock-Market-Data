import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate example data
np.random.seed(0)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1 * np.random.normal(size=len(x))

# Apply Lowess smoothing to detrend the data
lowess = sm.nonparametric.lowess(y, x, frac=0.1)

# Plot original and detrended data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')
plt.plot(lowess[:, 0], lowess[:, 1], color='red', label='Lowess Smoothing')
plt.legend()
plt.title('Detrended Data using Lowess')
plt.show()
