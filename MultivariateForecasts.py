import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf


# Define time horizon and quantiles
horizon = 48  # Forecast horizon of 48 hours
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # Quantiles we're interested in
matplotlib.style.use("dark_background")

# Generate synthetic probabilistic forecasts for the time horizon
np.random.seed(42)  # For reproducibility
true_values = np.sin(np.linspace(0, 4 * np.pi, horizon)) + np.random.normal(0, 0.1, horizon)  # True series
forecasts = {
    q: true_values + np.random.normal(q-0.5, 0.05, horizon) for q in quantiles  # Simulate density forecasts per quantile
}

# Plot
plt.figure(figsize=(10, 4))
plt.plot(range(horizon), true_values, label='True Values', color='blue', linewidth=2)
for q in quantiles:
    if q == 0.5:
        color = "red"
    else:
        color = "white"
    plt.plot(range(horizon), forecasts[q], linestyle='--', label=f'Quantile {q}', alpha=0.7, color = color)

plt.fill_between(range(horizon), forecasts[0.1], forecasts[0.9], color='white', alpha=0.2, label='Quantile range (0.1 - 0.9)')
plt.title('Multivariate Probabilistic Forecast (Quantiles)')
plt.xlabel('Time Horizon (Hours)')
plt.ylabel('Forecast Values')
plt.legend()
plt.show()

# Structure of predictions
predictions = np.array([forecasts[q] for q in quantiles]).T  
print(f"Shape of predictions (R^{{48x5}}): {predictions.shape}") # Shape (48, 5), as expected.

df_forecasts = pd.DataFrame(forecasts)
df_forecasts['True Values'] = true_values

# ACF plot for each quantile
plt.figure(figsize=(10, 8))
for i, q in enumerate(quantiles):
    plot_acf(df_forecasts[q], ax=plt.subplot(len(quantiles), 1, i+1), title=f'ACF for Quantile {q}', lags=20)
plt.tight_layout()
plt.show()

# Correlation matrix for the forecasts
correlation_matrix = df_forecasts[quantiles].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(6, 5))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
plt.colorbar()
plt.xticks(range(len(quantiles)), [f'Q{q}' for q in quantiles], rotation=90)
plt.yticks(range(len(quantiles)), [f'Q{q}' for q in quantiles])
plt.title('Correlation Matrix of Quantiles', pad=20)
plt.show()

