# Calculate CRPS
import numpy as np
import properscoring as ps

def calculate_crps(actuals, corrected_ensembles):
    try:
        crps = ps.crps_ensemble(actuals, corrected_ensembles)
        return np.mean(crps)
    except:
        print(f"CRPS failed for {actuals.name}, transposing and trying again...")
        crps = np.mean(ps.crps_ensemble(actuals, corrected_ensembles.T))
        return crps

# Calculate Variogram score
def variogram_score(x, y, p=0.5, t1=12, t2=36):
    """
    Calculate the Variogram score for all observations for the time horizon t1 to t2.
    From the paper in Energy and AI, >> An introduction to multivariate probabilistic forecast evaluation <<.
    Assumes that x and y starts from day 0, 00:00.
    
    Parameters:
    x : array
        Ensemble forecast (m x k), where m is the size of the ensemble, and k is the maximal forecast horizon.
    y : array
        Actual observations (k,)
    p : float
        Power parameter for the variogram score.
    t1 : int
        Start of the hour range for comparison (inclusive).
    t2 : int
        End of the hour range for comparison (exclusive).
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0
    if m > k:
        x = x.T
        m,k = k,m
    else:
        print("m,k: ", m, k)
    

    # Iterate through every 24-hour block
    for start in range(0, k, 24):
        # Ensure we don't exceed the forecast horizon
        if start + t2 <= k:
            for i in range(start + t1, start + t2 - 1):
                for j in range(i + 1, start + t2):
                    Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j])**p)
                    score += (1 / np.abs(i - j)) * (np.abs(y[i] - y[j])**p - Ediff)**2

    # Variogram score
    return score/(100_000)