import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def calculate_rank_histogram(ensemble, observations):
    ranks = []
    for ens, obs in zip(ensemble, observations):
        ranks.append(np.sum(ens < obs) + 1)
    rank_histogram, _ = np.histogram(ranks, bins=np.arange(1, ensemble.shape[1] + 2))
    return rank_histogram

def plot_rank_histogram(rank_histogram):
    plt.bar(range(1, len(rank_histogram) + 1), rank_histogram, color='skyblue')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Rank Histogram')
    plt.show()

def calculate_pit_score(ensemble, observations):
    pit_scores = []
    for ens, obs in zip(ensemble, observations):
        sorted_ens = np.sort(ens)
        cdf = np.searchsorted(sorted_ens, obs, side='right') / len(sorted_ens)
        pit_scores.append(cdf)
    return np.array(pit_scores)

def plot_pit_histogram(pit_scores):
    plt.hist(pit_scores, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
    plt.xlabel('PIT Score')
    plt.ylabel('Frequency')
    plt.title('PIT Histogram')
    plt.show()

def calculate_band_depth_rank_histogram(ensemble, observations):
    ranks = []
    for ens, obs in zip(ensemble, observations):
        depth = np.sum((ens <= obs) & (obs <= np.max(ens)))
        ranks.append(depth)
    band_depth_histogram, _ = np.histogram(ranks, bins=np.arange(0, ensemble.shape[1] + 1))
    return band_depth_histogram

def plot_band_depth_histogram(band_depth_histogram):
    plt.bar(range(len(band_depth_histogram)), band_depth_histogram, color='skyblue')
    plt.xlabel('Band Depth Rank')
    plt.ylabel('Frequency')
    plt.title('Band Depth Rank Histogram')
    plt.show()

if __name__ == "__main__":
    # Example data (replace with actual ensemble and observation data)
    np.random.seed(0)
    ensemble = np.random.rand(100, 10)  # 100 samples, 10 ensemble members
    observations = np.random.rand(100)  # 100 observations

    # Calculate and plot Rank Histogram
    rank_histogram = calculate_rank_histogram(ensemble, observations)
    plot_rank_histogram(rank_histogram)

    # Calculate and plot PIT Histogram
    pit_scores = calculate_pit_score(ensemble, observations)
    plot_pit_histogram(pit_scores)

    # Calculate and plot Band Depth Rank Histogram
    band_depth_histogram = calculate_band_depth_rank_histogram(ensemble, observations)
    plot_band_depth_histogram(band_depth_histogram)