import numpy as np
from scipy.stats import norm, wasserstein_distance

def normalize_scores(scores, min_val, max_val):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores)) * (max_val - min_val) + min_val
