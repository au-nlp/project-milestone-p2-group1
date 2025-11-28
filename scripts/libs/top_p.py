import numpy as np
from scipy.special import softmax

# ============================================
# 1. Top-p selection function
# ============================================
def get_top_p_indices(similarities, top_p):
    """
    Select words until cumulative probability reaches top_p threshold.
    
    Args:
        similarities: numpy array of similarity scores
        top_p: probability threshold (e.g., 0.01, 0.05, 0.95)
    
    Returns:
        numpy array of indices that meet the top_p threshold
    """
    sorted_indices = similarities.argsort()[::-1]
    sorted_sims = similarities[sorted_indices]
    
    # Convert to probabilities using softmax
    exp_sims = np.exp(sorted_sims - np.max(sorted_sims))
    probs = exp_sims / np.sum(exp_sims)
    
    # Find cumulative probability cutoff
    cumsum = np.cumsum(probs)
    cutoff_idx = np.where(cumsum >= top_p)[0][0]
    
    return sorted_indices[:cutoff_idx + 1]


def get_top_p_indices2(scores, top_p):
    """
    Nucleus (top-p) selection.
    Takes a 1D array of scores (e.g., similarities),
    converts to probabilities via softmax,
    and returns the smallest set of indices whose
    cumulative probability >= top_p.
    """
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]

    probs = softmax(sorted_scores)
    cumsum = np.cumsum(probs)
    cutoff_idx = np.searchsorted(cumsum, top_p)

    print("First 20 probabilities:\n", probs[:20])
    print("Min and max probability:\n", probs.min(), probs.max())

    return sorted_indices[:cutoff_idx + 1]
