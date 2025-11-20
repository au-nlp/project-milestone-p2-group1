import numpy as np

def get_top_p_indices(similarities, top_p):
    """
    Select words until cumulative probability reaches top_p threshold.
    """
    sorted_indices = similarities.argsort()[::-1]
    sorted_sims = similarities[sorted_indices]
    
    exp_sims = np.exp(sorted_sims - np.max(sorted_sims))
    probs = exp_sims / np.sum(exp_sims)
    
    cumsum = np.cumsum(probs)
    cutoff_idx = np.where(cumsum >= top_p)[0][0]
    
    return sorted_indices[:cutoff_idx + 1]