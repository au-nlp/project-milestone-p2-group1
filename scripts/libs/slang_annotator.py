import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer
from scripts.libs.utils import get_top_p_indices
from matplotlib import pyplot as plt

def compute_all_similarities(df, genz_data, model, slang_embeddings):
    """
    Compute similarity scores between each TL;DR and all slang words.
    """
    df = df.copy()
    
    # Prepare lists to collect data
    all_slang_list = []
    all_similarities_list = []
    
    # Compute similarities for each TL;DR
    print("\nComputing similarities for each TL;DR...")
    for completion in tqdm(df['completion'], desc="Computing similarities"):
        # Encode TL;DR
        tldr_embedding = model.encode([completion])[0]
        
        # Compute similarities
        similarities = cosine_similarity([tldr_embedding], slang_embeddings)[0]
        
        # Normalize to [0, 1] range
        # L2 normalization instead of min-max to avoid zero-division
        # L1 normalization could also be used
        similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        
        # Store as semicolon-separated strings
        all_slang_list.append(';'.join(genz_data['Slang'].tolist()))
        all_similarities_list.append(';'.join([f"{sim:.6f}" for sim in similarities]))
    
    # Add columns to dataframe
    df['all_slang'] = all_slang_list 
    df['all_similarities'] = all_similarities_list
    
    return df

def apply_top_p_selection(df, top_p=0.01):
    """
    Apply top-p selection to DataFrame that already has similarity data.
    
    Args:
        df: DataFrame with columns [all_slang, all_similarities]
        top_p: probability threshold
        
    Returns:
        DataFrame with added columns:
            - relevant_slang: semicolon-separated string of relevant slang
            - relevant_similarities: semicolon-separated string of relevant scores
            - relevant_count: number of relevant slang words
    """
    df = df.copy()
    
    # Prepare lists to collect data
    relevant_slang_list = []
    relevant_similarities_list = []
    relevant_count_list = []
    
    print(f"Applying top-p selection (top_p={top_p})...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying top-p"):
        # Parse similarities
        all_similarities = np.array([float(x) for x in row['all_similarities'].split(';')])
        
        # Get top-p indices
        top_p_indices = get_top_p_indices(all_similarities, top_p=top_p)
        
        # Parse slang and neutral text lists
        all_slang_split = row['all_slang'].split(';')
        
        # Extract relevant items
        relevant_slang = [all_slang_split[i] for i in top_p_indices]
        relevant_sims = [all_similarities[i] for i in top_p_indices]
        
        # Store
        relevant_slang_list.append(';'.join(relevant_slang))
        relevant_similarities_list.append(';'.join([f"{sim:.6f}" for sim in relevant_sims]))
        relevant_count_list.append(len(relevant_slang))
    
    # Add columns to dataframe
    df['relevant_slang'] = relevant_slang_list
    df['relevant_similarities'] = relevant_similarities_list
    df['relevant_count'] = relevant_count_list
    
    print(f"✓ Applied top-p selection")
    print(f"  Added columns: ['relevant_slang', 'relevant_similarities', 'relevant_count']\n")
    
    return df


def apply_top_k_selection(df, max_k=6):
    """
    Select top-k slang words based on similarity scores from the relevant_slang column.
    
    Args:
        df: DataFrame with 'relevant_slang' and 'relevant_similarities' columns
        max_k: maximum number of slang words to select
    Returns:
        DataFrame with added columns:
            - top_k_slang: semicolon-separated string of top-k slang
            - top_k_slang_count: number of top-k slang words
    """
    df = df.copy()
    top_k_slang_list = []
    top_k_slang_count_list = []
    print(f"Selecting top-{max_k} slang words...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Selecting top-k slang"):
        slang_split = [s.strip() for s in row['all_slang'].split(";") if s.strip()]
        all_similarities = np.array([float(x) for x in row['all_similarities'].split(';')])
        
        # Combine and sort by similarity
        slang_sim_pairs = list(zip(slang_split, all_similarities))
        slang_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k
        top_k_pairs = slang_sim_pairs[:max_k]
        top_k_slang = [pair[0] for pair in top_k_pairs]
        
        top_k_slang_list.append("; ".join(top_k_slang))
        top_k_slang_count_list.append(len(top_k_slang))
        
    df['top_k_slang'] = top_k_slang_list
    df['top_k_slang_count'] = top_k_slang_count_list
    
    print(f"✓ Selected top-{max_k} slang words")
    print(f"  Added columns: ['top_k_slang', 'top_k_slang_count']\n")
    return df


# Maybe delete this function later if not used
def sample_slang_column(df, sample_ratio=0.7):
    """
    Sample a subset of slang from the relevant_slang column.
    
    Args:
        df: DataFrame with 'relevant_slang' column
        sample_ratio: proportion to sample (0.0 to 1.0)
        
    Returns:
        DataFrame with added columns:
            - sampled_slang: semicolon-separated string of sampled slang
            - sampled_slang_count: number of sampled slang words
    """
    df = df.copy()
    
    sampled_slang_list = []
    sampled_slang_count_list = []
    
    print(f"Sampling slang (sample_ratio={sample_ratio})...")
    for slang_string in tqdm(df['relevant_slang'], desc="Sampling slang"):
        slang_split = [s.strip() for s in slang_string.split(";") if s.strip()]
        n_samples = max(1, int(len(slang_split) * sample_ratio))
        sampled = np.random.choice(slang_split, size=n_samples, replace=False)
        
        sampled_slang_list.append("; ".join(sampled))
        sampled_slang_count_list.append(len(sampled))
    
    # Add columns to dataframe
    df['sampled_slang'] = sampled_slang_list
    df['sampled_slang_count'] = sampled_slang_count_list
    
    print(f"✓ Sampled slang")
    print(f"  Added columns: ['sampled_slang', 'sampled_slang_count']\n")
    
    return df

def analyze_similarity_distribution_simple(df, row_idx=0, bins=40):
    """
    Simple analysis of similarity distribution for one TL;DR post.
    - Histogram of similarities
    - Annotated with mean, std, min, max
    """
    row = df.iloc[row_idx]
    similarities = np.array([float(x) for x in row['all_similarities'].split(';')])

    mean_sim = similarities.mean()
    std_sim = similarities.std()
    sim_min = similarities.min()
    sim_max = similarities.max()
    print(f"Similarity stats - Count: {len(similarities)}, Mean: {mean_sim:.4f}, Std: {std_sim:.4f}, Min: {sim_min:.4f}, Max: {sim_max:.4f}")

    plt.figure(figsize=(7, 4))
    plt.hist(similarities, bins=bins, color='steelblue',
             edgecolor='black', alpha=0.8)
    plt.xlabel("Normalized similarity", fontsize=11)
    plt.ylabel("Count", fontsize=11)

    completion_text = row['completion'][:80] + "..."
    plt.title(f"Similarity distribution for one TL;DR\n{completion_text}",
              fontsize=11)

    # Text box with tightness info
    text = (
        f"Mean: {mean_sim:.3f}\n"
        f"Std dev: {std_sim:.3f}\n"
        f"Min: {sim_min:.3f}, Max: {sim_max:.3f}\n"
        f"Range: {sim_max - sim_min:.3f}"
    )
    plt.gca().text(0.98, 0.95, text, transform=plt.gca().transAxes,
                   ha="right", va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                   fontsize=9)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "mean": mean_sim,
        "std": std_sim,
        "min": sim_min,
        "max": sim_max,
        "range": sim_max - sim_min,
    }