import pandas as pd
import matplotlib.pyplot as plt
from scripts.libs.tldr_stats import compute_tldr_stats


def plot_tldr_analysis(tldr_stats):
    """
    Generate and save visualizations for TL;DR text length analysis.
    
    Args:
        df_stats: DataFrame containing 'prompt' and 'completion' columns
    """
    # Compute statistics
    prompt_len_stats = {
        "mean": tldr_stats["prompt_word_length"].mean(),
        "std_dev": tldr_stats["prompt_word_length"].std(),
    }
    completion_len_stats = {
        "mean": tldr_stats["completion_word_length"].mean(),
        "std_dev": tldr_stats["completion_word_length"].std(),
    }
    
    # Create figure
    fig, axes = plt.subplots(2, figsize=(14, 10))
    fig.suptitle("TL;DR Dataset Text Length Analysis", fontsize=16, fontweight="bold")
    
    # --- Plot 1: Overlaid histograms ---
    ax1 = axes[0]
    ax1.hist(tldr_stats["prompt_word_length"], bins=50, alpha=0.6, label="Prompt length (words)")
    ax1.hist(tldr_stats["completion_word_length"], bins=50, alpha=0.6, label="Completion length (words)")
    ax1.set_xlabel("Number of words", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of Text Lengths", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # --- Plot 2: Box plots for comparison ---
    ax2 = axes[1]
    box_data = [tldr_stats["prompt_word_length"], tldr_stats["completion_word_length"]]
    bp = ax2.boxplot(box_data, labels=["Prompt", "Completion"], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightsalmon']):
        patch.set_facecolor(color)
    ax2.set_ylabel("Number of words", fontsize=11)
    ax2.set_title("Text Length Box Plot Comparison", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
