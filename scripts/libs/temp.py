import matplotlib.pyplot as plt
import numpy as np

def plot_tldr_slang_distribution(df, row_idx=0, top_n=30):
    """
    Plot similarity scores for each slang word in a single TL;DR post.
    Slang words are displayed vertically (y-axis), similarities on x-axis.
    Highest similarity scores appear at the TOP.
    Displays the original TL;DR completion text above the chart.
    """
    # Extract data from the selected row
    row = df.iloc[row_idx]
    
    # Parse slang words and similarities
    slang_list = row['all_slang'].split(';')
    similarities = np.array([float(x) for x in row['all_similarities'].split(';')])
    
    # Create pairs and sort by similarity (descending)
    slang_sim_pairs = list(zip(slang_list, similarities))
    slang_sim_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_pairs = slang_sim_pairs[:top_n]
    top_slang = [pair[0] for pair in top_pairs]
    top_sims = [pair[1] for pair in top_pairs]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(8, len(top_slang) * 0.25)))
    
    # Plot bars (keep natural order - highest at top)
    bars = ax.barh(range(len(top_slang)), top_sims, color='steelblue', edgecolor='black', alpha=0.8)
    
    # Set y-axis labels to slang words
    ax.set_yticks(range(len(top_slang)))
    ax.set_yticklabels(top_slang, fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Normalized Similarity Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Slang Word Similarity Distribution: TL;DR Post #{row_idx}\n', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, sim) in enumerate(zip(bars, top_sims)):
        ax.text(sim + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{sim:.4f}', va='center', fontsize=8)
    
    ax.set_xlim(0, max(top_sims) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Add original TL;DR completion text above the chart
    completion_text = row['completion'][:100] + '...'
    ax.text(0.5, 1.02, completion_text, 
            transform=ax.transAxes, fontsize=10, ha='center', va='bottom', 
            wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return top_slang, top_sims


# Usage
plot_tldr_slang_distribution(df_with_sims, row_idx=0, top_n=30)
