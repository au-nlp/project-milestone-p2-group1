"""
rag_utils.py

Utility functions for checkpoint management and data saving.
"""

import pandas as pd
import os
from datetime import datetime
import logging




# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(results: list, processed_count: int, reason: str = "checkpoint", 
                   checkpoint_dir: str = "checkpoints") -> str:
    """
    Save results to checkpoint CSV.
    
    Args:
        results: List of result dicts
        processed_count: Number of items processed
        reason: Reason for checkpoint (e.g., "regular", "error", "final")
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Path to saved checkpoint file
    """
    if not results:
        print("No results to save")
        return None
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(
        checkpoint_dir,
        f"checkpoint_{reason}_{processed_count:06d}_{timestamp}.csv"
    )
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(checkpoint_file, index=False)
    
    print(f" Checkpoint: {checkpoint_file}")
    print(f"  Processed: {processed_count} datapoints")
    
    return checkpoint_file


def load_last_checkpoint(checkpoint_dir: str = "checkpoints") -> tuple:
    """
    Load the last checkpoint to resume.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Tuple of (results_list, processed_count) or (None, 0) if no checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoint_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.csv')
    ])
    
    if not checkpoint_files:
        print("No previous checkpoints found")
        return None, 0
    
    last_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
    
    df_checkpoint = pd.read_csv(checkpoint_path)
    results = df_checkpoint.to_dict('records')
    processed_count = len(results)
    
    print(f" Loaded checkpoint: {last_checkpoint}")
    print(f"  Resuming from: {processed_count} datapoints")
    
    return results, processed_count


# ============================================================================
# DATA SAVING
# ============================================================================

def save_final_results(results: list, filename: str = None) -> str:
    """
    Save full detailed results CSV.
    
    Args:
        results: List of result dicts
        filename: Output filename (auto-generated if None)
        
    Returns:
        Path to saved file
    """
    if not results:
        print("No results to save")
        return None
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tldr_to_genz_fast_{timestamp}.csv"
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False)
    
    print(f"✓ Final results: {filename} ({len(df_results)} datapoints)")
    
    return filename


def save_to_tldr_dataset(tldr_df: pd.DataFrame, results: list, 
                        output_file: str = None) -> pd.DataFrame:
    """
    Merge Gen Z outputs into TLDR dataset.
    
    Adds two columns:
    - genz_completion: Best Gen Z output
    - genz_score: Hybrid score
    
    Args:
        tldr_df: Original TLDR DataFrame
        results: List of result dicts from RAG
        output_file: Output filename (auto-generated if None)
        
    Returns:
        Enhanced DataFrame (also saved to CSV)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tldr_with_genz_fast_{timestamp}.csv"
    
    df_enhanced = tldr_df.copy()
    result_map = {r['tldr_completion']: r for r in results}
    
    def extract_completion(completion):
        """Normalize TLDR completion format."""
        if isinstance(completion, dict):
            return completion.get('content', str(completion))
        elif isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict):
                return completion[0].get('content', str(completion))
            else:
                return ' '.join([str(c) for c in completion])
        return str(completion)
    
    completion_texts = df_enhanced['completion'].apply(extract_completion)
    
    # Add only best output and score
    df_enhanced['genz_completion'] = completion_texts.map(
        lambda x: result_map.get(x, {}).get('best_genz_output', '')
    )
    df_enhanced['genz_score'] = completion_texts.map(
        lambda x: result_map.get(x, {}).get('best_hybrid_score', 0.0)
    )
    
    df_enhanced.to_csv(output_file, index=False)
    print(f"✓ Enhanced TLDR: {output_file}")
    print(f"  Columns added: genz_completion, genz_score")
    
    return df_enhanced


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def extract_completion(completion) -> str:
    """Normalize various TLDR completion formats to text."""
    if isinstance(completion, dict):
        return completion.get('content', str(completion))
    elif isinstance(completion, list) and len(completion) > 0:
        if isinstance(completion[0], dict):
            return completion[0].get('content', str(completion))
        else:
            return ' '.join([str(c) for c in completion])
    return str(completion)
