import json
import pandas as pd
from typing import Optional


def merge_results_to_csv(
    input_file: str,
    csv: str,
    output_filepath: str,
    target_column: str = 'GenZ_completion'
) -> int:
    """
    Merge CSV batch results into a base CSV file by direct index mapping.
    
    Args:
        input_file: Path to input batch CSV file (to merge from)
        csv: Path to base CSV file (to merge into) - can be with or without .csv extension
        output_filepath: Path to save merged output
        target_column: Column name to store merged results
    
    Returns:
        Number of rows successfully merged
    
    Example:
        merge_results_to_csv(
            input_file='tldr-genz-batch-000000-010000.csv',
            csv='tldr_top_k',
            output_filepath='tldr.csv'
        )
    """
    # Handle csv path without extension
    csv_filepath = csv if csv.endswith('.csv') else f'{csv}.csv'
    
    # 1. Load the Data
    print(f"Loading batch CSV from {input_file}...")
    df_batch = pd.read_csv(input_file)
    
    print(f"Loading base CSV from {csv_filepath}...")
    df_base = pd.read_csv(csv_filepath)

    # 2. Prepare for Merge
    if target_column not in df_base.columns:
        df_base[target_column] = None
        print(f"Created new column: {target_column}")

    # 3. Merge batch data into base
    updates_count = 0
    
    print("\nStarting Merge...")
    for idx, row in df_batch.iterrows():
        try:
            # Direct index mapping
            csv_idx = idx
            
            if csv_idx in df_base.index:
                # Get the value from batch - try parsed_text if available, otherwise use full row
                value = row.get('parsed_text', row.get('model_output', None))
                df_base.at[csv_idx, target_column] = value
                updates_count += 1
                
        except Exception as e:
            print(f"⚠️ Error at index {idx}: {e}")

    # 4. Save Result
    df_base.to_csv(output_filepath, index=False, quoting=1, escapechar='\\', encoding='utf-8')

    print(f"\n✓ Success! Merged {updates_count} rows.")
    print(f"✓ Saved to {output_filepath}")

    # 5. Verification Preview
    print("\nPreview of mapped data:")
    cols_to_show = [col for col in [target_column, 'completion'] if col in df_base.columns]
    
    updated_rows = df_base[df_base[target_column].notna()]
    if not updated_rows.empty:
        print(updated_rows[cols_to_show].head(5))
    else:
        print(df_base[cols_to_show].head(5))
    
    return updates_count