from collections import Counter
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
import pandas as pd
import sacrebleu
from scripts.libs.utils import compute_word_length, intersection_over_union_unigram

def compute_tldr_stats(df_tldr):
    """Compute TLDR statistics and return dataframes"""
    # Create dataframe with all word length data
    tldr_stats = pd.DataFrame({
        "prompt_word_length": df_tldr["prompt"].apply(compute_word_length),
        "completion_word_length": df_tldr["completion"].apply(compute_word_length),
    })
    
    tldr_stats["compression_ratio"] = tldr_stats["prompt_word_length"] / (tldr_stats["completion_word_length"] + 1e-6)
    
    # Create statistics summary dataframe
    stats_summary = pd.DataFrame({
        "Metric": ["Mean", "Variance", "Std Dev", "Coefficient of Variation", "Max", "Min"],
        "Prompt Length": [
            tldr_stats["prompt_word_length"].mean(),
            tldr_stats["prompt_word_length"].var(),
            tldr_stats["prompt_word_length"].std(),
            tldr_stats["prompt_word_length"].std() / tldr_stats["prompt_word_length"].mean(),
            tldr_stats["prompt_word_length"].max(),
            tldr_stats["prompt_word_length"].min()
        ],
        "Completion Length": [
            tldr_stats["completion_word_length"].mean(),
            tldr_stats["completion_word_length"].var(),
            tldr_stats["completion_word_length"].std(),
            tldr_stats["completion_word_length"].std() / tldr_stats["completion_word_length"].mean(),
            tldr_stats["completion_word_length"].max(),
            tldr_stats["completion_word_length"].min()
        ],
        "Compression Ratio": [
            tldr_stats["compression_ratio"].mean(),
            tldr_stats["compression_ratio"].var(),
            tldr_stats["compression_ratio"].std(),
            tldr_stats["compression_ratio"].std() / tldr_stats["compression_ratio"].mean(),
            tldr_stats["compression_ratio"].max(),
            tldr_stats["compression_ratio"].min()
        ]
    })
    
    return tldr_stats, stats_summary




def compute_alignment_stats(df):
    """
    Compute alignment metrics (Jaccard IoU and length ratio) for each example
    in the input DataFrame, without modifying the original DF.
    
    Returns a new DataFrame with 'iou' and 'len_ratio' columns.
    """
    iou_list = []
    len_ratio_list = []
    
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        completion = row["completion"]
        iou = intersection_over_union_unigram(prompt, completion)
        len_ratio = len(completion) / (len(prompt) + 1e-6)
        iou_list.append(iou)
        len_ratio_list.append(len_ratio)
    
    alignment_stats = pd.DataFrame({
        "iou": iou_list,
        "len_ratio": len_ratio_list
    })
    
    return alignment_stats



def tokenize_texts(df_tldr, tldr_token_stats, tokenizer):
    """Tokenize prompts and completions"""
    prompt_tok_lens = []
    completion_tok_lens = []

    for _, row in df_tldr.iterrows():
        prompt_tokens = tokenizer.encode(row['prompt'], add_special_tokens=False)
        completion_tokens = tokenizer.encode(row['completion'], add_special_tokens=False)
        prompt_tok_lens.append(len(prompt_tokens))
        completion_tok_lens.append(len(completion_tokens))

    token_stats_df = pd.DataFrame({
        "prompt_tok_len": prompt_tok_lens,
        "completion_tok_len": completion_tok_lens,
    })
    token_stats_df["total_tok_len"] = token_stats_df["prompt_tok_len"] + token_stats_df["completion_tok_len"]

    # Calculate summary statistics
    summary_dict = {
        "prompt_tok_mean": np.mean(token_stats_df["prompt_tok_len"]),
        "prompt_tok_p95": np.percentile(token_stats_df["prompt_tok_len"], 95),
        "completion_tok_mean": np.mean(token_stats_df["completion_tok_len"]),
        "completion_tok_p95": np.percentile(token_stats_df["completion_tok_len"], 95),
        "total_tok_p95": np.percentile(token_stats_df["total_tok_len"], 95),
    }

    summary_df = pd.DataFrame(list(summary_dict.items()), columns=["Metric", "Value"])
    summary_df["Value"] = summary_df["Value"].map(lambda x: f"{x:.2f}")

    return token_stats_df, summary_df

def echo_baseline(prompt, tokenizer, max_tokens=32):
    """Simple baseline that outputs truncated prompt decoded"""
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:max_tokens])

def evaluate_echo_baseline(test_ds, tokenizer, sample_size=1000):
    """Evaluate echo baseline on a test dataset."""
    
    n_samples = min(sample_size, len(test_ds))
    sample_indices = np.random.choice(len(test_ds), size=n_samples, replace=False)
    test_sample = test_ds.select(sample_indices)
    
    preds = [echo_baseline(ex["prompt"], tokenizer) for ex in test_sample]
    refs = test_sample["completion"]
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    rouge_scores = defaultdict(list)
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
    
    rouge_means = {
        "rouge1": np.mean(rouge_scores["rouge1"]),
        "rouge2": np.mean(rouge_scores["rouge2"]),
        "rougeL": np.mean(rouge_scores["rougeL"]),
    }
    
    chrf_score = sacrebleu.corpus_chrf(preds, [refs]).score
    
    # Prepare summary DataFrame
    metrics_dict = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "chrF"],
        "Value": [rouge_means["rouge1"], rouge_means["rouge2"], rouge_means["rougeL"], chrf_score]
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df["Value"] = metrics_df["Value"].map(lambda x: f"{x:.4f}")
    
    return metrics_df