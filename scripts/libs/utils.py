from langdetect import detect
import numpy as np
import pandas as pd
from unicodedata import normalize
import regex as re

def detect_language(text, max_len=1000):
    """Detect language of text sample"""
    try:
        return detect(text[:max_len])
    except:
        return "unk"

def sample_lang_distribution(ds, n=2000):
    """Sample language distribution from dataset by concatenating all columns/fields of each example"""
    sample_size = min(n, len(ds))
    indices = np.random.choice(len(ds), size=sample_size, replace=False)

    langs = []
    for idx in indices:
        example = ds.iloc[int(idx)] if isinstance(ds, pd.DataFrame) else ds[int(idx)]

        # build text from all fields/columns
        if isinstance(example, str):
            text = example
        elif isinstance(example, pd.Series):
            values = example.values
            parts = [str(v).strip() for v in values if isinstance(v, str) and v.strip()]
            if not parts:
                parts = [str(v).strip() for v in values if not pd.isna(v) and str(v).strip()]
            text = " ".join(parts)
        elif isinstance(example, dict):
            values = example.values()
            parts = [str(v).strip() for v in values if isinstance(v, str) and v.strip()]
            if not parts:
                parts = [str(v).strip() for v in values if not pd.isna(v) and str(v).strip()]
            text = " ".join(parts)
        else:
            # fallback stringify
            text = str(example)

        langs.append(detect_language(text) if text and text.strip() else "unk")

    lang_counts = pd.Series(langs).value_counts(normalize=True)
    return lang_counts

def normalize_text(example):
    """Normalize text using NFKC and whitespace collapsing"""
    for field in ["prompt", "completion"]:
        text = example[field]
        text = "" if text is None else str(text)
        text = normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        example[field] = text
    return example

def compute_word_length(text):
    """Compute word count from text"""
    if pd.isna(text):
        return 0
    return len(str(text).split())

def intersection_over_union_unigram(a, b):
    """Calculate Jaccard similarity between two texts"""
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
