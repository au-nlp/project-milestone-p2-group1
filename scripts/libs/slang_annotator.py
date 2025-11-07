import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer

class SlangAnnotator:
    """
    Annotate TL;DR completions with Gen Z slang words using embeddings.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", genz_data=None):
        """
        Initialize the annotator with embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.genz_data = genz_data
        self.slang_embeddings = self.model.encode(
            genz_data['Neutral_Text'].tolist(), 
            show_progress_bar=True
        )
    
    def _get_top_p_indices(self, similarities, top_p=0.9):
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
    
    def get_relevant_slang(self, tldr_text, top_p=0.01):
        """
        Get relevant slang words for a single TL;DR completion.
        Returns: tuple of (slang_string, count)
        """
        tldr_embedding = self.model.encode([tldr_text])[0]
        similarities = cosine_similarity([tldr_embedding], self.slang_embeddings)[0]
        top_p_indices = self._get_top_p_indices(similarities, top_p=top_p)
        
        slang_list = self.genz_data.iloc[top_p_indices]['Slang'].tolist()
        slang_string = "; ".join(slang_list)
        count = len(slang_list)
        
        return slang_string, count
    
    @staticmethod
    def sample_slang(slang_string, sample_ratio=0.7):
        """
        Sample a subset of slang words from the semi-colon separated string.
        Returns: tuple of (sampled_string, count)
        """
        slang_list = [s.strip() for s in slang_string.split(";") if s.strip()]
        n_samples = max(1, int(len(slang_list) * sample_ratio))
        sampled = np.random.choice(slang_list, size=n_samples, replace=False)
        sampled_string = "; ".join(sampled)
        count = len(sampled)
        
        return sampled_string, count

    def annotate_dataset(self, df, top_p=0.01, sample_ratio=0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Annotate entire dataset with slang words.
        Returns: (annotated_df, stats_dict)
        """
        df = df.copy()
        
        # Step 1-5: Get relevant slang
        relevant_slang_list = []
        relevant_slang_count_list = []
        
        for completion in tqdm(df['completion'], desc="Computing relevant slang"):
            slang, count = self.get_relevant_slang(completion, top_p=top_p)
            relevant_slang_list.append(slang)
            relevant_slang_count_list.append(count)
        
        df['relevant_slang'] = relevant_slang_list
        df['relevant_slang_count'] = relevant_slang_count_list
        
        # Step 6: Sample slang
        sampled_slang_list = []
        sampled_slang_count_list = []
        
        for slang_string in tqdm(df['relevant_slang'], desc="Sampling slang"):
            sampled, count = self.sample_slang(slang_string, sample_ratio=sample_ratio)
            sampled_slang_list.append(sampled)
            sampled_slang_count_list.append(count)
        
        df['sampled_slang'] = sampled_slang_list
        df['sampled_slang_count'] = sampled_slang_count_list
        
        # Calculate statistics
        df_stats = pd.DataFrame({
            'relevant_slang': {
                'mean': np.mean(relevant_slang_count_list),
                'min': np.min(relevant_slang_count_list),
                'max': np.max(relevant_slang_count_list),
                'std': np.std(relevant_slang_count_list)
            },
            'sampled_slang': {
                'mean': np.mean(sampled_slang_count_list),
                'min': np.min(sampled_slang_count_list),
                'max': np.max(sampled_slang_count_list),
                'std': np.std(sampled_slang_count_list)
            }
        })

        return df, df_stats

