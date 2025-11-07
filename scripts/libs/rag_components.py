"""
rag_components.py

Reusable components for Gen Z style transfer RAG pipeline.
Import these into Jupyter notebooks or scripts.
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================================
# STYLE EVALUATOR
# ============================================================================

class DataDrivenGenZEvaluator:
    """
    Score text authenticity based on explicit Gen Z slang terms.
    
    Usage:
        evaluator = DataDrivenGenZEvaluator(df_genz)
        score = evaluator.score_style_authenticity("Yo I'm hella tired fr fr")
    """
    
    def __init__(self, df_genz: pd.DataFrame):
        """
        Extract slang terms from 'Slang' column.
        
        Args:
            df_genz: DataFrame with 'Slang' column containing Gen Z terms
        """
        self.slang_terms = set()
        
        for slang in df_genz['Slang'].fillna(''):
            slang_clean = slang.strip().strip('"').lower()
            if slang_clean and len(slang_clean) > 1:
                self.slang_terms.add(slang_clean)
        
        print(f"✓ Loaded {len(self.slang_terms)} explicit Gen Z slang terms")
        
        sample = sorted(list(self.slang_terms))[:20]
        print(f"Sample: {', '.join(sample)}")
    
    def score_style_authenticity(self, text: str) -> float:
        """
        Score how Gen Z the text sounds (0-1).
        
        Higher score = more Gen Z slang terms detected.
        """
        if not text or len(text) < 3:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for term in self.slang_terms if term in text_lower)
        
        if not self.slang_terms:
            return 0.0
        
        score = min(1.0, matches / max(1, len(self.slang_terms) / 50))
        return max(0.0, min(1.0, score))


# ============================================================================
# HYBRID SCORER
# ============================================================================

class HybridScorer:
    """
    Combine semantic similarity + style authenticity.
    
    Usage:
        scorer = HybridScorer(evaluator, similarity_weight=0.5, style_weight=0.5)
        hybrid_score = scorer.score(0.75, "Yo fr fr no cap")
    """
    
    def __init__(self, style_evaluator: DataDrivenGenZEvaluator, 
                 similarity_weight: float = 0.5, style_weight: float = 0.5):
        self.style_evaluator = style_evaluator
        
        # Normalize weights
        total = similarity_weight + style_weight
        self.similarity_weight = similarity_weight / total
        self.style_weight = style_weight / total
    
    def score(self, semantic_similarity: float, text: str) -> float:
        """
        Calculate hybrid score (0-1).
        
        Args:
            semantic_similarity: Score from embedding comparison (0-1)
            text: Generated text to evaluate
            
        Returns:
            Weighted combination of semantic + style scores
        """
        style_score = self.style_evaluator.score_style_authenticity(text)
        hybrid = (self.similarity_weight * semantic_similarity) + (self.style_weight * style_score)
        return max(0.0, min(1.0, hybrid))


# ============================================================================
# MISTRAL CLIENT
# ============================================================================

class MistralOllamaClient:
    """
    Generate text using local Mistral via Ollama API.
    
    Usage:
        client = MistralOllamaClient()
        text = client.generate("Convert to Gen Z: I am very tired")
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 model: str = "mistral", temperature: float = 0.7, 
                 top_p: float = 0.9, timeout: int = 120):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.health_check()
    
    def health_check(self) -> bool:
        """Verify Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/list", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama is running at {self.base_url}")
                return True
            return False
        except Exception as e:
            print(f"Cannot connect to Ollama: {e}")
            return False
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text using Mistral."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": stream,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()['response'].strip()
                return result.split('\n')[0]
            return ""
        except Exception as e:
            print(f"Generation error: {e}")
            return ""


# ============================================================================
# RAG PIPELINE
# ============================================================================

class GenZSlangRAG:
    """
    Main RAG pipeline for Gen Z style transfer.
    
    - Retrieves similar examples from indexed data
    - Generates Gen Z paraphrases
    - Scores with hybrid metric
    - Handles checkpointing for recovery
    
    Usage:
        rag = GenZSlangRAG(df_genz)
        rag.index_examples(df_genz)
        result = rag.process_one("I am very tired")
    """
    
    def __init__(self, df_genz: pd.DataFrame, embedding_model_name: str = "all-MiniLM-L6-v2",
                 ollama_base_url: str = "http://localhost:11434", ollama_model: str = "mistral",
                 top_k: int = 5):
        """Initialize RAG pipeline."""
        self.top_k = top_k
        
        print("Initializing Mistral Ollama client...")
        self.mistral = MistralOllamaClient(base_url=ollama_base_url, model=ollama_model)
        
        print(f"Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)
        
        print("Initializing style evaluator...")
        self.style_evaluator = DataDrivenGenZEvaluator(df_genz)
        self.hybrid_scorer = HybridScorer(self.style_evaluator, 0.5, 0.5)
        
        self.examples = []
        self.embeddings = None
        self.index = None
        self.results = []
        self.processed_count = 0
    
    def index_examples(self, df: pd.DataFrame):
        """Build FAISS index from examples."""
        print(f"Indexing {len(df)} examples...")
        
        self.examples = [
            {'neutral': row['Neutral_Text'], 'slang': row['Example']}
            for _, row in df.iterrows()
        ]
        
        neutral_texts = [ex['neutral'] for ex in self.examples]
        self.embeddings = self.embed_model.encode(neutral_texts, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        print(f"✓ Indexed {len(self.examples)} examples")

    def retrieve_similar_examples(self, query: str) -> List[Dict]:
        """Retrieve top-k similar examples."""
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        similar = []
        for idx, score in zip(indices[0], scores[0]):
            similar.append({
                'neutral': self.examples[idx]['neutral'],
                'slang': self.examples[idx]['slang'],
            })
        
        return similar
    
    def evaluate_semantic_similarity(self, input_text: str, output: str) -> float:
        """Calculate semantic similarity (0-1)."""
        if not output or len(output.strip()) < 3:
            return 0.0
        
        input_emb = self.embed_model.encode([input_text], convert_to_numpy=True)
        output_emb = self.embed_model.encode([output], convert_to_numpy=True)
        
        faiss.normalize_L2(input_emb)
        faiss.normalize_L2(output_emb)
        
        similarity = float(np.dot(input_emb[0], output_emb[0]))
        return max(0.0, min(1.0, similarity))
    
    def generate_slang_output(self, query: str, examples: List[Dict]) -> str:
        """Generate Gen Z paraphrase."""
        prompt = "You are an expert at converting formal text into Gen Z slang. Here are examples:\n\n"
        
        for i, ex in enumerate(examples[:3], 1):
            prompt += f"Example {i}:\nFormal: {ex['neutral']}\nGen Z: {ex['slang']}\n\n"
        
        prompt += f"Now convert to Gen Z slang (ONLY the slang sentence, no explanations):\nFormal: {query}\nGen Z:"
        
        output = self.mistral.generate(prompt, stream=False)
        return output
    
    def process_one(self, query: str) -> Dict:
        """
        Process ONE text: retrieve, generate, evaluate, store.
        
        Returns:
            Dict with input, output, and hybrid_score
        """
        examples = self.retrieve_similar_examples(query)
        output = self.generate_slang_output(query, examples)
        semantic_sim = self.evaluate_semantic_similarity(query, output)
        hybrid_score = self.hybrid_scorer.score(semantic_sim, output)
        
        result_row = {
            'tldr_completion': query,
            'retrieved_similar_1': examples[0]['slang'] if len(examples) > 0 else '',
            'retrieved_similar_2': examples[1]['slang'] if len(examples) > 1 else '',
            'retrieved_similar_3': examples[2]['slang'] if len(examples) > 2 else '',
            'best_genz_output': output,
            'best_hybrid_score': hybrid_score,
            'best_semantic_similarity': semantic_sim,
        }
        
        self.results.append(result_row)
        self.processed_count += 1
        
        return {
            'input': query,
            'output': output,
            'score': hybrid_score
        }
