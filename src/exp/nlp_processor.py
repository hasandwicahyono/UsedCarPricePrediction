from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Converts text columns into dense embeddings using sentence-transformers.
    Useful for descriptions or unstructured textual metadata in car listings.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        import torch
        from sentence_transformers import SentenceTransformer
        
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Determine embedding dimension
        dummy_emb = self.model.encode(["test"], show_progress_bar=False)
        self.emb_dim_ = dummy_emb.shape[1]
        self.feature_names_out_ = [f"text_emb_{i}" for i in range(self.emb_dim_)]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Concatenate multiple text columns if provided into a single context
            text_data = X.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()
        else:
            text_data = list(X)

        embeddings = self.model.encode(
            text_data,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
             return np.array([], dtype=object)
        return np.array(self.feature_names_out_, dtype=object)