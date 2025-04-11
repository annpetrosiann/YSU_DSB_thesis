from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, List, Union

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: Optional[int] = 128):
        """
        Initialize the text embedder with a SentenceTransformer model.

        Args:
            model_name: Name of the SentenceTransformer model
            batch_size: Default batch size for encoding (can be overridden in encode method)
        """
        self.model = SentenceTransformer(model_name)
        self.default_batch_size = batch_size

    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode input texts into embeddings.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding (overrides default if specified)

        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use provided batch_size or fall back to default
        current_batch_size = batch_size if batch_size is not None else self.default_batch_size

        return self.model.encode(
            texts,
            batch_size=current_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False  # Typically you want this off in production
        )