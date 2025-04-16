from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import Optional, List, Union

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: Optional[int] = 128, fine_tuned_model: Optional[SentenceTransformer] = None, use_huggingface: bool = False):
        """
        Initialize the text embedder with a SentenceTransformer or HuggingFace model.

        Args:
            model_name: Name of the SentenceTransformer model (used if no fine-tuned model is provided)
            batch_size: Default batch size for encoding (can be overridden in encode method)
            fine_tuned_model: Optional fine-tuned model to use for embeddings (overrides model_name)
            use_huggingface: Flag to use Hugging Face model instead of SentenceTransformer
        """
        self.use_huggingface = use_huggingface

        if self.use_huggingface:
            # If using Hugging Face model, initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        else:
            # Otherwise, use SentenceTransformer
            self.model = fine_tuned_model if fine_tuned_model else SentenceTransformer(model_name)

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

        current_batch_size = batch_size if batch_size is not None else self.default_batch_size

        if self.use_huggingface:
            # Tokenizing the texts for Hugging Face model
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Averaging token embeddings to get a fixed-size representation

        else:
            embeddings = self.model.encode(
                texts,
                batch_size=current_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False  # Typically you want this off in production
            )

        return embeddings.astype("float32")  # Ensure embeddings are in float32 format for FAISS
