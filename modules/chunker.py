import os
import json
from typing import List, Tuple, Optional
import nltk
import spacy

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

class TextChunker:
    def __init__(self, method: str = "spacy", max_words: int = 150, overlap: int = 20):
        """
        :param method: "nltk" or "spacy"
        :param max_words: Max words per chunk
        :param overlap: Number of overlapping words between chunks
        """
        self.method = method
        self.overlap = overlap
        self.max_words = max_words
        self.min_chunk_size = max_words // 3

    def load_json_folder(self, folder: str) -> List[Tuple[str, str, Optional[str]]]:
        """Load and chunk all JSON files from a folder."""
        chunks = []
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                with open(os.path.join(folder, filename), "r") as f:
                    try:
                        raw_data = json.load(f)
                        flat_items = self._flatten_nested_list(raw_data)

                        for i, item in enumerate(flat_items):
                            if isinstance(item, dict):
                                text = item.get("data", "")
                                title = item.get("title") or f"{filename}_{i}"
                                if text.strip():
                                    chunks.extend(self.chunk_text(text, title=title))
                            elif isinstance(item, str):
                                chunks.extend(self.chunk_text(item, title=f"{filename}_{i}"))

                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        return chunks

    def _flatten_nested_list(self, data):
        """Recursively flatten nested lists."""
        flat = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    flat.extend(self._flatten_nested_list(item))
                else:
                    flat.append(item)
        else:
            flat.append(data)
        return flat

    def chunk_text(self, text: str, title: str) -> List[Tuple[str, str, str]]:
        """
        Chunk text with title preservation.
        Returns:
            List of (chunk_id, chunk_text, title) tuples
            where chunk_id = f"{title}_{chunk_number}"
        """
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        word_count = 0
        chunk_number = 0
        overlap_buffer = []

        for sent in sentences:
            words = sent.split()
            current_chunk.extend(words)
            word_count += len(words)

            if word_count >= self.max_words:
                chunk_text = " ".join(current_chunk[:self.max_words])
                overlap_buffer = current_chunk[self.max_words - self.overlap:self.max_words]

                chunks.append((
                    f"{title}_{chunk_number}",  # chunk_id combines title and number
                    chunk_text,
                    title  # Keep original title
                ))
                chunk_number += 1

                current_chunk = overlap_buffer + current_chunk[self.max_words:]
                word_count = len(current_chunk)
                overlap_buffer = []

        if word_count >= self.min_chunk_size:
            chunks.append((
                f"{title}_{chunk_number}",
                " ".join(current_chunk),
                title
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            if self.method == "nltk":
                return nltk.sent_tokenize(text)
            elif self.method == "spacy":
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents]
            else:
                raise ValueError("Unsupported method. Choose 'nltk' or 'spacy'.")
        except Exception as e:
            print(f"Sentence splitting error: {str(e)}")
            return [text]