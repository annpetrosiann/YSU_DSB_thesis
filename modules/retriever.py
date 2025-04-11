from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words("english"))

class Retriever:
    def __init__(self, embedder, faiss_index, texts, chunk_ids,chunk_titles=None):
        """
        Initialize the Retriever with an embedder, FAISS index, texts, and chunk IDs.
        """
        self.embedder = embedder
        self.faiss_index = faiss_index  # Use FAISS index directly
        self.texts = texts
        self.ids = chunk_ids
        self.titles = chunk_titles if chunk_titles else [None]*len(texts)

    def keyword_score(self, query: str, text: str) -> float:
        query_words = set(re.findall(r"\w+", query.lower())) - stop_words
        text_words = set(re.findall(r"\w+", text.lower()))
        return len(query_words.intersection(text_words)) / (len(query_words) + 1e-5)

    def retrieve(self, query: str, top_k=5) -> list:
        """
        Retrieve the top-k most relevant chunks using FAISS and keyword scoring.
        """
        query_embedding = self.embedder.encode([query]).astype("float32")  # FAISS requires float32
        distances, indices = self.faiss_index.search(query_embedding, top_k * 2)  # Retrieve top-k*2 candidates

        scored = []
        for idx, dist in zip(indices[0], distances[0]):  # Iterate over results
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            idx = int(idx)  # Convert NumPy index to Python integer
            chunk_text = self.texts[idx]
            score = self.keyword_score(query, chunk_text)
            scored.append((idx, score, dist))

        # Sort by keyword match score, then by FAISS distance as a fallback
        scored = sorted(scored, key=lambda x: (-x[1], x[2]))
        # Prepare results with metadata
        results = []
        for i, (idx, score, dist) in enumerate(scored[:top_k]):
            results.append({
                "text": self.texts[idx],
                "id": self.ids[idx],
                "title": self.titles[idx],
                "score": float(score),
                "distance": float(dist),
                "rank": i+1
            })

        return results
