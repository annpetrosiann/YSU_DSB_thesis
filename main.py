import sqlite3  # Add SQLite for local database storage
from modules.chunker import TextChunker
from modules.embedder import TextEmbedder
from modules.retriever import Retriever
from modules.rag_engine import RAGEngine
import pickle  # For saving and loading the model
import faiss  # Import FAISS for vector indexing
from modules.eval_pipeline import RAGEvaluator

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FaissVectorStore:
    """
    A custom vector store using FAISS for indexing and retrieval.
    """
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)  # L2 distance-based FAISS index
        self.chunk_ids = []
        self.chunk_texts = []
        self.chunk_titles = []

    def add(self, embeddings, chunk_ids, chunk_texts,chunk_titles):
        """
        Add embeddings and metadata to the FAISS index.
        """
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        self.chunk_texts.extend(chunk_texts)
        self.chunk_titles.extend(chunk_titles)

    def save(self, index_path="faiss_index.bin", metadata_path="faiss_metadata.pkl"):
        """
        Save the FAISS index and metadata to files.
        """
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "chunk_texts": self.chunk_texts,
                "chunk_titles": self.chunk_titles
            }, f)
        print(f"✅ FAISS index saved to {index_path}")
        print(f"✅ Metadata saved to {metadata_path}")

    def load(self, index_path="faiss_index.bin", metadata_path="faiss_metadata.pkl"):
        """
        Load the FAISS index and metadata from files.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        self.chunk_ids = metadata["chunk_ids"]
        self.chunk_texts = metadata["chunk_texts"]
        self.chunk_titles = metadata.get("chunk_titles", [None]*len(self.chunk_ids))

        print(f"📂 FAISS index loaded from {index_path}")
        print(f"📂 Metadata loaded from {metadata_path}")

def main():
    data_dir = "/Users/annpetrosiann/Desktop/YSU_DSB_thesis/data"

    print("🔍 Loading and chunking data...")
    chunker = TextChunker(method="spacy", max_words=150,overlap=20)
    chunks = chunker.load_json_folder(data_dir)
    chunk_ids, chunk_texts, chunk_titles = zip(*chunks)


    print("📐 Embedding chunks...")
    embedder = TextEmbedder()
    embeddings = embedder.encode(chunk_texts).astype("float32")  # FAISS requires float32

    # Initialize and add data to the FaissVectorStore
    vector_store = FaissVectorStore(dim=embeddings.shape[1])
    vector_store.add(embeddings, chunk_ids, chunk_texts, chunk_titles)

    # Save the FAISS index and metadata
    vector_store.save()

    # Optionally, load the FAISS index and metadata (for demonstration purposes)
    # vector_store.load()

    retriever = Retriever(embedder, vector_store.index, chunk_texts, chunk_ids,chunk_titles)
    rag = RAGEngine(model="llama3", temperature=0.5)
    evaluator = RAGEvaluator()

    print("🧠 RAG system ready. Ask me anything!\n")

    while True:
        query = input("❓ Your question (or 'exit'): ")
        if query.lower() == "exit":
            break
        top_chunks = retriever.retrieve(query, top_k=3)
        prompt = rag.build_prompt(top_chunks, query)
        response = rag.query(prompt)
        print("\n🦙 LLaMA (Markdown output):\n")
        print(response.strip())
        print("\n" + "-" * 60 + "\n")

        # Evaluation block (properly indented)
        feedback = input("⭐️ Do you have a reference answer to evaluate? (y/n): ")
        if feedback.lower().strip() == "y":
            reference = input("🔖 Paste reference answer:\n")
            scores = evaluator.evaluate_batch([response.strip()], [reference])

            print("\n📊 Evaluation Results:")
            for metric, value in scores.items():
                if isinstance(value, list):
                    value = value[0]  # Unpack single-item batch
                print(f"- {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
