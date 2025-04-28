import json
from modules.chunker import TextChunker
from modules.embedder import TextEmbedder
from modules.retriever import Retriever
from modules.rag_engine import RAGEngine
import pickle  # For saving and loading the model
import faiss  # Import FAISS for vector indexing
from modules.eval_pipeline import RAGEvaluator
from modules.fine_tuning_engine import FineTuningEngine

# Prevent tokenizer parallelism warning
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
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    hp = config["hyperparameters"]
    TRAIN_PATH = config["train_path"]
    VAL_PATH = config["val_path"]
    OUTPUT_DIR = config["output_dir"]
    data_dir = config["data_dir"]
    DO_FINETUNE = config.get("finetune", True)

    # Load and chunk data
    print("🔍 Loading and chunking data...")
    chunker = TextChunker(method="spacy", max_words=150,overlap=20)
    chunks = chunker.load_json_folder(data_dir)
    chunk_ids, chunk_texts, chunk_titles = zip(*chunks)

    # Fine-tune or load embedding model
    print("📦 Loading fine-tuned model...")
    if DO_FINETUNE and not os.path.exists(OUTPUT_DIR):
        print("📈 Fine-tuning embedding model...")
        fine_tuning_engine = FineTuningEngine(train_path=TRAIN_PATH, val_path=VAL_PATH, output_dir=OUTPUT_DIR)
        fine_tuning_engine.setup()
        fine_tuning_engine.run(**hp)
        print("✅ Fine-tuning complete.")
    else:
        try:
            fine_tuned_model = SentenceTransformer(OUTPUT_DIR)
            print(f"✅ Fine-tuned model loaded from '{OUTPUT_DIR}'.")
        except Exception as e:
            print(f"❌ Failed to load fine-tuned model: {e}")
            print("Fallback: Loading base model instead.")
            fine_tuned_model = SentenceTransformer("BAAI/bge-small-en")
    
    print(f"Model is ready and loaded.")       
        
    # Embedding chunks
    print("📐 Embedding chunks...")

    # Fine-tuned model as the embedder
    embedder = TextEmbedder(fine_tuned_model=fine_tuned_model, use_huggingface=True)
    embeddings_path = "embeddings.pkl"

    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        print("✅ Loaded existing embeddings.")
    else:
        embeddings = embedder.encode(chunk_texts, batch_size=32)
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)
        print("✅ New embeddings saved.")

    # Default Model as the embedder
    # embedder = TextEmbedder()
    # embeddings = embedder.encode(chunk_texts)

    # Vector store
    vector_store = FaissVectorStore(dim=embeddings.shape[1])
    if os.path.exists("faiss_index.bin"):
        vector_store.load()
    else:
        vector_store.add(embeddings, chunk_ids, chunk_texts, chunk_titles)
        vector_store.save()

    # Initialize the RAG pipeline
    retriever = Retriever(embedder, vector_store.index, chunk_texts, chunk_ids,chunk_titles)
    rag = RAGEngine(model="llama3", temperature=0.5)
    evaluator = RAGEvaluator()

    print("\n🧠 RAG system ready. Ask me anything!\n")

    while True:
        query = input("❓ Your question (or 'exit'): ")
        if query.lower().strip() == "exit":
            break
            
        # Retrieve and build prompt
        top_chunks = retriever.retrieve(query, top_k=3)
        prompt = rag.build_prompt(top_chunks, query)
        response = rag.query(prompt)

        print("\n🦙 LLaMA (Markdown output):\n")
        print(response.strip())
        print("\n" + "-" * 60 + "\n")

        # Evaluation 
        feedback = input("⭐️ Do you have a reference answer to evaluate? (y/n): ")
        if feedback.lower().strip() == "y":
            reference = input("🔖 Paste reference answer:\n")
            scores = evaluator.evaluate_batch([response.strip()], [reference])

            print("\n📊 Evaluation Results:")
            for metric, value in scores.items():
                if isinstance(value, list):
                    value = value[0]
                print(f"- {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
