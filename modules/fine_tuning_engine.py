import json
from pathlib import Path
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset


class FineTuningEngine:
    def __init__(self, train_path: str, val_path: str, model_id: str = "BAAI/bge-small-en", output_dir: str = "fine_tuned_model"):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.finetune_engine = None
        self.embed_model = None

    def _load_dataset(self, path: Path, split: str = "train"):
        print(f"Loading dataset '{split}' from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_pairs = data[split]

        corpus = {f"doc{i}": pair["answer"] for i, pair in enumerate(raw_pairs)}
        queries = {f"q{i}": pair["question"] for i, pair in enumerate(raw_pairs)}
        relevant_docs = {f"q{i}": [f"doc{i}"] for i in range(len(raw_pairs))}

        dataset = EmbeddingQAFinetuneDataset(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs
        )
        print(f"Loaded {len(raw_pairs)} items.")
        return dataset

    def setup(self):
        print("Setting up fine-tuning engine...")
        train_dataset = self._load_dataset(self.train_path, split="train")
        val_dataset = self._load_dataset(self.val_path, split="test")


        self.finetune_engine = SentenceTransformersFinetuneEngine(
            dataset=train_dataset,
            val_dataset=val_dataset,
            model_id=self.model_id,
            model_output_path=str(self.output_dir)
        )
        print("Engine setup complete.")

    def run(self):
        if self.finetune_engine is None:
            raise ValueError("Engine not set up. Call setup() first.")
        print("Starting fine-tuning...")
        self.finetune_engine.finetune()
        print("Fine-tuning complete.")


    def get_model(self):
        if self.finetune_engine is None:
            raise ValueError("Engine not set up. Call setup() first.")
        print("Retrieving fine-tuned model...")
        self.embed_model = self.finetune_engine.get_finetuned_model()
        return self.embed_model