import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
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
        self.val_dataset = None

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
        self.val_dataset = self._load_dataset(self.val_path, split="test")

        self.finetune_engine = SentenceTransformersFinetuneEngine(
            dataset=train_dataset,
            val_dataset=self.val_dataset,
            model_id=self.model_id,
            model_output_path=str(self.output_dir)
        )
        print("Engine setup complete.")

    def run(self,
            num_epochs=5,
            learning_rate=2e-5,
            batch_size=32,
            max_seq_length=256,
            warmup_ratio=0.1):
        if self.finetune_engine is None:
            raise ValueError("Engine not set up. Call setup() first.")

        print("Starting fine-tuning with parameters:")
        print(f"Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}, Max Seq: {max_seq_length}, Warmup: {warmup_ratio}")

        self.finetune_engine.finetune(
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            warmup_ratio=warmup_ratio
        )
        print("Fine-tuning complete.")

    def get_model(self):
        if self.finetune_engine is None:
            raise ValueError("Engine not set up. Call setup() first.")
        print("Retrieving fine-tuned model...")
        self.embed_model = self.finetune_engine.get_finetuned_model()
        return self.embed_model

    def evaluate_mrr_recall(self, dataset, embed_model, k=5):
        print("Evaluating MRR and Recall@k...")
        query_ids = list(dataset.queries.keys())
        doc_ids = list(dataset.corpus.keys())

        query_texts = [dataset.queries[qid] for qid in query_ids]
        doc_texts = [dataset.corpus[did] for did in doc_ids]

        query_embeddings = [embed_model.get_text_embedding(text) for text in query_texts]
        doc_embeddings = [embed_model.get_text_embedding(text) for text in doc_texts]

        query_embeddings = np.array(query_embeddings)
        doc_embeddings = np.array(doc_embeddings)

        sim_matrix = cosine_similarity(query_embeddings, doc_embeddings)

        mrr_total = 0
        recall_total = 0

        for i, qid in enumerate(query_ids):
            relevant = dataset.relevant_docs[qid]
            relevant_idx = [doc_ids.index(doc_id) for doc_id in relevant]

            ranked_indices = np.argsort(sim_matrix[i])[::-1]

            # MRR
            rr = 0
            for rank, idx in enumerate(ranked_indices, start=1):
                if idx in relevant_idx:
                    rr = 1 / rank
                    break
            mrr_total += rr

            # Recall@k
            top_k = ranked_indices[:k]
            hit = any(idx in relevant_idx for idx in top_k)
            recall_total += int(hit)

        mrr = mrr_total / len(query_ids)
        recall = recall_total / len(query_ids)

        print(f"MRR: {mrr:.4f}, Recall@{k}: {recall:.4f}")
        return mrr, recall

    def grid_search(self):
        learning_rates = [2e-5, 5e-5]
        batch_sizes = [16, 32]
        num_epochs_list = [3, 5]
        warmup_ratios = [0.1, 0.2]
        max_seq_lengths = [128, 256]

        best_score = float("-inf")
        best_params = None

        for lr in learning_rates:
            for bs in batch_sizes:
                for ep in num_epochs_list:
                    for wr in warmup_ratios:
                        for msl in max_seq_lengths:
                            print(f"\nTesting config: lr={lr}, batch={bs}, epochs={ep}, warmup={wr}, max_seq_length={msl}")
                            self.setup()
                            self.run(
                                num_epochs=ep,
                                learning_rate=lr,
                                batch_size=bs,
                                warmup_ratio=wr,
                                max_seq_length=msl
                            )
                            model = self.get_model()
                            mrr, recall = self.evaluate_mrr_recall(self.val_dataset, model, k=5)

                            score = mrr + recall
                            print(f"Eval Score (MRR + Recall@5): {score:.4f}")

                            if score > best_score:
                                best_score = score
                                best_params = (lr, bs, ep, wr, msl)

        print("\n=== Best Configuration ===")
        print(f"Learning Rate: {best_params[0]}")
        print(f"Batch Size: {best_params[1]}")
        print(f"Num Epochs: {best_params[2]}")
        print(f"Warmup Ratio: {best_params[3]}")
        print(f"Max Seq Length: {best_params[4]}")
        print(f"Best Score (MRR + Recall@5): {best_score:.4f}")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    TRAIN_PATH = config["train_path"]
    VAL_PATH = config["val_path"]
    engine = FineTuningEngine(TRAIN_PATH, VAL_PATH)
    engine.grid_search()