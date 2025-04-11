from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import warnings
from transformers import logging as transformers_logging
import numpy as np

# Suppress unnecessary warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning, module='nltk.translate.bleu_score')

class RAGEvaluator:
    def __init__(self):
        """Initialize evaluation metrics"""
        self.sim_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.bertscore = evaluate.load("bertscore")
        self.smoother = SmoothingFunction()

    def cosine_similarity(self, generated: str, reference: str) -> float:
        """Enhanced semantic similarity with error handling"""
        try:
            emb1 = self.sim_model.encode(generated, convert_to_tensor=True)
            emb2 = self.sim_model.encode(reference, convert_to_tensor=True)
            return float(util.pytorch_cos_sim(emb1, emb2)[0][0])
        except Exception as e:
            print(f"⚠️ Cosine similarity error: {str(e)}")
            return 0.0

    def evaluate_batch(self, generations: list, references: list) -> dict:
        """Comprehensive evaluation with improved metrics"""
        print("🔍 Running full evaluation...")

        # Initialize scores with numpy arrays for better numerical handling
        scores = {
            "cosine_sim": np.zeros(len(generations)),
            "bleu": np.zeros(len(generations)),
            "rougeL": np.zeros(len(generations)),
            "bertscore_f1": np.zeros(len(generations)),
        }

        try:
            # ROUGE (handles list inputs natively)
            rouge_results = self.rouge.compute(
                predictions=generations,
                references=references,
                use_stemmer=True
            )
            scores["rougeL"] = [rouge_results["rougeL"]]

            # BERTScore with optimized parameters
            bertscore_results = self.bertscore.compute(
                predictions=generations,
                references=references,
                lang="en",
                model_type="roberta-large",
                device="cpu"  # Change to "cuda" if using GPU
            )
            scores["bertscore_f1"] = bertscore_results["f1"]

            # Individual sample processing
            for i, (gen, ref) in enumerate(zip(generations, references)):
                # Cosine similarity
                scores["cosine_sim"][i] = self.cosine_similarity(gen, ref)

                # BLEU with smoothing
                scores["bleu"][i] = sentence_bleu(
                    [ref.split()],
                    gen.split(),
                    smoothing_function = self.smoother.method1,
                    weights=(0.25, 0.25, 0.25, 0.25)  # 4-gram balanced
                )


        except Exception as e:
            print(f"⚠️ Evaluation error: {str(e)}")

        return {
            "cosine_sim": float(np.mean(scores["cosine_sim"])),
            "bleu": float(np.mean(scores["bleu"])),
            "rougeL": float(np.mean(scores["rougeL"])),
            "bertscore": float(np.mean(scores["bertscore_f1"]))
        }