from modules.eval_pipeline import RAGEvaluator

queries = [
    "What is the role of mitochondria?",
    "Who wrote Hamlet?",
]
generated = [
    "Mitochondria produce energy in the form of ATP through cellular respiration.",
    "William Shakespeare wrote Hamlet.",
]
references = [
    "The mitochondria are responsible for producing ATP via cellular respiration.",
    "Hamlet was written by William Shakespeare.",
]

evaluator = RAGEvaluator()
results = evaluator.evaluate_batch(generated, references)
print("\n📊 Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")