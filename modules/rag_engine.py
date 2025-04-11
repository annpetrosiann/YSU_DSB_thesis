import subprocess
from typing import List, Dict

class RAGEngine:
    def __init__(self, model: str = "llama3",temperature: float = 0.7):
        self.model = model
        self.temperature = temperature


    def build_prompt(self, context_chunks: List[Dict[str, str]], question: str) -> str:
        """
        Build a structured prompt using the given context and user question.

        Args:
            context_chunks (List[Dict[str, str]]): List of dictionaries containing 'text' and optional 'title'.
            question (str): The user's question.

        Returns:
            str: A well-formatted prompt string.
        """
        context_lines = [
            f"### {chunk['title']}\n{chunk['text']}" if chunk.get('title') else f"- {chunk['text']}"
            for chunk in context_chunks
        ]

        joined_context = "\n\n".join(context_lines)

        prompt = (
            "You are an expert assistant. Use the following context to answer the question.\n"
            "If the answer is not found in your context, say: "
            "\"I'm sorry. I don't have enough information in my context to answer your question. "
            "But feel free to ask me something else — I’ll do my best to help! :)\"\n\n"
            "## Context:\n"
            f"{joined_context}\n\n"
            "## Question:\n"
            f"{question}\n\n"
            "## Answer (in Markdown):"
        )
        return prompt

    def query(self, prompt: str) -> str:
        """
        Run the model using a subprocess with the given prompt.

        Args:
            prompt (str): The constructed prompt to send to the model.

        Returns:
            str: The model's response or an error message.
        """
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout.decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode("utf-8").strip()
            return f"Error running model: {error_msg}"
