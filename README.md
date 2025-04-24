# 🧠 Armenian Financial Document RAG System

This repository contains the full pipeline for my Master's thesis at **Yerevan State University (YSU)**, developed as part of the **Data Science for Business master's program**. The goal of this project is to build an **end-to-end Retrieval-Augmented Generation (RAG)** system using **semantic search**, **LLMs**, and **fine-tuned embeddings** to answer questions on **Armenian banks’ financial PDF documents**.

---

## 📦 Features

- 🔎 PDF-to-JSON conversion using `pdfplumber`
- ✂️ Text chunking with `spaCy` or `nltk`
- 🧠 Embedding with SentenceTransformers (with optional fine-tuning)
- 📚 Semantic retrieval using FAISS with keyword boosting
- 🤖 Query response generation via LLaMA 3 (via Ollama)
- 📊 Evaluation metrics: BLEU, ROUGE, BERTScore, cosine similarity
- 📈 Fine-tuning support with MRR and Recall@5 evaluation

---

## ⚙️ Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally
- A model available in Ollama (e.g., `llama3`, `mistral`, etc.)
- (Optional) GPU support for faster embedding and inference

---
## 📂 Project Structure
```
YSU-DSB-thesis
├── data/
│   └── json files
│
├── data_process/          # Scripts for preprocessing PDF files
│   ├── pdf2json.py              
│   └──  comb_json.py   
│         
├── modules/               # Core pipeline modules
│   ├── chunker.py            
│   ├── embedder.py          
│   ├── retriever.py          
│   ├── rag_engine.py       
│   ├── eval_pipeline.py     
│   └── fine_tuning_engine.py
│
├── testing/               # Tests for key modules
│   ├── test_chunks.py             
│   ├── test_embedder.py          
│   ├── test_evaluation.py          
│   └──  test_vector_database.py   
│
├── main.py                # Main script to run the full pipeline       
├── config.json                    
├── .gitignore
├── requirements.txt       
├── setup.sh
└── README.md
```


## 🧰 Installation

1. **Clone the repository:**

```bash
  git clone https://github.com/annpetrosiann/YSU_DSB_thesis.git
```
```bash  
  cd YSU_DSB_thesis
```

2. **Create a virtual environment (optional)**:
  ```bash
    python3 -m venv {virtual_environment_name}
  ```
**Activate the virtual environment**:

    # On Windows

      python3 -m venv {virtual_environment_name}

    # On macOS or Linux
  
      source {virtual_environment_name}/bin/activate
     
3. **Install Dependencies**

You have two options:

- Manual install: using the provided requirements.txt
```bash
    pip install -r requirements.txt --upgrade 
 ```
- Automatic setup: run the setup.sh script which installs everything for you

```bash
    bash setup.sh
```
4. **Upgrade sentence-transformers and llama-index** (only if you didn’t run setup.sh)

```bash
    pip install --upgrade sentence-transformers llama-index
```

## 🗃️ Data
The data is sourced from the financial statements published on the official websites of Armenian banks. According to the Central Bank of Armenia, there are currently 18 licensed banks operating in the country. This project includes data from all 18 banks except VTB Bank, as its financial reports were not available in English.

The collected reports include:
- 📄 Annual reports
- 🧾 Audit reports
- 📘 Reports with consolidated notes
- 📊 Interim quarterly reports, which feature:
- Balance sheets
- Income statements
- Statements of changes in equity
- Cash flow statements
- Normative indicators
- Other statutory reports

While every effort was made to collect all available English-language reports, some documents were either missing or not publicly accessible for certain banks or years. The coverage period varies by bank, depending on when they began publishing reports in English. The latest available data across all banks is from 2024.

## 🛠️ Data Processing
All PDF files were converted into structured JSON format using the scripts in the `data_process` directory:
```bash
  # Step 1: Convert PDFs to JSON
  python3 data_process/pdf2json.py

  # Step 2: Combine JSONs into a single structured file
  python3 data_process/comb_json.py
```
The final dataset used for training is `arm_banks.json`, which is not included in this repository due to size and privacy concerns. Feel free to check out sample JSON files available in the `data` directory.

## 🧩 Modules + Config + Tests

Inside the `modules` directory, you'll find the core components of the pipeline:

| Script                  | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| `chunker.py`            | Splits long documents into manageable text chunks using `spaCy` or `nltk` |
| `embedder.py`           | Converts text chunks into embeddings using `SentenceTransformers` or HuggingFace |
| `retriever.py`          | Retrieves top-k relevant chunks using FAISS and keyword matching        |
| `rag_engine.py`         | Constructs prompts and queries LLaMA via `Ollama`                       |
| `eval_pipeline.py`      | Evaluates model output with metrics like BLEU, ROUGE-L, BERTScore, and cosine similarity |
| `fine_tuning_engine.py` | Fine-tunes embedding models and evaluates best configs using MRR and Recall@5 |

You can run any of these scripts directly:
```bash
    python3 modules/{script_name}.py
```
For example, running the fine-tuning engine:

```bash
    python3 modules/fine_tuning_engine.py
```
This will perform a grid search and print the best hyperparameter configuration along with MRR and Recall@5 scores. The best configuration is saved in `config.json`.

🧪 Unit tests for most modules can be found in the `testing` directory. These help validate that each script is functioning as expected.

## 🚀 Full Pipeline Execution

The `main.py` script orchestrates the entire RAG pipeline:
- Loads and chunks the financial text data
- Fine-tunes or loads a pre-trained embedding model
- Generates embeddings and stores them in a FAISS vector index
- Launches an interactive Q&A interface powered by LLaMA via Ollama

```bash
    python3 main.py
```
Once running, you can ask questions about the Armenian banking system, and the RAG system will generate responses based on the financial reports it has indexed.

### 💬 Example Interaction

```text
❓ Your question (or 'exit'): What was Ameriabank’s total equity in 2022?

🦙 LLaMA (Markdown output):

In 2022, Ameriabank's total equity amounted to  'X' AMD, as reported in their annual financial statements.

----------------------------------------------------------------------------------------------------------
⭐️ Do you have a reference answer to evaluate? (y/n): 
```
If you provide a reference answer, the system will compute evaluation metrics:

- **BLEU**: Measures n-gram overlap between the generated and reference text.
- **ROUGE-L**: Evaluates the longest common subsequence (LCS) between texts to capture fluency and informativeness.
- **BERTScore**: Uses contextual embeddings from BERT to evaluate semantic similarity at a token level.
- **Cosine Similarity**:  Measures overall vector similarity between the generated and reference embeddings.

These metrics help you assess how well the system’s output matches a known correct answer — both syntactically and semantically.

## 📌 Credits

Student: Anahit Petrosyan

Supervisor: Yenok Hakobyan

Master’s Thesis – Data Science for Business, Yerevan State University, 2025.

## 📭 Contact
For questions, feedback, or collaboration opportunities, feel free to connect via [Linkedin](https://www.linkedin.com/in/anahit-petrosian-2647a821b/).

## 📄 License
This project is **not currently licensed.** All rights are reserved by the author.
If you wish to use, reproduce, or reference any part of this work, please contact the author for permission.

