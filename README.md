# RAG 10-K Q&A — Apple & Tesla

Answers financial questions from Apple's 2024 10-K and Tesla's 2023 10-K using RAG (retrieval-augmented generation).

**Live Notebook:** https://colab.research.google.com/drive/13KlZFKq0YgXfwGaLvNXGpPtJZ4ytSKmJ?usp=sharing

---

## Repo Structure

```
rag-10k-qa/
├── rag_pipeline.py          # core RAG logic — parse, embed, retrieve, generate
├── run_evaluation.py        # runs all 13 test questions
├── colab_notebook_code.py   # full notebook code split into cells
├── design_report.md         # chunking + model choice explanation
├── requirements.txt
├── 10-Q4-2024-As-Filed.pdf  # Apple 2024 10-K  (add manually — too large for git)
└── tsla-20231231-gen.pdf    # Tesla 2023 10-K  (add manually — too large for git)
```

---

## How It Works

- PDFs parsed with PyMuPDF, chunked at 500 words with 100-word overlap
- Embeddings via `BAAI/bge-small-en-v1.5`, stored in FAISS flat index
- Top-20 retrieved, re-ranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-5 passed to LLM
- `Phi-3-mini-4k-instruct` generates answers with forced source citations
- Out-of-scope questions caught by keyword filter before hitting the LLM

---

## Local Setup

```bash
git clone https://github.com/abhilash-ag06/rag-10k-qa.git
cd rag-10k-qa
pip install -r requirements.txt

# place both PDFs in the repo root, then:
python run_evaluation.py
```

---

## Single Question

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline("10-Q4-2024-As-Filed.pdf", "tsla-20231231-gen.pdf")
print(rag.answer_question("What was Apple's total revenue for fiscal year 2024?"))
```
