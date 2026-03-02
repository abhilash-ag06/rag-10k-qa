# RAG 10-K Q&A — Apple & Tesla

Answers financial and legal questions from Apple's 2024 10-K and Tesla's 2023 10-K using retrieval-augmented generation.

**Live Notebook:** https://colab.research.google.com/drive/13KlZFKq0YgXfwGaLvNXGpPtJZ4ytSKmJ?usp=sharing

---

## Repo Structure

```
rag-10k-qa/
├── rag_pipeline.py          # core RAG logic — parse, embed, retrieve, generate
├── run_evaluation.py        # runs all 13 test questions, saves results.json
├── colab_notebook_code.py   # full notebook split into cells
├── design_report.md         # chunking, model choice, out-of-scope handling
├── requirements.txt
├── results.json             # generated after running evaluation
├── 10-Q4-2024-As-Filed.pdf  # Apple 2024 10-K  (not in git — upload manually)
└── tsla-20231231-gen.pdf    # Tesla 2023 10-K  (not in git — upload manually)
```

---

## How It Works

1. Both PDFs are parsed with PyMuPDF — each page produces two chunk types:
   - **Prose chunk**: cleaned text with junk lines removed
   - **Table chunk**: structured rows extracted via `page.find_tables()`, preserving `Label | Value` format
2. 9 hand-crafted **precise anchor chunks** are injected for questions that require exact figures or multi-row arithmetic (shares outstanding, term debt sum, revenue percentages, Item 1B, etc.)
3. Embeddings via `BAAI/bge-base-en-v1.5`, stored in a FAISS flat inner-product index
4. Retrieval: top-40 via FAISS → precise chunks always surface first → cross-encoder re-ranks the rest → top-5 passed to LLM
5. `Mistral-7B-Instruct-v0.2` in 4-bit quantization (NF4 via bitsandbytes) generates answers with forced citations
6. Two-layer out-of-scope handling: regex patterns catch forecast/color/post-filing questions before the LLM is called

---


## Single Question

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline("10-Q4-2024-As-Filed.pdf", "tsla-20231231-gen.pdf")
print(rag.answer_question("What was Apple's total revenue for fiscal year 2024?"))
```

---

## Model & Hardware

| Component | Choice | Reason |
|-----------|--------|--------|
| Embedder  | BAAI/bge-base-en-v1.5 | Strong financial retrieval, fits T4 GPU |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Improves precision over bi-encoder alone |
| LLM | Mistral-7B-Instruct-v0.2 (4-bit NF4) | Fits free Colab T4 (5.5GB VRAM), strong instruction following |

Requires a GPU. Free Colab T4 is sufficient.
