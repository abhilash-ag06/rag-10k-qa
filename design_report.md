# RAG System Design Report
**Apple 2024 10-K + Tesla 2023 10-K — Question Answering**

---

## Chunking

PDFs are parsed page-by-page with PyMuPDF. Each page goes through a sliding window: 500 words per chunk, 100-word overlap (step 400). Pages under 500 words stay as one chunk. SEC filings are dense — smaller chunks lose context, larger ones dilute retrieval signal. Overlap stops answers from splitting across chunk boundaries.

Metadata stored per chunk: `doc` (Apple 10-K / Tesla 10-K), `page` (1-indexed), `section` (regex match on `Item X` headers). This makes every citation traceable to a document, section, and page.

---

## Embedding Model

`BAAI/bge-small-en-v1.5` — 33M parameter bi-encoder. Fast on T4 GPU, strong on retrieval benchmarks, outputs normalized vectors so cosine similarity maps directly to FAISS inner-product search.

---

## Vector Store

FAISS `IndexFlatIP` — exact search over inner products. Corpus size (~4k chunks) is small enough that approximate methods add complexity with no real speed gain.

---

## Re-Ranker

Top-20 FAISS results get re-ranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`. Cross-encoders jointly encode query + passage together, catching relevance that bi-encoders miss. Cost is acceptable at 20 pairs; it would not be at 5000.

---

## LLM

`microsoft/Phi-3-mini-4k-instruct` (3.8B). Fits in float16 on a free Colab T4. Instruction-following is strong for its size, fully open-weight, 4k context handles 5 chunks + prompt comfortably.

---

## Out-of-Scope Handling

Two layers:

1. **Keyword filter** — if query has no Apple/Tesla/financial terms, refuse immediately without hitting the LLM.
2. **Prompt constraint** — prompt tells the model to reply "Not specified in the document." if the answer isn't in the retrieved context.

Questions 11 (stock forecast), 12 (CFO as of 2025), 13 (HQ color) all hit one of these two layers.
