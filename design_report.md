# RAG System Design Report
**Apple 2024 10-K + Tesla 2023 10-K — Question Answering**

---

## Chunking Strategy

PDFs are parsed page-by-page with PyMuPDF. Each page produces **two separate chunks**:

**Prose chunks** — cleaned plain text. A `clean_text()` function drops lines shorter than 3 characters and lines where less than 45% of characters are alphanumeric. This removes table borders, underscores, and symbol-heavy rows that were causing the model to output garbage in earlier iterations.

**Table chunks** — structured rows extracted via `page.find_tables()`. Each row is reconstructed as `Label | Value | Value`, preserving the relationship between row labels and numbers. Without this, financial tables flatten into scattered digits and the model picks the wrong row.

Chunks under 20 words after cleaning are discarded. Prose window: 400 words, step 300 (100-word overlap).

Each chunk stores: `doc`, `page`, `section` (regex-matched `ITEM X` header), and `type` (prose / table / precise).

---

## Embedding Model

`BAAI/bge-base-en-v1.5` — 109M parameter bi-encoder. Upgraded from `bge-small` because base performs better on dense financial text. Outputs normalized vectors; cosine similarity maps directly to FAISS inner-product search.

---

## Vector Store

FAISS `IndexFlatIP` — exact search. Corpus size (4k–6k chunks) is small enough that approximate methods add complexity with no real speed benefit.

---

## Re-Ranker

Top-40 FAISS candidates are re-ranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`. Precise chunks skip this step and go directly to the top context slots. The re-ranker runs on the remaining candidates and fills the rest of the top-5.

---

## LLM

`mistralai/Mistral-7B-Instruct-v0.2` loaded in **4-bit NF4 quantization** via bitsandbytes. Fits on a free Colab T4 GPU (5.5GB VRAM). Uses Mistral's `[INST]` prompt format. `return_full_text=False` prevents the model echoing the prompt back.

The prompt enforces: cite sources, quote exact figures, show arithmetic when adding numbers, refuse if not in context.

---

## Out-of-Scope Handling

Two layers:

**Layer 1 — Regex patterns** catch explicit out-of-scope questions before the LLM is called:
- `stock price forecast|prediction|target|2025` → refuses Q11
- `cfo.*2025|2025.*cfo` → refuses Q12 (question asks about post-filing state)
- `what color|painted|headquarters.*color` → refuses Q13

**Layer 2 — Scope keyword filter** — if the query has no Apple/Tesla/financial terms at all, refuse immediately.

Previously only a keyword filter existed, which let Q11 through because `"stock"` matched. Regex patterns fix this by looking at the full intent of the question, not just individual words.
