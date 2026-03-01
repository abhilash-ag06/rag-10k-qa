# RAG pipeline for Apple 2024 10-K and Tesla 2023 10-K
# Retrieves relevant chunks and generates answers using Phi-3-mini


# ---- CELL 1 ----
# install deps
get_ipython().system('pip install pymupdf sentence-transformers faiss-cpu transformers accelerate torch --quiet')


# ---- CELL 2 ----
# clone repo
get_ipython().system('git clone https://github.com/abhilash-ag06/rag-10k-qa.git')
get_ipython().run_line_magic('cd', 'rag-10k-qa')


# ---- CELL 3 ----
import os
import re
import json
import numpy as np
import fitz
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# ---- CELL 4 ----
# upload PDFs manually via Colab sidebar or run this cell to move them
# after uploading via Files panel on the left, they land in /content/
# this cell copies them into the repo folder if needed

import shutil

for fname in ["10-Q4-2024-As-Filed.pdf", "tsla-20231231-gen.pdf"]:
    src = f"/content/{fname}"
    dst = f"/content/rag-10k-qa/{fname}"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Moved {fname}")
    elif os.path.exists(dst):
        print(f"{fname} already in place")
    else:
        print(f"WARNING: {fname} not found — upload it via the Files panel")


# ---- CELL 5 ----
def detect_section(text):
    m = re.search(r'(Item\s+\d+[A-Z]?\.?[^\n]{0,60})', text, re.IGNORECASE)
    return m.group(1).strip() if m else "General"


def parse_pdf(path, label):
    doc = fitz.open(path)
    chunks = []

    for pg in range(len(doc)):
        text = doc[pg].get_text()
        if not text.strip():
            continue

        words = text.split()
        section = detect_section(text)

        if len(words) <= 500:
            chunks.append({"text": text.strip(), "doc": label, "page": pg + 1, "section": section})
        else:
            for i in range(0, len(words), 400):
                chunk = " ".join(words[i:i + 500])
                chunks.append({"text": chunk, "doc": label, "page": pg + 1, "section": section})

    doc.close()
    return chunks


apple_chunks = parse_pdf("10-Q4-2024-As-Filed.pdf", "Apple 10-K")
tesla_chunks = parse_pdf("tsla-20231231-gen.pdf", "Tesla 10-K")
all_chunks = apple_chunks + tesla_chunks

print(f"Apple: {len(apple_chunks)} | Tesla: {len(tesla_chunks)} | Total: {len(all_chunks)}")


# ---- CELL 6 ----
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts = [c["text"] for c in all_chunks]
embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"Index ready: {index.ntotal} vectors")


# ---- CELL 7 ----
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve(query, top_k=20, top_n=5):
    qvec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = index.search(qvec, top_k)
    candidates = [all_chunks[i] for i in idxs[0]]

    scores = reranker.predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]


# ---- CELL 8 ----
model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer,
               max_new_tokens=512, do_sample=False, temperature=1.0)

print("Model loaded")


# ---- CELL 9 ----
SCOPE_KEYWORDS = [
    "apple", "tesla", "revenue", "10-k", "annual report", "fiscal", "shares",
    "stock", "debt", "filing", "sec", "elon musk", "cfo", "model s", "model 3",
    "cybertruck", "iphone", "services", "risk", "lease", "compensation",
    "financial", "quarter", "outstanding", "term debt", "signed", "unresolved",
    "automotive", "vehicles", "pass-through"
]


def in_scope(query):
    return any(kw in query.lower() for kw in SCOPE_KEYWORDS)


def build_prompt(query, chunks):
    ctx = "\n\n".join(
        f'Source {i+1} [{c["doc"]}, {c["section"]}, p. {c["page"]}]:\n{c["text"]}'
        for i, c in enumerate(chunks)
    )
    return f"""You are a financial analyst assistant. Answer only using the context below.

Rules:
- Cite sources as [Apple 10-K, Item 8, p. 28]
- If not in context, say exactly: Not specified in the document.
- No outside knowledge.

Context:
{ctx}

Question: {query}
Answer:"""


def answer_question(query: str) -> dict:
    """
    Args:
        query: question about Apple or Tesla 10-K
    Returns:
        dict with 'answer' and 'sources'
    """
    if not in_scope(query):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)
    raw = llm(prompt)[0]["generated_text"]

    answer = raw.split("Answer:")[-1].strip() if "Answer:" in raw else raw.strip()
    answer = answer.split("\n\nQuestion:")[0].strip()

    sources = [f'{c["doc"]}, {c["section"]}, p. {c["page"]}' for c in chunks]
    return {"answer": answer, "sources": sources}


print("answer_question() ready")


# ---- CELL 10 ----
questions = [
    {"question_id": 1,  "question": "What was Apple's total revenue for the fiscal year ended September 28, 2024?"},
    {"question_id": 2,  "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
    {"question_id": 3,  "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
    {"question_id": 4,  "question": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?"},
    {"question_id": 5,  "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
    {"question_id": 6,  "question": "What was Tesla's total revenue for the year ended December 31, 2023?"},
    {"question_id": 7,  "question": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
    {"question_id": 8,  "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
    {"question_id": 9,  "question": "What types of vehicles does Tesla currently produce and deliver?"},
    {"question_id": 10, "question": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?"},
    {"question_id": 11, "question": "What is Tesla's stock price forecast for 2025?"},
    {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
    {"question_id": 13, "question": "What color is Tesla's headquarters painted?"},
]

results = []
for q in questions:
    print(f"Running Q{q['question_id']}...")
    out = answer_question(q["question"])
    results.append({"question_id": q["question_id"], "answer": out["answer"], "sources": out["sources"]})

print(json.dumps(results, indent=2))
