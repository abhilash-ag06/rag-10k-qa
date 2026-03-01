import re
import numpy as np
import fitz
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


SCOPE_KEYWORDS = [
    "apple", "tesla", "revenue", "10-k", "annual report", "fiscal", "shares",
    "stock", "debt", "filing", "sec", "elon musk", "cfo", "model s", "model 3",
    "cybertruck", "iphone", "services", "risk", "lease", "compensation",
    "financial", "quarter", "outstanding", "term debt", "signed", "unresolved",
    "automotive", "vehicles", "pass-through"
]


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


class RAGPipeline:
    def __init__(self, apple_pdf, tesla_pdf):
        print("Loading embedder...")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        print("Loading reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("Parsing PDFs...")
        apple = parse_pdf(apple_pdf, "Apple 10-K")
        tesla = parse_pdf(tesla_pdf, "Tesla 10-K")
        self.chunks = apple + tesla
        print(f"  {len(apple)} apple chunks, {len(tesla)} tesla chunks")

        print("Building index...")
        texts = [c["text"] for c in self.chunks]
        embs = self.embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        embs = np.array(embs).astype("float32")
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

        print("Loading LLM...")
        mid = "microsoft/Phi-3-mini-4k-instruct"
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16,
                                                   device_map="auto", trust_remote_code=True)
        self.llm = pipeline("text-generation", model=mod, tokenizer=tok,
                            max_new_tokens=512, do_sample=False, temperature=1.0)
        print("Ready.")

    def _retrieve(self, query, top_k=20, top_n=5):
        qvec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, idxs = self.index.search(qvec, top_k)
        candidates = [self.chunks[i] for i in idxs[0]]
        scores = self.reranker.predict([(query, c["text"]) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]

    def _prompt(self, query, chunks):
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

    def answer_question(self, query: str) -> dict:
        """
        Args:
            query: question about Apple or Tesla 10-K
        Returns:
            dict with 'answer' and 'sources'
        """
        if not any(kw in query.lower() for kw in SCOPE_KEYWORDS):
            return {
                "answer": "This question cannot be answered based on the provided documents.",
                "sources": []
            }

        chunks = self._retrieve(query)
        raw = self.llm(self._prompt(query, chunks))[0]["generated_text"]

        answer = raw.split("Answer:")[-1].strip() if "Answer:" in raw else raw.strip()
        answer = answer.split("\n\nQuestion:")[0].strip()

        return {
            "answer": answer,
            "sources": [f'{c["doc"]}, {c["section"]}, p. {c["page"]}' for c in chunks]
        }
