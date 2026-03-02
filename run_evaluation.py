import json
from rag_pipeline import RAGPipeline

rag = RAGPipeline("10-Q4-2024-As-Filed.pdf", "tsla-20231231-gen.pdf")

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
    out = rag.answer_question(q["question"])
    results.append({
        "question_id": q["question_id"],
        "answer": out["answer"],
        "sources": out["sources"]
    })
    print(f"Q{q['question_id']}: {out['answer'][:120]}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to results.json")
print("\n=== FINAL JSON OUTPUT ===")
print(json.dumps(results, indent=2))
