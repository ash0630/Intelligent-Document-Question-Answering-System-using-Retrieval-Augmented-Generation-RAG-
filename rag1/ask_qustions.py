import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# =========================
# Gemini Client
# =========================
client = genai.Client(api_key="AIzaSyAzxzQdZzifdN0Rb6XoE0tC68mDewPPjMo")

# =========================
# Pick a Supported Text Model Automatically
# =========================
def get_text_model():
    models = client.models.list()
    for m in models:
        if "generateContent" in m.supported_actions:
            return m.name
    raise RuntimeError("No supported text generation model found")

LLM_MODEL = get_text_model()
print(f" Using Gemini model: {LLM_MODEL}")

# =========================
# Local Embedding Model
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Load FAISS Index & Metadata
# =========================
print(" Loading vector database...")

index = faiss.read_index("vectors.index")

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
metadata = data["metadata"]

print(f"Loaded {len(chunks)} chunks")

# =========================
# Helper Functions
# =========================
def embed_query(text):
    return embedding_model.encode(text).astype("float32")

def retrieve_context(query, top_k=5):
    query_vector = embed_query(query).reshape(1, -1)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "text": chunks[idx],
            "page": metadata[idx]["estimated_page"]
        })

    return results

def generate_answer(question, context_chunks):
    context_text = "\n\n".join(
        f"(Page {c['page']}): {c['text']}"
        for c in context_chunks
    )

    prompt = f"""
Answer the question strictly using the context below.

Context:
{context_text}

Question:
{question}

Rules:
- Use only the provided context
- If the answer is not present, say:
  "The document does not contain this information."
- Mention page numbers
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text.strip()

# =========================
# Interactive Chat
# =========================
def chat():
    print("\n Ask questions about your PDF")
    print("Type 'exit' to quit\n")

    while True:
        query = input("❓ Question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        context = retrieve_context(query, top_k=5)
        answer = generate_answer(query, context)

        print("\n Answer:\n")
        print(answer)
        print("\n" + "-" * 80 + "\n")

# =========================
# Run
# =========================
if __name__ == "__main__":
    chat()


