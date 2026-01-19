import faiss
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

def get_embedding(text):
    return model.encode(text).astype("float32")


def pdf_to_vectors(pdf_path):
    print(f"📄 Reading PDF: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        page_texts = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            page_texts.append({
                "text": text,
                "page_number": i + 1
            })

        full_text = "".join(p["text"] for p in page_texts)

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(full_text):,}")

    chunk_size = 1200
    overlap = 200

    chunks = []
    metadata = []

    for i in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[i:i + chunk_size]

        if len(chunk.strip()) < 100:
            continue

        chunks.append(chunk)

        estimated_page = min(
            (i // (len(full_text) // total_pages)) + 1,
            total_pages
        )

        metadata.append({
            "start_pos": i,
            "estimated_page": estimated_page
        })

    print(f" Created {len(chunks)} chunks")

    print("🔄 Generating local embeddings...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding {i + 1}/{len(chunks)}")
        embeddings.append(get_embedding(chunk))

    embeddings = np.array(embeddings, dtype="float32")

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    print("💾 Saving vector database...")
    faiss.write_index(index, "vectors.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata,
            "total_pages": total_pages
        }, f)

    print(" Vector database created successfully!")
    print(f" Saved files: vectors.index, chunks.pkl")
    print(f" Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    pdf_file = "Notes.pdf"
    pdf_to_vectors(pdf_file)
