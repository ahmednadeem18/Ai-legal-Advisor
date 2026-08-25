from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

print("Downloading Supreme Court precedents dataset from Hugging Face...")
dataset = load_dataset("Ibtehaj10/supreme-court-of-pak-judgments", split="train")

print(f"Loaded {len(dataset)} precedent records. Preparing text chunks...")

docs = []
for item in dataset:
    text_content = item.get("text", "")
    case_meta = item.get("case_details", "Supreme Court Precedent")
    
    if text_content:
        docs.append(
            Document(
                page_content=text_content,
                metadata={"source": str(case_meta), "type": "court_precedent"}
            )
        )

# Chunk the judgments
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"Total chunks generated: {len(chunks)}. Starting batch embedding...")

# Initialize embeddings and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./pak_law_chroma_db",
    embedding_function=embeddings
)

# Batch insert to prevent hanging (100 chunks at a time)
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_store.add_documents(batch)
    print(f"Progress: Processed and embedded batch {i // batch_size + 1} / {(len(chunks) + batch_size - 1) // batch_size}")

print("✅ Court precedents successfully embedded and added to your database without freezing!")