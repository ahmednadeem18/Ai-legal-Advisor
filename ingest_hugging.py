import os
import shutil
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Reset old heavy database
DB_PATH = "./pak_law_chroma_db"


# 2. Download dataset
print("Fetching dataset from Hugging Face...")
dataset = load_dataset("Ibtehaj10/supreme-court-of-pak-judgments", split="train")

# 3. Set a balanced limit (e.g., 350 records — large enough for diverse cases, small enough for GitHub)
LIMIT = 120
print(f"Selecting {LIMIT} Supreme Court precedents for a balanced, GitHub-friendly database size...")
sample_data = dataset.select(range(min(LIMIT, len(dataset))))

docs = []
for item in sample_data:
    text_content = item.get("text", "")
    case_meta = item.get("case_details", "Supreme Court Precedent")
    citation = item.get("citation_number", "")
    
    if text_content:
        docs.append(
            Document(
                page_content=text_content,
                metadata={
                    "source": f"{case_meta} (Citation: {citation})",
                    "type": "court_precedent"
                }
            )
        )

# 4. Chunk with efficient sizes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)
print(f"Generated {len(chunks)} text chunks. Starting embedding and batch indexing...")

# 5. Embed and save to Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_store.add_documents(batch)
    print(f"Progress: Processed batch {i // batch_size + 1} / {(len(chunks) + batch_size - 1) // batch_size}")

print("✅ Balanced database created successfully!")