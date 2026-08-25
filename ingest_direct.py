import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load PDFs from the folder
print("Loading PDFs from ./lawbooks ...")
loader = PyPDFDirectoryLoader("./lawbooks")
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} pages.")

# 2. Chunk text with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(raw_docs)
print(f"Created {len(chunks)} searchable chunks.")

# 3. Free local embedding model (downloads once automatically ~80MB)
print("Initializing free local embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Save into ChromaDB
print("Creating Chroma vector store...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./pak_law_chroma_db"
)

print("All law books indexed successfully into './pak_law_chroma_db'!")
