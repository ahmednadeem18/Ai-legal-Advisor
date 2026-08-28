import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from datasets import load_dataset
from langchain_core.documents import Document
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

all_docs = []

# --- Part 1: Load Local Law Books ---
if os.path.exists("./lawbooks"):
    print("Loading local PDFs from ./lawbooks ...")
    loader = PyPDFDirectoryLoader("./lawbooks")
    law_docs = loader.load()
    for doc in law_docs:
        doc.metadata["type"] = "statute_book"
    all_docs.extend(law_docs)
    print(f"Loaded {len(law_docs)} pages from law books.")
else:
    print("No ./lawbooks folder found. Skipping local books.")

# --- Part 2: Load Hugging Face Precedents ---
print("Fetching Supreme Court precedents from Hugging Face...")
dataset = load_dataset("Ibtehaj10/supreme-court-of-pak-judgments", split="train")

LIMIT = 220
print(f"Selecting {LIMIT} Supreme Court precedents...")
sample_data = dataset.select(range(min(LIMIT, len(dataset))))

precedent_docs = []
for item in sample_data:
    text_content = item.get("text", "")
    case_meta = item.get("case_details", "Supreme Court Precedent")
    citation = item.get("citation_number", "")
    
    if text_content:
        precedent_docs.append(
            Document(
                page_content=text_content,
                metadata={
                    "source": f"{case_meta} (Citation: {citation})",
                    "type": "court_precedent"
                }
            )
        )
all_docs.extend(precedent_docs)
print(f"Added {len(precedent_docs)} precedent documents.")

# --- Part 3: Chunk Everything Uniformly ---
print("Chunking all combined documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(all_docs)
print(f"Total chunks generated: {len(chunks)}. Uploading to Supabase cloud...")

# --- Part 4: Batch Upload to Supabase ---
SupabaseVectorStore.from_documents(
    chunks,
    embeddings,
    client=supabase,
    table_name="legal_documents",
    query_name="match_legal_documents",
    chunk_size=100
)

print("✅ Successfully ingested both law books and Supreme Court precedents directly to Supabase cloud!")