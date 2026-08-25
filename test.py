import dotenv
dotenv.load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Local Embeddings & Chroma DB
print("Loading vector database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./pak_law_chroma_db",
    embedding_function=embeddings
)

# 2. Test Retrieval
query = "Someone dispossessed me from my property without legal process"
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(query)

print("\n--- Retrieved Legal Chunks from DB ---")
for doc in docs:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(doc.page_content[:300] + "...")
    print("-" * 50)

# 3. Initialize Gemini Model (from Google AI Studio)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# 4. Generate Grounded Legal Response
context_text = "\n\n".join([d.page_content for d in docs])

prompt = ChatPromptTemplate.from_template("""
You are a legal assistant specializing in Pakistani Law.
Answer the user query based ONLY on the following retrieved statutory provisions:

<context>
{context}
</context>

User Query: {query}

Provide a clear and concise explanation including relevant sections/articles and the next legal steps.
""")

chain = prompt | llm

print("\n--- Generating Gemini Response ---")
response = chain.invoke({"context": context_text, "query": query})
print("\nGemini Response:\n")
print(response.content)