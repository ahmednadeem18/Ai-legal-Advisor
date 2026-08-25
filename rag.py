import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Any
import operator

load_dotenv()

# --- 1. Page Config ---
st.set_page_config(
    page_title="Pakistani Legal Advisor AI",
    page_icon="⚖️",
    layout="centered"
)

st.title("🏛️ Pakistani Legal Consultation AI")
st.caption("Your pro-user assistant for Pakistani Law (with private custom document support)")

# --- 2. Initialize Global Resources (Base Laws) ---
@st.cache_resource
def init_base_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Base shared database
    base_vector_store = Chroma(
        persist_directory="./pak_law_chroma_db",
        embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )
    return embeddings, base_vector_store, llm

embeddings, base_vector_store, llm = init_base_rag()

# --- 3. Session State Management for User-Specific Data ---
if "session_id" not in st.session_state:
    # Create a unique temporary directory for this specific user session
    st.session_state.session_id = tempfile.mkdtemp()
    st.session_state.user_db_dir = os.path.join(st.session_state.session_id, "chroma_db")
    
    # Copy or initialize a user-specific vector store pointing to this private directory
    st.session_state.user_vector_store = Chroma(
        persist_directory=st.session_state.user_db_dir,
        embedding_function=embeddings
    )
    
    # Add base documents to user store initially by merging or pointing retriever to both
    # For simplicity, we create a retriever that searches BOTH base law and user private law:

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []

# --- 4. Sidebar: Private Document Uploader ---
with st.sidebar:
    st.header("📂 Custom Law Library")
    st.write("Upload private legal PDFs (e.g., Copyright, Trademark laws). These will **only** apply to your current session and remain private to you.")
    
    uploaded_file = st.file_uploader("Upload a Legal PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files_names:
            with st.spinner(f"Processing and indexing {uploaded_file.name} privately..."):
                # Save uploaded file to user's private temp folder
                temp_pdf_path = os.path.join(st.session_state.session_id, uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and chunk the private PDF
                loader = PyPDFLoader(temp_pdf_path)
                private_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                private_chunks = text_splitter.split_documents(private_docs)
                
                # Add chunks ONLY to the user's private vector store
                st.session_state.user_vector_store.add_documents(private_chunks)
                st.session_state.uploaded_files_names.append(uploaded_file.name)
                
            st.success(f"Successfully added {uploaded_file.name} to your private context!")

    if st.session_state.uploaded_files_names:
        st.markdown("**Your Private Documents:**")
        for name in st.session_state.uploaded_files_names:
            st.text(f"🔒 {name}")

# --- 5. Unified Retriever (Searches Base Laws + User's Private Laws) ---
# We combine results from both the main DB and the private user DB
class CombinedRetriever:
    def __init__(self, base_store, user_store):
        self.base_retriever = base_store.as_retriever(search_kwargs={"k": 3})
        self.user_retriever = user_store.as_retriever(search_kwargs={"k": 3})
        
    def invoke(self, query):
        base_docs = self.base_retriever.invoke(query)
        user_docs = self.user_retriever.invoke(query)
        # Combine user private docs first so they take priority, followed by base docs
        return user_docs + base_docs

retriever = CombinedRetriever(base_vector_store, st.session_state.user_vector_store)

# --- 6. Define LangGraph State & Nodes ---
class LegalRAGState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    retrieved_context: str
    rewritten_query: str

def query_rewriter_node(state: LegalRAGState):
    messages = state["messages"]
    if len(messages) <= 2: 
        return {"rewritten_query": messages[-1].content}
        
    prompt = [
        SystemMessage(content="Given the conversation history, rewrite the latest user input into a concise search query for Pakistani legal statutes. Output ONLY the search query text."),
    ] + messages
    
    response = llm.invoke(prompt)
    return {"rewritten_query": response.content.strip()}

def retrieval_node(state: LegalRAGState):
    query = state["rewritten_query"]
    docs = retriever.invoke(query)
    
    formatted_docs = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Private Custom Document"))
        formatted_docs.append(f"[Source: {source}]\n{doc.page_content}")
        
    return {"retrieved_context": "\n\n---\n\n".join(formatted_docs)}

def legal_generation_node(state: LegalRAGState):
    context = state["retrieved_context"]
    messages = state["messages"]
    
    system_prompt = f"""You are a friendly, supportive, and pro-user Pakistani Legal Advisor. 
Your goal is to protect the user's rights, explain things in simple everyday language, and avoid heavy legalese or dense blocks of text.

Retrieved Legal Context (including any private custom documents uploaded by the user):
\"\"\"
{context}
\"\"\"

Formatting and Rule-Mentioning Rules:
1. **Mandatory Rule/Section Number**: Whenever you mention a legal right, claim, or remedy, you MUST clearly state the specific rule or section number (e.g., "Section 9 of the Specific Relief Act, 1877", or sections from user-uploaded laws). 
2. **Plain Language**: Avoid difficult legal jargon. Translate complex rules into simple terms a regular person can understand.
3. **Sector Categorization**: Start by clearly stating the legal sector (e.g., Property & Civil Law, Criminal Law, Intellectual Property).
4. **Situation Analysis**: Explain clearly how the law protects or favors the user based on their specific situation.
5. **Actionable Next Steps**: Provide a short, easy-to-follow bulleted list of what they can do next.
6. **Follow-up**: Ask 1 simple, friendly question to gather any missing facts needed to help them better.
7. **Clean Formatting**: Use clear bullet points, bold headings, and line breaks. Avoid massive walls of text.
"""

    chat_prompt = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(chat_prompt)
    return {"messages": [response]}

# Build Workflow Graph
workflow = StateGraph(LegalRAGState)
workflow.add_node("rewriter", query_rewriter_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("generator", legal_generation_node)

workflow.add_edge(START, "rewriter")
workflow.add_edge("rewriter", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()

# --- 7. Streamlit Chat Interface ---

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Describe your legal issue or ask about your uploaded laws..."):
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    lc_messages = []
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Analyzing standard laws and your private documents..."):
            try:
                state_output = app_graph.invoke({
                    "messages": lc_messages,
                    "retrieved_context": "",
                    "rewritten_query": ""
                })
                ai_reply = state_output["messages"][-1].content
                st.markdown(ai_reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})
            except Exception as e:
                st.error(f"An error occurred: {e}")