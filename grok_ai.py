import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # Grok uses OpenAI-compatible client
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Any
import operator

load_dotenv()

# --- 1. Page Config ---
st.set_page_config(
    page_title="Pakistani Legal Advisor AI (Grok Powered)",
    page_icon="⚖️",
    layout="centered"
)

st.title("🏛️ Pakistani Legal Consultation AI")
st.caption("Your concise, pro-user assistant powered by Grok AI, Pakistani Law, and Court Precedents.")

# --- 2. Initialize Base Resources (Cached globally to save RAM) ---
@st.cache_resource
def init_base_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load shared pre-indexed base vector database (Statutes & Precedents)
    base_vector_store = Chroma(
        persist_directory="./pak_law_chroma_db",
        embedding_function=embeddings
    )
    
    # Safe check for Streamlit secrets or local env variable for Grok (xAI)
    api_key = None
    try:
        api_key = st.secrets.get("GROK_API_KEY")
    except Exception:
        pass
    
    if not api_key:
        api_key = os.getenv("GROK_API_KEY")
    
    # Initialize Grok via xAI's OpenAI-compatible API structure
    llm = ChatOpenAI(
        model="grok-beta",  # or grok-2 depending on your xAI model preference
        temperature=0.2,
        openai_api_key=api_key,
        openai_api_base="https://api.x.ai/v1"
    )
    return embeddings, base_vector_store, llm

embeddings, base_vector_store, llm = init_base_rag()

# --- 3. Isolated User Session Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = tempfile.mkdtemp()
    
    # Create a private, ephemeral Chroma collection unique to this visitor's browser session
    st.session_state.user_vector_store = Chroma(
        embedding_function=embeddings
    )

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []

# --- 4. Sidebar: Private Document Uploader ---
with st.sidebar:
    st.header("📂 Custom Law Library")
    st.write("Upload private legal PDFs. These stay **strictly private** to your current session and are wiped when you close the tab.")
    
    uploaded_file = st.file_uploader("Upload a Legal PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files_names:
            with st.spinner(f"Indexing {uploaded_file.name} privately into your session..."):
                temp_pdf_path = os.path.join(st.session_state.session_id, uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_pdf_path)
                private_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                private_chunks = text_splitter.split_documents(private_docs)
                
                st.session_state.user_vector_store.add_documents(private_chunks)
                st.session_state.uploaded_files_names.append(uploaded_file.name)
                
            st.success(f"Added {uploaded_file.name} to your private context!")

    if st.session_state.uploaded_files_names:
        st.markdown("**Your Active Private Files:**")
        for name in st.session_state.uploaded_files_names:
            st.text(f"🔒 {name}")

# --- 5. Combined Retriever ---
class CombinedRetriever:
    def __init__(self, base_store, user_store):
        self.base_retriever = base_store.as_retriever(search_kwargs={"k": 3})
        self.user_retriever = user_store.as_retriever(search_kwargs={"k": 3})
        
    def invoke(self, query):
        try:
            user_docs = self.user_retriever.invoke(query)
        except Exception:
            user_docs = []
        base_docs = self.base_retriever.invoke(query)
        return user_docs + base_docs

retriever = CombinedRetriever(base_vector_store, st.session_state.user_vector_store)

# --- 6. LangGraph Workflow Definition ---
class LegalRAGState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    retrieved_context: str
    rewritten_query: str

def query_rewriter_node(state: LegalRAGState):
    messages = state["messages"]
    if len(messages) <= 2: 
        return {"rewritten_query": messages[-1].content}
        
    prompt = [
        SystemMessage(content="Given the conversation history, rewrite the latest user input into a concise search query for Pakistani legal statutes and precedents. Output ONLY the search query text."),
    ] + messages
    
    response = llm.invoke(prompt)
    return {"rewritten_query": response.content.strip()}

def retrieval_node(state: LegalRAGState):
    query = state["rewritten_query"]
    docs = retriever.invoke(query)
    
    formatted_docs = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Custom Document"))
        formatted_docs.append(f"[Source: {source}]\n{doc.page_content}")
        
    return {"retrieved_context": "\n\n---\n\n".join(formatted_docs)}

def legal_generation_node(state: LegalRAGState):
    context = state["retrieved_context"]
    messages = state["messages"]
    
    system_prompt = f"""You are a concise, not too short but structured answer, pro-user Pakistani Legal Advisor. 
Your goal is to provide brief, highly structured, and well-cited answers. Avoid long blocks of text or heavy paragraphs.

Retrieved Legal Context (Statutes & Supreme Court Precedents):
\"\"\"
{context}
\"\"\"

Strict Output Format & Rules:
1. **Direct Answer**: Start immediately with a clear summary or direct response (e.g., "Yes, you have a strong legal right..." or "No, you cannot legally...").
2. **Mandatory Section / Article Citation**: You MUST explicitly cite the exact law, act, and section/article number (e.g., "Section 9 of the Specific Relief Act, 1877" or "Article 199 of the Constitution of Pakistan"). Do not skip this.
3. **Relevant Case Precedent**: Briefly name a relevant court case or order found in the context supporting this claim (e.g., "Relevant Precedent: Supreme Court Case C.A. No...").
4. **Actionable Next Steps**: Provide a short, 2-item bulleted list of immediate actions.
5. **Follow-up Question**: End with 1 short, direct question to gather any missing information.
"""

    chat_prompt = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(chat_prompt)
    return {"messages": [response]}

workflow = StateGraph(LegalRAGState)
workflow.add_node("rewriter", query_rewriter_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("generator", legal_generation_node)

workflow.add_edge(START, "rewriter")
workflow.add_edge("rewriter", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()

# --- 7. Streamlit UI Chat Loop ---
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
        with st.spinner("Analyzing standard laws, precedents, and your documents via Grok..."):
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