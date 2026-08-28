import os
import requests
from fastapi import FastAPI, Request, Response
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
app = FastAPI()

# Initialize Supabase & Grok Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="legal_documents",
    query_name="match_legal_documents"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
api_key = os.getenv("GROK_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # Or another active Groq model
    temperature=0.2,
    groq_api_key=api_key
)

VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "pakistan_legal_bot_token")
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")

@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            return Response(content=challenge, media_type="text/plain", status_code=200)
        return Response(status_code=403)
    return Response(status_code=400)

@app.post("/webhook")
async def handle_instagram_message(request: Request):
    body = await request.json()
    
    try:
        if body.get("object") == "instagram":
            for entry in body.get("entry", []):
                for event in entry.get("messaging", []):
                    sender_id = event.get("sender", {}).get("id")
                    message_text = event.get("message", {}).get("text")
                    
                    if sender_id and message_text:
                        docs = retriever.invoke(message_text)
                        context = "\n\n---\n\n".join([d.page_content for d in docs])
                        
                        system_prompt = f"""You are a concise, structured Pakistani Legal Advisor answering an Instagram DM. Provide a direct answer, mandatory section/article citation, and a brief case precedent. Keep it mobile-friendly and brief.

Retrieved Legal Context:
\"\"\"
{context}
\"\"\"
"""
                        ai_response = llm.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=message_text)
                        ])
                        
                        send_instagram_reply(sender_id, ai_response.content)
                        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        
    return {"status": "ok"}

def send_instagram_reply(recipient_id: str, text: str):
    url = "https://graph.facebook.com/v18.0/me/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
        "access_token": ACCESS_TOKEN
    }
    requests.post(url, json=payload)