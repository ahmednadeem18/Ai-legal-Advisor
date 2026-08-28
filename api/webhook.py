import os
import requests
from fastapi import FastAPI, Request, Response
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import create_client
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Use OpenAI API for embeddings instead of local sentence-transformers to keep bundle size under 500MB
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="legal_documents",
    query_name="match_legal_documents"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

api_key = os.getenv("GROK_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    groq_api_key=api_key
)

VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "pakistan_legal_bot_token")
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
@app.get("/")
async def root():
    return {"status": "online", "message": "Pakistani Legal Advisor Instagram Webhook is running!"}
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
from fastapi.responses import RedirectResponse
import urllib.parse

@app.get("/auth/login")
async def instagram_login():
    client_id = os.getenv("META_APP_ID")
    redirect_uri = os.getenv("META_REDIRECT_URI") # e.g., https://ai-legal-advisor-bx36.vercel.app/auth/callback
    
    # Required permissions for managing Instagram messages and basic profile
    scope = "instagram_basic,instagram_manage_messages,pages_show_list,pages_messaging"
    
    auth_url = (
        f"https://www.facebook.com/v18.0/dialog/oauth?"
        f"client_id={client_id}&redirect_uri={urllib.parse.quote(redirect_uri)}&"
        f"scope={scope}&response_type=code"
    )
    return RedirectResponse(url=auth_url)
@app.get("/auth/callback")
async def auth_callback(code: str):
    client_id = os.getenv("META_APP_ID")
    client_secret = os.getenv("META_APP_SECRET")
    redirect_uri = os.getenv("META_REDIRECT_URI")
    
    # 1. Exchange code for short-lived access token
    token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code": code
    }
    response = requests.get(token_url, params=params)
    token_data = response.json()
    short_token = token_data.get("access_token")
    
    if not short_token:
        return {"error": "Failed to obtain access token"}
        
    # 2. Upgrade to a long-lived token (approx. 60 days)
    long_token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
    long_params = {
        "grant_type": "fb_exchange_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "fb_exchange_token": short_token
    }
    long_response = requests.get(long_token_url, params=long_params)
    long_token = long_response.json().get("access_token")

    # 3. Fetch Facebook Pages connected to user, then find the associated Instagram Business Account ID
    pages_url = f"https://graph.facebook.com/v18.0/me/accounts?access_token={long_token}"
    pages_res = requests.get(pages_url).json()
    
    # Store the long_token and ig_business_account_id securely in your Supabase database tied to this business user
    for page in pages_res.get("data", []):
        page_id = page["id"]
        page_token = page["access_token"]
        
        ig_info = requests.get(
            f"https://graph.facebook.com/v18.0/{page_id}?fields=instagram_business_account&access_token={page_token}"
        ).json()
        
        ig_account_id = ig_info.get("instagram_business_account", {}).get("id")
        if ig_account_id:
            # TODO: Save ig_account_id and long_token into Supabase database table `business_tokens`
            pass

    return {"status": "success", "message": "Instagram Business account successfully linked!"}
def send_instagram_reply(recipient_id: str, text: str):
    url = "https://graph.facebook.com/v18.0/me/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
        "access_token": ACCESS_TOKEN
    }
    requests.post(url, json=payload)
