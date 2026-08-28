# Pakistani Legal AI Advisor

An AI-powered legal assistant designed to navigate the Pakistani legal system, providing instant, structured answers, relevant section citations, and case precedents through an intuitive web interface.

## Live Deployment
* **Web App**: [ai-legal-advisor-rag.streamlit.app](https://ai-legal-advisor-rag.streamlit.app/)

---

## The Core Idea
Navigating Pakistani laws—such as the Pakistan Penal Code (PPC), Criminal Procedure Code (CrPC), and civil statutes can be complex and time-consuming. This project bridges that gap by acting as an intelligent legal research partner. 

By combining Retrieval-Augmented Generation (RAG) with high-speed LLM inference, the system ingests comprehensive legal documents, indexes them via vector embeddings, and precisely retrieves applicable laws to answer complex natural language legal inquiries in seconds.

---

## How It Works (Architecture & Implementation)

1. **Data Ingestion & Embedding**:
   * Legal documents, case laws, and statutes are parsed and chunked.
   * OpenAI's embedding model (`text-embedding-3-small`) converts these legal texts into high-dimensional vector representations.

2. **Vector Database Storage (`pgvector`)**:
   * The vector embeddings and raw text are securely stored in a Supabase PostgreSQL database utilizing the `pgvector` extension for efficient similarity searches.

3. **Retrieval-Augmented Generation (RAG)**:
   * When a user inputs a legal query through the Streamlit interface, the app performs a semantic similarity search against Supabase to retrieve the most relevant sections, articles, and case histories.

4. **Structured Inference (Groq & LangChain)**:
   * The retrieved legal context is bundled into a targeted system prompt and sent to Groq for ultra-fast model inference.
   * The model formats the final output to ensure it is structured, mobile-friendly, and includes precise statutory citations and case precedents.

---

## Tech Stack
* **Frontend UI**: Streamlit
* **Database & Vector Store**: Supabase (PostgreSQL with `pgvector`)
* **LLM & Inference**: Groq SDK
* **Embeddings**: OpenAI API (`text-embedding-3-small`)
* **Orchestration**: Python & LangChain

---

## Environment Variables
To run or deploy this project locally, configure the following keys in your `.env` file:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Local Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-legal-advisor.git
   cd ai-legal-advisor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
