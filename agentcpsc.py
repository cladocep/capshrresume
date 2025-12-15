from dotenv import load_dotenv
import os
import sys

try:
    import streamlit as st
    has_streamlit = True
except:
    has_streamlit = False
    st = None

from langchain_openai import ChatOpenAI
from ragcpc import get_retriever

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Try to get from Streamlit secrets if not in .env
if not OPENAI_API_KEY and has_streamlit:
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    except Exception as e:
        pass

# Validate API key exists
if not OPENAI_API_KEY:
    error_msg = """
    ❌ ERROR: OPENAI_API_KEY not found!
    
    Fix this:
    
    Option 1 (Local Development):
    1. Create .env file:
       cp .env.example .env
    2. Edit .env and add:
       OPENAI_API_KEY=sk-proj-xxxxx_your_key_xxxxx
    3. Restart app:
       streamlit run appc.py
    
    Option 2 (Streamlit Cloud):
    1. Go to: https://share.streamlit.io
    2. Click "Manage app" (lower right)
    3. Settings → Secrets
    4. Add:
       OPENAI_API_KEY = sk-proj-xxxxx_your_key_xxxxx
    5. Save (app will restart)
    
    Get OpenAI key from: https://platform.openai.com/api-keys
    """
    print(error_msg)
    if has_streamlit:
        st.error(error_msg)
    sys.exit(1)

# Initialize lazily to handle Streamlit Cloud environment
llm = None
retriever = None

def _init_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
    return llm

def _init_retriever():
    global retriever
    if retriever is None:
        retriever = get_retriever()
    return retriever

system_prompt = """
Kamu adalah Chatbot HR yang membantu recruiter mencari dan menganalisa kandidat berdasarkan database resume yang tersedia.

Aturan:
- Selalu lakukan analisa berdasarkan isi resume, meskipun kategori atau jabatan kandidat tidak tertulis secara eksplisit.
- Jangan menolak analisa hanya karena data tidak lengkap.
- Gunakan indikasi pengalaman, skill, aktivitas kerja, dan konteks resume untuk memperkirakan kecocokan kandidat.
- Jika relevansi lemah, tetap berikan estimasi kecocokan dan jelaskan alasannya.

""" 
def run_agent(query: str) -> dict:
    """
   Manual Rag :
    1. Cari kandidat dari retriever (Qdrant)
    2. Susun konteks kandidat.
    3. Kirim ke LLM + sistem prompt untuk di analisis."""

    # Initialize lazily
    retriever_instance = _init_retriever()
    llm_instance = _init_llm()

    # 1 rag : ambil kandidat dari retriever
    try:
        docs = retriever_instance.invoke(query)
    except Exception as e:
        return {
            "answer": f"Agent error saat retrieval: {str(e)}",
            "debug": {
                "query": query,
                "num_docs": 0,
                "doc_preview": [],
                "raw_documents": [],
                "error": str(e),
            },
        }

    # jika tidak ada dokumen ditemukan
    if not docs:
        return {
            "answer": "Maaf, tidak ditemukan kandidat yang sesuai dengan kriteria Anda.",
            "debug": {
                "query": query,
                "num_docs": 0,
                "doc_preview": [],
                "raw_documents": []
            },
        }

    # 2 rag : susun konteks kandidat
    doc_preview = []
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip() if doc.page_content else ""
        category = doc.metadata.get("Category", "Unknown")
        candidate_id = doc.metadata.get("id", i)  # Get actual ID from CSV, fallback to index
        similarity_score = doc.metadata.get("score", 0)  # Get similarity score
        snippet = text[:1200]

        context_parts.append(
            f"[Kandidat {i} - ID #{candidate_id}] Kategori: {category}\nResume:\n{snippet}"
        )
        doc_preview.append({
            "idx": candidate_id,  # Use actual ID from CSV, not just numbering
            "category": category,
            "snippet": snippet[:300],
            "score": round(similarity_score, 4),  # Similarity score from Qdrant
            "source": "RAG Search",  # Track that this came from RAG search
        })

    context = "\n".join(context_parts)

    #3 susun pesan untuk LLM
    messages = [
        {"role": "system",
            "content": system_prompt.strip()
         },
        {
            "role": "user",
            "content": f"""
Query recruiter
{query}
Berikut adalah data resume kandidat yang ditemukan:
{context}
Tugas anda:
- Jangan mengatakan tidak ada data
- Analisa masing-masing kandidat untuk posisi yang dimaksud recruiter
- Identifikasi indikasi pengalaman
- Tentukan kandidat paling relevan dan jelaskan alasannya secara singkat
- Jika relevansi lemah, tetap berikan estimasi dan ranking""",
        },
    ]
    # panggil llm
    response = llm_instance.invoke(messages)
    answer = response.content if hasattr(response, "content") else str (response)

    # return ke app
    return {
        "answer": answer,
        "debug": {
            "query": query,
            "num_docs": len(docs),
            "doc_preview": doc_preview,
            "raw_documents": docs  # Add raw Document objects from retriever
        },
    }
