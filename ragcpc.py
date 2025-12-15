from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import sys

try:
    import streamlit as st
    has_streamlit = True
except:
    has_streamlit = False
    st = None

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Try to get from Streamlit secrets if not in .env
if not QDRANT_URL and has_streamlit:
    try:
        QDRANT_URL = st.secrets.get("QDRANT_URL")
    except:
        pass

if not QDRANT_API_KEY and has_streamlit:
    try:
        QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")
    except:
        pass

if not OPENAI_API_KEY and has_streamlit:
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    except:
        pass

# Validate all required keys exist
missing_keys = []
if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")
if not QDRANT_URL:
    missing_keys.append("QDRANT_URL")
if not QDRANT_API_KEY:
    missing_keys.append("QDRANT_API_KEY")

if missing_keys:
    error_msg = f"""
    ❌ ERROR: Missing required environment variables: {', '.join(missing_keys)}
    
    Fix this:
    
    Option 1 (Local Development):
    1. Create .env file:
       cp .env.example .env
    2. Edit .env and add all 3 keys:
       OPENAI_API_KEY=sk-proj-xxxxx
       QDRANT_URL=https://xxxxx.us-east-1-0.aws.cloud.qdrant.io
       QDRANT_API_KEY=ey...
    3. Restart app:
       streamlit run appc.py
    
    Option 2 (Streamlit Cloud):
    1. Go to: https://share.streamlit.io
    2. Find your app
    3. Click "Manage app" (lower right)
    4. Settings → Secrets
    5. Add all 3 keys
    6. Save (app will restart)
    
    Get keys from:
    - OpenAI: https://platform.openai.com/api-keys
    - Qdrant: https://cloud.qdrant.io/
    """
    print(error_msg)
    if has_streamlit:
        st.error(error_msg)
        st.stop()
    else:
        raise ValueError(error_msg)

collection_name = "resume_embeddings"

# Initialize lazily to handle Streamlit Cloud environment
embeddings = None
qdrant_client = None

def _init_embeddings():
    global embeddings
    if embeddings is None:
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Error initializing OpenAIEmbeddings: {e}")
            raise
    return embeddings

def _init_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=QDRANT_URL, api_key=QDRANT_API_KEY
        )
    return qdrant_client

class CustomQdrantRetriever:
    """Custom retriever yang extract teks dari Qdrant dengan proper payload handling."""
    
    def __init__(self, k=5):
        self.embeddings = None  # Will be initialized lazily
        self.client = None  # Will be initialized lazily
        self.k = k
    
    def _get_embeddings(self):
        """Get or initialize embeddings"""
        if self.embeddings is None:
            self.embeddings = _init_embeddings()
        return self.embeddings
    
    def _get_client(self):
        """Get or initialize Qdrant client"""
        if self.client is None:
            self.client = _init_qdrant_client()
        return self.client
    
    def invoke(self, query: str):
        """Search dan rekonstruksi Documents dengan teks dari payload."""
        try:
            embeddings = self._get_embeddings()
            client = self._get_client()
            
            # Embed query
            query_embedding = embeddings.embed_query(query)
            
            # Use QdrantClient's query_points method with search_params
            from qdrant_client.models import PointStruct
            
            # Query using scroll/search alternatives
            # Get all points first then filter - not ideal but works
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )
            
            # Calculate similarity scores manually
            import numpy as np
            query_vec = np.array(query_embedding)
            
            scored_points = []
            for point in points:
                if point.vector:
                    vec = np.array(point.vector)
                    # Cosine similarity
                    score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10)
                    scored_points.append((point, score))
            
            # Sort by score and take top k
            scored_points.sort(key=lambda x: x[1], reverse=True)
            top_points = scored_points[:self.k]
            
            # Rekonstruksi LangChain Documents
            documents = []
            for point, score in top_points:
                payload = point.payload or {}
                
                # Ambil teks dari payload
                page_content = payload.get("text", "")
                
                # Metadata
                metadata = {
                    "id": point.id,
                    "Category": payload.get("Category") or payload.get("category", "Unknown"),
                    "score": score,
                }
                
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

def get_retriever():
    """Return custom retriever."""
    return CustomQdrantRetriever(k=5)