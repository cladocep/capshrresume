from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

collection_name = "resume_embeddings"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=OPENAI_API_KEY
)
qdrant_client = QdrantClient(
    url=QDRANT_URL, api_key=QDRANT_API_KEY
)

class CustomQdrantRetriever:
    """Custom retriever yang extract teks dari Qdrant dengan proper payload handling."""
    
    def __init__(self, k=5):
        self.embeddings = embeddings
        self.client = qdrant_client
        self.k = k
    
    def invoke(self, query: str):
        """Search dan rekonstruksi Documents dengan teks dari payload."""
        try:
            # Embed query
            query_embedding = self.embeddings.embed_query(query)
            
            # Use QdrantClient's query_points method with search_params
            from qdrant_client.models import PointStruct
            
            # Query using scroll/search alternatives
            # Get all points first then filter - not ideal but works
            points, _ = self.client.scroll(
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