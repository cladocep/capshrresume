import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
import os, time

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
collection_name = "resume_embeddings"

#1. buat/reset collection
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
#2. load data
df = pd.read_csv("/Users/rerepane/Desktop/FINAL CAPSTONE 3/RESUMEc/Resume.csv")
df = df.dropna(subset=["Resume_str", "Category"])

# Load all available data and use ID column from CSV
print(f"Total candidates available: {len(df)}")
print(f"ID column sample: {df['ID'].head().tolist()}")

texts = df["Resume_str"].tolist()
cats = df["Category"].tolist()
ids = df["ID"].tolist()  # Get IDs dari CSV

BATCH = 100
MAX_TOKENS_PER_REQUEST = 300000  # model limit
SAFETY_MARGIN_TOKENS = 1000      # leave headroom to avoid exceeding the limit
SAFE_MAX_TOKENS = MAX_TOKENS_PER_REQUEST - SAFETY_MARGIN_TOKENS
CHARS_PER_TOKEN = 4  # heuristic: ~4 characters per token

def est_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))

def split_text_chunks(text: str, max_tokens: int):
    # naive char-based splitter to ensure each chunk is below token limit
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

batch_count = 0
i = 0
while i < len(texts):
    current_batch_texts = []
    current_batch_cats = []
    current_batch_ids = []
    current_tokens = 0

    # accumulate texts until safe token limit or batch size reached
    while i < len(texts) and len(current_batch_texts) < BATCH:
        t = texts[i]
        c = cats[i]
        candidate_id = ids[i]  # Get ID dari CSV
        t_tokens = est_tokens(t)

        if t_tokens > SAFE_MAX_TOKENS:
            # split very long text into smaller chunks (use half the safe size as chunk target)
            chunks = split_text_chunks(t, max_tokens=max(1, SAFE_MAX_TOKENS // 2))
            added_any = False
            for ch in chunks:
                ch_tokens = est_tokens(ch)
                if current_tokens + ch_tokens > SAFE_MAX_TOKENS or len(current_batch_texts) >= BATCH:
                    break
                current_batch_texts.append(ch)
                current_batch_cats.append(c)
                current_batch_ids.append(candidate_id)
                current_tokens += ch_tokens
                added_any = True
            # ensure progress even if we could not add all chunks
            if not added_any:
                current_batch_texts.append(chunks[0])
                current_batch_cats.append(c)
                current_batch_ids.append(candidate_id)
                current_tokens += est_tokens(chunks[0])
                i += 1
            else:
                i += 1
        else:
            if current_tokens + t_tokens > SAFE_MAX_TOKENS:
                break
            current_batch_texts.append(t)
            current_batch_cats.append(c)
            current_batch_ids.append(candidate_id)
            current_tokens += t_tokens
            i += 1

    if not current_batch_texts:
        break

    batch_count += 1
    print(f"Batch {batch_count} (approx tokens: {current_tokens}, safe max: {SAFE_MAX_TOKENS})")

    #3. create embeddings
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=current_batch_texts
    )

    #4. prepare data to upload to qdrant
    points = []
    for idx, e in enumerate(emb.data):
        points.append(
            PointStruct(
                id=int(current_batch_ids[idx]),  # Use ID dari CSV
                vector=e.embedding,
                payload={
                    "text": current_batch_texts[idx],
                    "Category": current_batch_cats[idx]
                }
            )
        )

    #5. upload to qdrant
    qdrant.upsert(collection_name=collection_name, points=points)
    time.sleep(0.5)  # to avoid hitting rate limits

print("Index selesai dibuat")