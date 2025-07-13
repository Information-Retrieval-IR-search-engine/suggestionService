from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import joblib
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# Load SentenceTransformer model
model = SentenceTransformer(f'quora_mpnet_v2_tuned_v3')

# Load FAISS index and queries
queries_index = faiss.read_index(f"quora_query_faiss.index")
all_queries = joblib.load(f"all_quora_queries.joblib")

def suggest_similar_queries(dataset, new_query, top_k=5):
    new_query_embedding = model.encode([new_query], convert_to_numpy=True)
    D, I = queries_index.search(new_query_embedding, top_k)
    return [all_queries[i][1] for i in I[0]]

@app.get("/suggest")
def suggest(q: str):
    return suggest_similar_queries('quora',q,top_k=3)
