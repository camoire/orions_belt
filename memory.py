# memory.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

class MemoryStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_dir, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("memories")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add(self, text: str, metadata: dict | None = None):
        emb = self.embedder.encode([text]).tolist()
        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=emb,
            documents=[text],
            metadatas=[metadata or {}],
        )
        return doc_id

    def search(self, query: str, top_k: int = 5):
        emb = self.embedder.encode([query]).tolist()
        res = self.collection.query(
            query_embeddings=emb,
            n_results=top_k,
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(docs, metas))

    def retrieve(self, query: str, top_k: int = 5):
        """Return only the memory text strings, not metadata."""
        results = self.search(query, top_k)
        return [doc for doc, meta in results]
