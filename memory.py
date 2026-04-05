import chromadb
import uuid
import time
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class MemoryStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_dir, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("memories")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add(self, text: str, metadata: dict | None = None) -> str:
        emb = self.embedder.encode([text]).tolist()
        doc_id = str(uuid.uuid4())

        meta = metadata or {}
        meta.setdefault("created_at", int(time.time()))
        meta.setdefault("source", "user")

        self.collection.add(
            ids=[doc_id],
            embeddings=emb,
            documents=[text],
            metadatas=[meta],
        )
        return doc_id

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def get(self, memory_id: str) -> dict | None:
        res = self.collection.get(ids=[memory_id], include=["documents", "metadatas"])
        if not res or not res.get("ids"):
            return None
        return {
            "id": res["ids"][0],
            "text": (res.get("documents") or [""])[0],
            "metadata": (res.get("metadatas") or [{}])[0],
        }

    def list_memories(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        # Chroma doesn't have a great "scan" API; get() without ids returns a page.
        res = self.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            {"id": ids[i], "text": docs[i], "metadata": metas[i]}
            for i in range(len(ids))
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        emb = self.embedder.encode([query]).tolist()
        res = self.collection.query(
            query_embeddings=emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out = []
        for i in range(len(docs)):
            out.append({"text": docs[i], "metadata": metas[i], "distance": dists[i]})
        return out

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return only the memory text strings, not metadata."""
        results = self.search(query, top_k)
        return [r["text"] for r in results]