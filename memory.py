import re
import time
import uuid
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class MemoryStore:
    """ChromaDB-backed store for long-term memories with importance scoring."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        client: Any = None,
        embedder: Any = None,
    ):
        if client is not None:
            self.client = client
        elif chromadb is not None:
            self.client = chromadb.Client(
                Settings(persist_directory=persist_dir, anonymized_telemetry=False)
            )
        else:
            self.client = None

        if self.client is not None:
            self.collection = self.client.get_or_create_collection("memories")
        else:
            self.collection = None

        if embedder is not None:
            self.embedder = embedder
        elif SentenceTransformer is not None:
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.embedder = None

    def _encode(self, text: str) -> list[float]:
        if self.embedder is None:
            return []
        return self.embedder.encode([text]).tolist()

    def count(self) -> int:
        """Return the total number of memories stored."""
        if self.collection is None:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

    def add(
        self,
        text: str,
        metadata: dict | None = None,
        importance: float | int | None = None,
    ) -> str:
        """Add a memory with importance score (1-10) and metadata to ChromaDB."""
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Memory text cannot be empty.")

        meta = dict(metadata or {})
        meta.setdefault("created_at", int(time.time()))
        meta.setdefault("source", "user")

        # Importance scoring: range 1.0 to 10.0, default 5.0
        if importance is not None:
            score = float(importance)
        elif "importance" in meta:
            score = float(meta["importance"])
        else:
            score = 5.0
        score = max(1.0, min(10.0, score))
        meta["importance"] = score

        # Check for near-duplicate memory to avoid clutter
        if self.collection is not None:
            try:
                existing_res = self.collection.get(
                    where={"source": meta.get("source", "user")},
                    include=["documents", "metadatas"],
                    limit=100,
                )
                existing_docs = existing_res.get("documents") or []
                existing_ids = existing_res.get("ids") or []
                for idx, doc in enumerate(existing_docs):
                    if doc.strip().lower() == cleaned_text.lower():
                        existing_id = existing_ids[idx]
                        # Update importance if higher
                        existing_meta = (existing_res.get("metadatas") or [{}])[idx]
                        if score > float(existing_meta.get("importance", 5.0)):
                            self.update(existing_id, importance=score)
                        return existing_id
            except Exception:
                pass

        doc_id = str(uuid.uuid4())
        emb = self._encode(cleaned_text)

        if self.collection is not None:
            add_kwargs = {
                "ids": [doc_id],
                "documents": [cleaned_text],
                "metadatas": [meta],
            }
            if emb:
                add_kwargs["embeddings"] = emb
            self.collection.add(**add_kwargs)

        return doc_id

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        metadata: dict | None = None,
        importance: float | int | None = None,
    ) -> bool:
        """Update an existing memory document, metadata, or importance score."""
        if self.collection is None:
            return False

        existing = self.get(memory_id)
        if not existing:
            return False

        new_text = text.strip() if text else existing["text"]
        new_meta = dict(existing.get("metadata", {}))
        if metadata:
            new_meta.update(metadata)

        if importance is not None:
            new_meta["importance"] = max(1.0, min(10.0, float(importance)))
        new_meta["updated_at"] = int(time.time())

        update_kwargs = {
            "ids": [memory_id],
            "documents": [new_text],
            "metadatas": [new_meta],
        }
        if text and self.embedder is not None:
            emb = self._encode(new_text)
            if emb:
                update_kwargs["embeddings"] = emb

        self.collection.update(**update_kwargs)
        return True

    def delete(self, memory_id: str) -> None:
        """Delete a memory by its unique ID."""
        if self.collection is not None:
            self.collection.delete(ids=[memory_id])

    def get(self, memory_id: str) -> dict | None:
        """Retrieve a single memory by ID, including text, importance, and metadata."""
        if self.collection is None:
            return None
        res = self.collection.get(ids=[memory_id], include=["documents", "metadatas"])
        if not res or not res.get("ids"):
            return None
        meta = (res.get("metadatas") or [{}])[0] or {}
        return {
            "id": res["ids"][0],
            "text": (res.get("documents") or [""])[0],
            "importance": float(meta.get("importance", 5.0)),
            "metadata": meta,
        }

    def list_memories(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List memories with pagination, including importance scores and metadata."""
        if self.collection is None:
            return []
        res = self.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            {
                "id": ids[i],
                "text": docs[i],
                "importance": float(metas[i].get("importance", 5.0)),
                "metadata": metas[i],
            }
            for i in range(len(ids))
        ]

    def get_top_by_importance(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top memories ranked primarily by importance score."""
        memories = self.list_memories(limit=100)
        memories.sort(
            key=lambda m: (
                float(m.get("importance", 5.0)),
                m.get("metadata", {}).get("created_at", 0),
            ),
            reverse=True,
        )
        out = []
        for m in memories[:top_k]:
            out.append({
                "id": m["id"],
                "text": m["text"],
                "importance": m["importance"],
                "score": round(m["importance"] / 10.0, 4),
                "distance": 0.0,
                "metadata": m.get("metadata", {}),
            })
        return out

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Search memories using hybrid ranking: vector similarity + importance score."""
        total = self.count()
        if total == 0 or self.collection is None:
            return []

        if not query or not query.strip():
            return self.get_top_by_importance(top_k=top_k)

        emb = self._encode(query)
        if not emb:
            return self.get_top_by_importance(top_k=top_k)

        # Retrieve a candidate pool to re-rank with importance score
        n_candidates = min(max(top_k * 3, 10), total)
        res = self.collection.query(
            query_embeddings=emb,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        candidates = []
        for i in range(len(docs)):
            dist = float(dists[i]) if dists else 0.0
            meta = metas[i] if i < len(metas) else {}
            imp = float(meta.get("importance", 5.0))
            if imp < min_importance:
                continue

            # Normalized similarity: cosine distance d in [0, 2], sim in [0, 1]
            similarity = 1.0 / (1.0 + max(0.0, dist))
            normalized_imp = max(1.0, min(10.0, imp)) / 10.0

            # Combined ranking: 70% semantic relevance + 30% importance score
            combined_score = (0.7 * similarity) + (0.3 * normalized_imp)

            doc_id = ids[i] if i < len(ids) else ""
            candidates.append({
                "id": doc_id,
                "text": docs[i],
                "importance": imp,
                "score": round(combined_score, 4),
                "distance": round(dist, 4),
                "metadata": meta,
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return the top 5 memory text strings ranked by relevance and importance."""
        results = self.search(query, top_k=top_k)
        return [r["text"] for r in results]


_HEURISTIC_PATTERNS = [
    # Allergies and health constraints (top priority)
    (re.compile(r"\b(?:i am|i'm)\s+allergic to\s+([^.,;!\n]+)", re.I), "User is allergic to {match}", 10.0),
    # Names and identity
    (re.compile(r"\b(?:my name is|i am called|call me)\s+([A-Z][a-zA-Z0-9_\- ]+?)(?:[.,;!\n]|$)", re.I), "User's name is {match}", 9.0),
    # Profession / role
    (re.compile(r"\b(?:i am an?|i'm an?|i work as an?)\s+([a-zA-Z0-9_\- ]+?)(?:\s+at|\s+for|[.,;!\n]|$)", re.I), "User is a {match}", 8.0),
    # Explicit memory instructions
    (re.compile(r"\b(?:please remember that|remember that|note that|keep in mind that)\s+([^.,;!\n]+)", re.I), "{match}", 8.0),
    # Location
    (re.compile(r"\b(?:i live in|i'm from|i am from|i reside in|i am based in)\s+([^.,;!\n]+)", re.I), "User lives in {match}", 7.0),
    # Strong preferences / dislikes
    (re.compile(r"\b(?:i prefer|i always use)\s+([^.,;!\n]+)", re.I), "User prefers {match}", 7.0),
    (re.compile(r"\b(?:i really like|i like|i love)\s+([^.,;!\n]+)", re.I), "User likes {match}", 6.0),
    (re.compile(r"\b(?:i dislike|i hate|i don't like|i do not like)\s+([^.,;!\n]+)", re.I), "User dislikes {match}", 6.0),
    # Favorites
    (re.compile(r"\b(?:my favorite\s+[^is\n]+is)\s+([^.,;!\n]+)", re.I), "User's {full}", 6.0),
    # Languages and skills
    (re.compile(r"\b(?:i speak|i write code in|i program in)\s+([^.,;!\n]+)", re.I), "User knows {match}", 7.0),
    # Pets / family
    (re.compile(r"\b(?:i have a (?:dog|cat|pet)(?:\s+named|\s+called)?)\s+([^.,;!\n]+)", re.I), "User has a pet {match}", 7.0),
]


class MemoryExtractor:
    """Extracts long-term user memories and importance scores with text patterns."""

    def _heuristic_extract(self, text: str) -> List[Dict[str, Any]]:
        extracted = []
        seen = set()
        for pattern, template, importance in _HEURISTIC_PATTERNS:
            for m in pattern.finditer(text):
                match_val = m.group(1).strip() if m.groups() else ""
                full_val = m.group(0).strip()
                if not match_val and "{match}" in template:
                    continue
                statement = template.format(match=match_val, full=full_val).strip()
                # Normalize statement
                statement = statement[0].upper() + statement[1:] if statement else ""
                key = statement.lower()
                if key and key not in seen:
                    seen.add(key)
                    extracted.append({
                        "text": statement,
                        "importance": float(importance),
                    })
        return extracted

    def extract(
        self,
        user_text: str | None,
    ) -> List[Dict[str, Any]]:
        """Extract memories from user text with importance scores."""
        user_text = (user_text or "").strip()
        if not user_text or len(user_text) < 4:
            return []

        # Trivial greetings check
        if re.fullmatch(r"(?i)(?:hi|hello|hey|thanks|thank you|ok|okay|bye|good morning)[.! ]*", user_text):
            return []

        return self._heuristic_extract(user_text)

    def extract_and_store(
        self,
        memory_store: MemoryStore,
        user_text: str | None,
        source: str = "automatic_extraction",
    ) -> List[Dict[str, Any]]:
        """Extract memories from conversation and store them in ChromaDB."""
        extracted = self.extract(user_text)
        stored = []
        for item in extracted:
            doc_id = memory_store.add(
                text=item["text"],
                importance=item["importance"],
                metadata={"source": source},
            )
            stored.append({
                "id": doc_id,
                "text": item["text"],
                "importance": item["importance"],
            })
        return stored
