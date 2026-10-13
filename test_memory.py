import unittest
from unittest.mock import MagicMock
import time

from memory import MemoryStore, MemoryExtractor, EXTRACTION_SYSTEM_PROMPT
from prompt_builder import build_messages, build_prompt


class MockCollection:
    """Mock ChromaDB collection for unit testing without full Chroma dependencies."""

    def __init__(self):
        self._data = {}

    def count(self):
        return len(self._data)

    def add(self, ids, documents, metadatas, embeddings=None):
        for i, doc_id in enumerate(ids):
            self._data[doc_id] = {
                "id": doc_id,
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i] if embeddings else [0.0],
            }

    def update(self, ids, documents, metadatas, embeddings=None):
        for i, doc_id in enumerate(ids):
            if doc_id in self._data:
                self._data[doc_id]["document"] = documents[i]
                self._data[doc_id]["metadata"] = metadatas[i]
                if embeddings:
                    self._data[doc_id]["embedding"] = embeddings[i]

    def delete(self, ids):
        for doc_id in ids:
            self._data.pop(doc_id, None)

    def get(self, ids=None, where=None, limit=50, offset=0, include=None):
        items = list(self._data.values())
        if ids:
            items = [item for item in items if item["id"] in ids]
        if where:
            for k, v in where.items():
                items = [item for item in items if item["metadata"].get(k) == v]

        paged = items[offset : offset + limit]
        return {
            "ids": [item["id"] for item in paged],
            "documents": [item["document"] for item in paged],
            "metadatas": [item["metadata"] for item in paged],
        }

    def query(self, query_embeddings, n_results=5, include=None):
        items = list(self._data.values())[:n_results]
        return {
            "ids": [[item["id"] for item in items]],
            "documents": [[item["document"] for item in items]],
            "metadatas": [[item["metadata"] for item in items]],
            "distances": [[0.1 * (idx + 1) for idx in range(len(items))]],
        }


class MockEmbedder:
    """Mock SentenceTransformer embedder."""

    def encode(self, texts):
        mock_arr = MagicMock()
        mock_arr.tolist.return_value = [[0.1, 0.2, 0.3] for _ in texts]
        return mock_arr


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_collection = MockCollection()
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        self.mock_embedder = MockEmbedder()
        self.store = MemoryStore(
            client=self.mock_client,
            embedder=self.mock_embedder,
        )

    def test_add_and_get_memory_with_importance(self):
        doc_id = self.store.add("User is a software engineer", importance=8.5)
        self.assertIsNotNone(doc_id)

        item = self.store.get(doc_id)
        self.assertIsNotNone(item)
        self.assertEqual(item["text"], "User is a software engineer")
        self.assertEqual(item["importance"], 8.5)
        self.assertEqual(item["metadata"]["source"], "user")

    def test_add_clamps_importance_between_1_and_10(self):
        id_high = self.store.add("Critical constraint", importance=15.0)
        item_high = self.store.get(id_high)
        self.assertEqual(item_high["importance"], 10.0)

        id_low = self.store.add("Low priority fact", importance=-5.0)
        item_low = self.store.get(id_low)
        self.assertEqual(item_low["importance"], 1.0)

    def test_deduplication_updates_importance(self):
        id1 = self.store.add("I like Python", importance=5.0)
        id2 = self.store.add("i like python", importance=9.0)
        self.assertEqual(id1, id2)
        item = self.store.get(id1)
        self.assertEqual(item["importance"], 9.0)

    def test_update_memory(self):
        doc_id = self.store.add("Initial memory text", importance=4.0)
        updated = self.store.update(doc_id, text="Updated memory text", importance=7.0)
        self.assertTrue(updated)

        item = self.store.get(doc_id)
        self.assertEqual(item["text"], "Updated memory text")
        self.assertEqual(item["importance"], 7.0)

    def test_delete_memory(self):
        doc_id = self.store.add("To be deleted", importance=5.0)
        self.assertIsNotNone(self.store.get(doc_id))
        self.store.delete(doc_id)
        self.assertIsNone(self.store.get(doc_id))

    def test_list_memories_pagination(self):
        for i in range(10):
            self.store.add(f"Fact {i}", importance=float(i + 1))

        self.assertEqual(self.store.count(), 10)
        page1 = self.store.list_memories(limit=5, offset=0)
        self.assertEqual(len(page1), 5)
        page2 = self.store.list_memories(limit=5, offset=5)
        self.assertEqual(len(page2), 5)

    def test_search_and_retrieve_top_5(self):
        for i in range(8):
            self.store.add(f"Memory statement number {i}", importance=float((i % 10) + 1))

        results = self.store.search("Memory", top_k=5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn("score", r)
            self.assertIn("importance", r)
            self.assertIn("text", r)

        retrieved = self.store.retrieve("Memory", top_k=5)
        self.assertEqual(len(retrieved), 5)
        self.assertIsInstance(retrieved[0], str)

    def test_empty_store_returns_empty_results(self):
        empty_client = MagicMock()
        empty_coll = MockCollection()
        empty_client.get_or_create_collection.return_value = empty_coll
        empty_store = MemoryStore(client=empty_client, embedder=MockEmbedder())

        self.assertEqual(empty_store.retrieve("anything", top_k=5), [])
        self.assertEqual(empty_store.search("anything", top_k=5), [])
        self.assertEqual(empty_store.list_memories(), [])
        self.assertIsNone(empty_store.get("nonexistent"))
        self.assertFalse(empty_store.update("nonexistent", text="test"))

    def test_search_combines_similarity_and_importance(self):
        # Adding items where distances are identical in MockCollection,
        # so higher importance gives higher score
        self.store.add("Item A", importance=2.0)
        self.store.add("Item B", importance=9.0)

        results = self.store.search("Item", top_k=2)
        self.assertEqual(len(results), 2)
        # Verify scores are present and positive
        self.assertGreater(results[0]["score"], 0)
        self.assertGreater(results[1]["score"], 0)


class TestMemoryExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_heuristic_extraction_name(self):
        memories = self.extractor.extract("Hello, my name is Alice Smith.")
        self.assertTrue(any("Alice Smith" in m["text"] for m in memories))
        name_mem = next(m for m in memories if "Alice Smith" in m["text"])
        self.assertEqual(name_mem["importance"], 9.0)

    def test_heuristic_extraction_allergy(self):
        memories = self.extractor.extract("I am allergic to penicillin.")
        self.assertTrue(any("penicillin" in m["text"] for m in memories))
        allergy_mem = next(m for m in memories if "penicillin" in m["text"])
        self.assertEqual(allergy_mem["importance"], 10.0)

    def test_heuristic_extraction_location_and_role(self):
        memories = self.extractor.extract("I am a backend engineer and I live in Seattle.")
        texts = [m["text"] for m in memories]
        self.assertTrue(any("backend engineer" in t for t in texts))
        self.assertTrue(any("Seattle" in t for t in texts))

    def test_heuristic_extraction_preference(self):
        memories = self.extractor.extract("I prefer dark mode in all editors.")
        self.assertTrue(any("dark mode" in m["text"] for m in memories))

    def test_heuristic_skips_greetings(self):
        self.assertEqual(self.extractor.extract("Hello!"), [])
        self.assertEqual(self.extractor.extract("Thanks!"), [])
        self.assertEqual(self.extractor.extract("ok"), [])

    def test_llm_extraction_parsing(self):
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n[{"text": "User works at OpenAI", "importance": 8.0}]\n```'
                    }
                }
            ]
        }
        extractor = MemoryExtractor(model=mock_model)
        memories = extractor.extract("I work at OpenAI.")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["text"], "User works at OpenAI")
        self.assertEqual(memories[0]["importance"], 8.0)

    def test_empty_llm_result_does_not_fall_back_to_heuristics(self):
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }
        extractor = MemoryExtractor(model=mock_model)

        self.assertEqual(
            extractor.extract("Can you remember that I live in Paris?"),
            [],
        )

    def test_extraction_ignores_assistant_reply_and_follow_up_questions(self):
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }
        extractor = MemoryExtractor(model=mock_model)

        memories = extractor.extract(
            "What's my name?",
            assistant_text=(
                "Your name is Cameron.\n"
                "Follow up: What are the names of my dogs?\n"
                "What kind of stories do you enjoy hearing about my dogs?"
            ),
        )

        self.assertEqual(memories, [])
        messages = mock_model.create_chat_completion.call_args.kwargs["messages"]
        self.assertEqual(messages[1]["content"], "User message:\nWhat's my name?")
        self.assertNotIn("Cameron", messages[1]["content"])
        self.assertIn("only when the user explicitly states or confirms them", messages[0]["content"])

    def test_extract_and_store(self):
        mock_client = MagicMock()
        mock_coll = MockCollection()
        mock_client.get_or_create_collection.return_value = mock_coll
        store = MemoryStore(client=mock_client, embedder=MockEmbedder())

        stored = self.extractor.extract_and_store(
            memory_store=store,
            user_text="My name is Bob and I am allergic to peanuts.",
        )
        self.assertGreaterEqual(len(stored), 1)
        self.assertGreaterEqual(store.count(), 1)


class TestPromptInjection(unittest.TestCase):
    def test_build_messages_injects_top_5_memories(self):
        memories = [f"Memory fact {i}" for i in range(5)]
        session_msgs = []
        result = build_messages(
            session_messages=session_msgs,
            user_text="Hello",
            image_description=None,
            voice_text=None,
            memory_items=memories,
        )
        # Check system prompt with memory context exists
        system_context_msg = next(
            (m for m in result if "Context (may be incomplete/untrusted):" in m["content"]),
            None,
        )
        self.assertIsNotNone(system_context_msg)
        for mem in memories:
            self.assertIn(mem, system_context_msg["content"])


if __name__ == "__main__":
    unittest.main()
