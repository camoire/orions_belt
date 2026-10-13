import asyncio
import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pydantic import ValidationError


def import_app_with_mocked_runtime():
    class MockMemoryStore:
        def __init__(self, *args, **kwargs):
            self.retrieve = MagicMock(return_value=[])

    class MockMemoryExtractor:
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model")

        def extract_and_store(self, **kwargs):
            return []

    class MockModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MagicMock()

    module_stubs = {
        "whisper": types.SimpleNamespace(load_model=MagicMock(return_value=MagicMock())),
        "llama_cpp": types.SimpleNamespace(Llama=MagicMock(return_value=MagicMock())),
        "PIL": types.SimpleNamespace(Image=MagicMock()),
        "transformers": types.SimpleNamespace(
            BlipForConditionalGeneration=MockModel,
            BlipProcessor=MockModel,
        ),
        "memory": types.SimpleNamespace(
            MemoryStore=MockMemoryStore,
            MemoryExtractor=MockMemoryExtractor,
        ),
        "behavior_logger": types.SimpleNamespace(log_interaction=MagicMock()),
        "prompt_builder": types.SimpleNamespace(build_messages=MagicMock()),
        "conversation_store": types.SimpleNamespace(conversations=MagicMock()),
        "train_lora": types.SimpleNamespace(train_lora_adapter=MagicMock()),
    }
    with patch.dict(sys.modules, module_stubs):
        return importlib.import_module("app")


app_module = import_app_with_mocked_runtime()


class TestOpenAIAPI(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        app_module.model = self.model
        app_module.memory.retrieve.reset_mock()
        app_module.memory.retrieve.return_value = []
        self.thread_patcher = patch.object(app_module.threading, "Thread")
        self.thread_patcher.start()
        self.addCleanup(self.thread_patcher.stop)

    def test_openai_routes_are_registered(self):
        routes = {
            (route.path, method)
            for route in app_module.app.routes
            for method in (getattr(route, "methods", None) or [])
        }
        self.assertIn(("/v1/models", "GET"), routes)
        self.assertIn(("/v1/models/{model_id}", "GET"), routes)
        self.assertIn(("/v1/chat/completions", "POST"), routes)
        self.assertIn(("/v1/completions", "POST"), routes)

    def test_list_models_uses_configured_model_id(self):
        response = asyncio.run(app_module.openai_list_models())
        self.assertEqual(response["object"], "list")
        self.assertEqual(response["data"][0]["id"], app_module.OPENAI_MODEL_ID)
        self.assertEqual(response["data"][0]["object"], "model")

    def test_retrieve_model_returns_model_or_not_found(self):
        response = asyncio.run(app_module.openai_retrieve_model(app_module.OPENAI_MODEL_ID))
        self.assertEqual(response["id"], app_module.OPENAI_MODEL_ID)
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_retrieve_model("not-a-model"))
        self.assertEqual(error.exception.status_code, 404)

    def test_chat_completion_forwards_supported_options_and_usage(self):
        self.model.create_chat_completion.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "  hello  "},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "length",
                },
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            max_completion_tokens=32,
            temperature=0.25,
            top_p=0.8,
            presence_penalty=0.5,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice="auto",
            seed=123,
        )

        response = asyncio.run(app_module.openai_chat_completions(request))

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "  hello  ")
        self.assertEqual(response["choices"][1]["finish_reason"], "length")
        self.assertEqual(response["usage"]["total_tokens"], 6)
        kwargs = self.model.create_chat_completion.call_args.kwargs
        self.assertEqual(kwargs["messages"][-1]["content"], "hello")
        self.assertEqual(kwargs["max_tokens"], 32)
        self.assertEqual(kwargs["temperature"], 0.25)
        self.assertEqual(kwargs["tools"], request.tools)
        self.assertEqual(kwargs["seed"], 123)

    def test_chat_completion_rejects_unknown_model_and_image_content(self):
        wrong_model = app_module.ChatCompletionRequest(
            model="unknown", messages=[{"role": "user", "content": "hello"}]
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_chat_completions(wrong_model))
        self.assertEqual(error.exception.status_code, 404)

        image_message = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}],
            }],
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_chat_completions(image_message))
        self.assertEqual(error.exception.status_code, 400)

    def test_chat_completion_validates_spec_ranges(self):
        with self.assertRaises(ValidationError):
            app_module.ChatCompletionRequest(
                model=app_module.OPENAI_MODEL_ID,
                messages=[{"role": "user", "content": "hello"}],
                temperature=3,
            )

    def test_chat_completion_stream_emits_deltas_and_done(self):
        self.model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        response = asyncio.run(app_module.openai_chat_completions(request))
        chunks = list(response.body_iterator)
        payload = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks)
        self.assertIn(b'"object": "chat.completion.chunk"', payload)
        self.assertIn(b'"content": "hello"', payload)
        self.assertIn(b"data: [DONE]", payload)

    def test_legacy_completion_returns_text_shape_and_usage(self):
        self.model.create_chat_completion.return_value = {
            "choices": [
                {"message": {"content": " world"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            prompt="hello",
            echo=True,
            max_tokens=8,
        )

        response = asyncio.run(app_module.openai_completions(request))

        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["choices"][0]["text"], "hello world")
        self.assertEqual(response["usage"]["total_tokens"], 3)
        self.assertEqual(self.model.create_chat_completion.call_args.kwargs["max_tokens"], 8)

    def test_legacy_completion_handles_multiple_prompts(self):
        self.model.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": "first"}}], "usage": {}},
            {"choices": [{"message": {"content": "second"}}], "usage": {}},
        ]
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID, prompt=["one", "two"]
        )

        response = asyncio.run(app_module.openai_completions(request))

        self.assertEqual([choice["text"] for choice in response["choices"]], ["first", "second"])
        self.assertEqual([call.kwargs["messages"][0]["content"] for call in self.model.create_chat_completion.call_args_list], ["one", "two"])

    def test_legacy_completion_stream_and_unsupported_options(self):
        self.model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "token"}}]},
        ])
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID, prompt="hello", stream=True
        )

        payload = "".join(app_module.completion_stream(request, ["hello"], "cmpl-1", 1))
        self.assertIn('"text": "token"', payload)
        self.assertIn("data: [DONE]", payload)

        unsupported = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID, prompt="hello", best_of=2
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_completions(unsupported))
        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
