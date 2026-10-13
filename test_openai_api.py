import asyncio
import importlib
import sys
import threading
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
        self.extraction_patcher = patch.object(app_module, "extract_memories_background")
        self.extraction_patcher.start()
        self.addCleanup(self.extraction_patcher.stop)

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

    def test_chat_completion_does_not_block_the_event_loop(self):
        started = threading.Event()
        release = threading.Event()
        original_lock = app_module.generation_lock
        app_module.generation_lock = threading.Lock()
        self.addCleanup(setattr, app_module, "generation_lock", original_lock)

        def slow_completion(**_):
            started.set()
            release.wait(timeout=1)
            return {"choices": [{"message": {"content": "hello"}}], "usage": {}}

        self.model.create_chat_completion.side_effect = slow_completion
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
        )

        async def check_responsiveness():
            completion = asyncio.create_task(app_module.openai_chat_completions(request))
            try:
                worker_started = await asyncio.wait_for(
                    asyncio.to_thread(started.wait), timeout=1
                )
                self.assertTrue(worker_started)
                self.assertFalse(completion.done())
            finally:
                release.set()
            await completion

        asyncio.run(check_responsiveness())

    def test_legacy_generation_serializes_turns_per_session(self):
        session_id = "serialized-turn-test"
        app_module.conversations.clear(session_id)
        self.addCleanup(app_module.conversations.clear, session_id)
        started = threading.Event()
        release = threading.Event()
        snapshots = []

        def generate_from_messages(messages):
            snapshots.append(list(messages))
            if len(snapshots) == 1:
                started.set()
                release.wait(timeout=2)
                return "first reply"
            return "second reply"

        async def run_concurrent_turns():
            with patch.object(
                app_module, "generate_from_messages", side_effect=generate_from_messages
            ):
                first = asyncio.create_task(
                    app_module.generate("first", session_id=session_id)
                )
                second = asyncio.create_task(
                    app_module.generate("second", session_id=session_id)
                )
                try:
                    worker_started = await asyncio.wait_for(
                        asyncio.to_thread(started.wait), timeout=1
                    )
                    self.assertTrue(worker_started)
                    self.assertEqual(len(snapshots), 1)
                finally:
                    release.set()
                    await asyncio.gather(first, second)

        asyncio.run(run_concurrent_turns())

        self.assertEqual(
            [message["content"] for message in snapshots[0] if message["role"] != "system"],
            ["User says:\nfirst"],
        )
        self.assertEqual(
            [
                (message["role"], message["content"])
                for message in snapshots[1]
                if message["role"] != "system"
            ],
            [
                ("user", "User says:\nfirst"),
                ("assistant", "first reply"),
                ("user", "User says:\nsecond"),
            ],
        )

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

    def test_chat_completion_repeats_choices_without_backend_n(self):
        self.model.create_chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
            n=2,
        )

        response = asyncio.run(app_module.openai_chat_completions(request))

        self.assertEqual([choice["index"] for choice in response["choices"]], [0, 1])
        self.assertEqual(self.model.create_chat_completion.call_count, 2)
        self.assertTrue(all("n" not in call.kwargs for call in self.model.create_chat_completion.call_args_list))

    def test_chat_completion_requires_content_and_rejects_unsupported_fields(self):
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[
                {"role": "system"},
                {"role": "user", "content": "hello"},
            ],
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_chat_completions(request))
        self.assertEqual(error.exception.status_code, 400)

        for options in ({"parallel_tool_calls": False}, {"stream_options": {"include_usage": True}}):
            request = app_module.ChatCompletionRequest(
                model=app_module.OPENAI_MODEL_ID,
                messages=[{"role": "user", "content": "hello"}],
                **options,
            )
            with self.assertRaises(HTTPException) as error:
                asyncio.run(app_module.openai_chat_completions(request))
            self.assertEqual(error.exception.status_code, 400)

    def test_chat_completion_accepts_legacy_function_results(self):
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[
                {"role": "assistant", "function_call": {"name": "lookup", "arguments": "{}"}},
                {"role": "function", "name": "lookup", "content": "result"},
            ],
        )
        app_module.prepare_openai_messages(request.messages)

    def test_chat_completion_prompt_hides_memory_action_payloads(self):
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "I have two dogs."},
            ],
        )

        messages, _ = app_module.prepare_openai_messages(request.messages)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Be helpful.", messages[0]["content"])
        self.assertIn("do not expose memory actions", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "I have two dogs.")

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
        async def collect_chunks():
            iterator = response.body_iterator
            first = await iterator.__anext__()
            first_text = first.decode() if isinstance(first, bytes) else first
            self.assertIn('"delta": {"role": "assistant"}', first_text)
            self.model.create_chat_completion.assert_not_called()
            return [first, *[chunk async for chunk in iterator]]

        chunks = asyncio.run(collect_chunks())
        payload = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks)
        self.assertIn('"object": "chat.completion.chunk"', payload)
        self.assertIn('"content": "hello"', payload)
        self.assertIn("data: [DONE]", payload)

    def test_generate_stream_sends_status_before_incremental_text(self):
        session_id = "generate-stream-test"
        session_messages = []
        self.model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": " hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world "}, "finish_reason": "stop"}]},
        ])

        with patch.object(app_module.conversations, "get", return_value=session_messages):
            response = asyncio.run(app_module.generate("hi", session_id, stream=True))

            async def collect_stream():
                iterator = response.body_iterator
                status = await iterator.__anext__()
                status_text = status.decode() if isinstance(status, bytes) else status
                self.assertIn("event: status", status_text)
                self.assertIn('"status": "generating"', status_text)
                self.model.create_chat_completion.assert_not_called()
                return [status, *[chunk async for chunk in iterator]]

            chunks = asyncio.run(collect_stream())
        payload = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks)

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn("event: delta", payload)
        self.assertIn('"generated_text": " hello"', payload)
        self.assertIn('"generated_text": " world "', payload)
        self.assertIn("event: done", payload)
        self.assertTrue(self.model.create_chat_completion.call_args.kwargs["stream"])
        self.assertEqual(session_messages[-1]["content"], " hello world ")
        self.assertIn('"generated_text": " hello world "', payload)

    def test_generate_stream_does_not_commit_messages_after_inference_error(self):
        session_id = "generate-stream-error-test"
        session_messages = [{"role": "system", "content": "existing"}]

        def build_pending_messages(session_messages, **_):
            session_messages.append({"role": "user", "content": "hi"})

        def failing_stream(**_):
            def chunks():
                yield {"choices": [{"delta": {"content": "partial"}}]}
                raise RuntimeError("inference failed")

            return chunks()

        self.model.create_chat_completion.side_effect = failing_stream

        with (
            patch.object(app_module.conversations, "get", return_value=session_messages),
            patch.object(app_module, "build_messages", side_effect=build_pending_messages),
        ):
            response = asyncio.run(app_module.generate("hi", session_id, stream=True))

            async def collect_stream():
                return [chunk async for chunk in response.body_iterator]

            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                asyncio.run(collect_stream())

        self.assertEqual(session_messages, [{"role": "system", "content": "existing"}])

    def test_generate_stream_does_not_commit_messages_when_client_disconnects(self):
        session_id = "generate-stream-disconnect-test"
        session_messages = [{"role": "system", "content": "existing"}]
        self.model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "partial"}}]},
        ])

        def build_pending_messages(session_messages, **_):
            session_messages.append({"role": "user", "content": "hi"})

        with (
            patch.object(app_module.conversations, "get", return_value=session_messages),
            patch.object(app_module, "build_messages", side_effect=build_pending_messages),
        ):
            response = asyncio.run(app_module.generate("hi", session_id, stream=True))

            async def receive_partial_then_disconnect():
                iterator = response.body_iterator
                await iterator.__anext__()
                partial = await iterator.__anext__()
                await iterator.aclose()
                return partial

            asyncio.run(receive_partial_then_disconnect())

        self.assertEqual(session_messages, [{"role": "system", "content": "existing"}])

    def test_generate_keeps_non_streaming_json_response_by_default(self):
        session_id = "generate-json-test"
        session_messages = []
        self.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hello"}}],
        }

        with patch.object(app_module.conversations, "get", return_value=session_messages):
            response = asyncio.run(app_module.generate("hi", session_id, stream=False))

        self.assertEqual(response, {"session_id": session_id, "generated_text": "hello"})
        self.assertEqual(session_messages[-1]["content"], "hello")
        self.assertNotIn("stream", self.model.create_chat_completion.call_args.kwargs)

    def test_chat_completion_stream_repeats_choices_with_distinct_indexes(self):
        self.model.create_chat_completion.side_effect = [
            iter([{"choices": [{"delta": {"content": "first"}, "finish_reason": "stop"}]}]),
            iter([{"choices": [{"delta": {"content": "second"}, "finish_reason": "stop"}]}]),
        ]
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
            n=2,
            stream=True,
        )

        payload = "".join(app_module.openai_stream(
            [{"role": "user", "content": "hello"}],
            "hello",
            "chatcmpl-1",
            1,
            app_module.OPENAI_MODEL_ID,
            request,
        ))

        self.assertIn('"index": 0', payload)
        self.assertIn('"index": 1', payload)
        self.assertTrue(all(
            "n" not in call.kwargs for call in self.model.create_chat_completion.call_args_list
        ))

    def test_legacy_completion_returns_text_shape_and_usage(self):
        self.model.create_completion.return_value = {
            "choices": [
                {"text": "hello world", "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            prompt="hello",
            echo=False,
            max_tokens=8,
        )

        response = asyncio.run(app_module.openai_completions(request))

        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["choices"][0]["text"], "hello world")
        self.assertEqual(response["usage"]["total_tokens"], 3)
        kwargs = self.model.create_completion.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 8)
        self.assertFalse(kwargs["echo"])

    def test_legacy_completion_handles_multiple_prompts(self):
        self.model.create_completion.side_effect = [
            {"choices": [{"text": "first"}], "usage": {}},
            {"choices": [{"text": "second"}], "usage": {}},
        ]
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID, prompt=["one", "two"]
        )

        response = asyncio.run(app_module.openai_completions(request))

        self.assertEqual([choice["text"] for choice in response["choices"]], ["first", "second"])
        self.assertEqual(
            [call.kwargs["prompt"] for call in self.model.create_completion.call_args_list],
            ["one", "two"],
        )

    def test_legacy_completion_repeats_choices_without_unsupported_arguments(self):
        self.model.create_completion.return_value = {
            "choices": [{"text": "answer"}],
            "usage": {},
        }
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            prompt="hello",
            n=2,
        )

        response = asyncio.run(app_module.openai_completions(request))

        self.assertEqual([choice["index"] for choice in response["choices"]], [0, 1])
        self.assertEqual(self.model.create_completion.call_count, 2)
        self.assertTrue(all(
            "n" not in call.kwargs and "best_of" not in call.kwargs
            for call in self.model.create_completion.call_args_list
        ))

    def test_legacy_completion_stream_and_token_prompts(self):
        self.model.create_completion.return_value = iter([
            {"choices": [{"text": "token", "index": 0, "finish_reason": None}]},
            {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]},
        ])
        request = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID, prompt="hello", stream=True
        )

        payload = "".join(app_module.completion_stream(request, ["hello"], "cmpl-1", 1))
        self.assertIn('"text": "token"', payload)
        self.assertIn("data: [DONE]", payload)

        token_prompt = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID, prompt=[12, 15, 22]
        )
        self.model.create_completion.return_value = {
            "choices": [{"text": "decoded"}],
            "usage": {},
        }
        asyncio.run(app_module.openai_completions(token_prompt))
        self.assertEqual(self.model.create_completion.call_args.kwargs["prompt"], [12, 15, 22])

    def test_legacy_completion_rejects_unknown_model_and_invalid_best_of(self):
        unknown = app_module.CompletionRequest(model="unknown", prompt="hello")
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_completions(unknown))
        self.assertEqual(error.exception.status_code, 404)

        invalid_best_of = app_module.CompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            prompt="hello",
            n=2,
            best_of=1,
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(app_module.openai_completions(invalid_best_of))
        self.assertEqual(error.exception.status_code, 400)

    def test_request_work_budget_limits_prompt_batches_and_compute(self):
        with self.assertRaises(HTTPException):
            app_module.enforce_request_work([1] * (app_module.MAX_REQUEST_PROMPTS + 1), 1, 1)
        with self.assertRaises(HTTPException):
            app_module.enforce_request_work([1000], 10, 1000)

    def test_chat_completion_rejects_excessive_work_before_memory_retrieval(self):
        request = app_module.ChatCompletionRequest(
            model=app_module.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "x" * 140_000}],
        )

        with self.assertRaises(HTTPException):
            asyncio.run(app_module.openai_chat_completions(request))

        app_module.memory.retrieve.assert_not_called()

    def test_openai_authentication_is_optional_or_requires_bearer_key(self):
        class Request:
            url = types.SimpleNamespace(path="/v1/models")
            headers = {}

        async def call_next(_request):
            return "allowed"

        with patch.object(app_module, "OPENAI_API_KEY", "configured-key"):
            denied = asyncio.run(app_module.authenticate_openai_requests(Request(), call_next))
            self.assertEqual(denied.status_code, 401)
            Request.headers = {"authorization": "Bearer " + "configured" + "-key"}
            self.assertEqual(
                asyncio.run(app_module.authenticate_openai_requests(Request(), call_next)),
                "allowed",
            )


if __name__ == "__main__":
    unittest.main()
