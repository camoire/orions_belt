import json
import requests
import gradio as gr

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "orions-belt"


def chat(message, history):
    """
    Stream responses from Orion's Belt.
    """

    history = history or []

    if not message.strip():
        yield history
        return

    api_messages = list(history)

    api_messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    assistant_message = {
        "role": "assistant",
        "content": ""
    }

    history.append(assistant_message)

    try:

        with requests.post(
            API_URL,
            json={
                "model": MODEL_NAME,
                "messages": api_messages,
                "stream": True
            },
            stream=True,
            timeout=(10, 300)
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):

                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                payload = line[6:].strip()

                if payload == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)

                    delta = (
                        chunk["choices"][0]
                        .get("delta", {})
                        .get("content", "")
                    )

                    if delta:
                        assistant_message["content"] += delta
                        yield history

                except json.JSONDecodeError:
                    continue

                except Exception as e:
                    assistant_message["content"] += (
                        f"\n\n⚠️ Stream parsing error: {e}"
                    )
                    yield history

    except requests.exceptions.ConnectionError:

        assistant_message["content"] = (
            "❌ Unable to connect to Orion's Belt."
        )

        yield history

    except requests.exceptions.Timeout:

        assistant_message["content"] = (
            "❌ Request timed out."
        )

        yield history

    except requests.exceptions.HTTPError as e:

        assistant_message["content"] = (
            f"❌ HTTP Error: {e}"
        )

        yield history

    except Exception as e:

        assistant_message["content"] = (
            f"❌ Unexpected Error:\n\n{e}"
        )

        yield history


def clear_chat():
    return []


with gr.Blocks() as demo:

    gr.Markdown("# 🚀 Orion's Belt")
    gr.Markdown("Local Qwen Assistant")

    chatbot = gr.Chatbot(
        height=700
    )

    msg = gr.Textbox(
        placeholder="Send a message...",
        show_label=False
    )

    clear = gr.Button("Clear Chat")

    msg.submit(
        chat,
        inputs=[
            msg,
            chatbot
        ],
        outputs=chatbot
    )

    clear.click(
        clear_chat,
        outputs=chatbot
    )


if __name__ == "__main__":

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )