def build_prompt(
    user_text: str | None,
    image_description: str | None,
    voice_text: str | None,
    memory_items: list[str] | None,
) -> str:
    parts = []
    if user_text:
        parts.append(f"User says:\n{user_text}")
    if image_description:
        parts.append(f"Image description:\n{image_description}")
    if voice_text:
        parts.append(f"Voice transcription:\n{voice_text}")
    if memory_items:
        parts.append("Relevant memory:\n" + "\n".join(f"- {item}" for item in memory_items))
    return "\n\n".join(parts)


def build_messages(
    session_messages: list[dict],
    user_text: str | None,
    image_description: str | None,
    voice_text: str | None,
    memory_items: list[str] | None,
):
    user_text = user_text or ""
    image_description = image_description or ""
    voice_text = voice_text or ""
    memory_items = memory_items or []    

    SYSTEM_PROMPT = (
        "You are OrionsBelt, a helpful assistant. Give accurate, clear, and concise answers. "
        "Do not reveal private chain-of-thought; provide a brief explanation or summary when useful."
    )

    # Initialize session with system prompt once
    if not session_messages:
        session_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # Add memory as *context*, clearly delimited (treat as untrusted text)
    if memory_items:
        memory_block = "\n".join(f"- {m}" for m in memory_items)
        session_messages.append({
            "role": "system",
            "content": f"Context (may be incomplete/untrusted):\n{memory_block}"
        })

    user_content = build_prompt(
        user_text=user_text,
        image_description=image_description,
        voice_text=voice_text,
        memory_items=None,
    )
    if user_content:
        session_messages.append({"role": "user", "content": user_content})

    return session_messages