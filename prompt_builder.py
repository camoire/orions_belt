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
        "You are a panel of experts (Physics, CS, Math, Political Philosophy, Law, Economics, Medicine). "
        "Respond with: (1) key take-away per expert, (2) one counterpoint, (3) brief critique, "
        "then a final consensus. Provide confidence percentages. "
        "Keep reasoning succinct; do NOT reveal hidden chain-of-thought. "
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

    # Build a single user message containing whichever modality arrived
    parts = []
    if user_text:
        parts.append(f"User says:\n{user_text}")
    if image_description:
        parts.append(f"Image description:\n{image_description}")
    if voice_text:
        parts.append(f"Voice transcription:\n{voice_text}")

    if parts:
        session_messages.append({"role": "user", "content": "\n\n".join(parts)})

    return session_messages