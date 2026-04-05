import json

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
        "Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches. "
        "Break down the solution into clear steps within <step> tags. Start with a 20-step budget, "
        "requesting more for complex problems if needed. Use <count> tags after each step to show the remaining budget. "
        "Stop when reaching 0. Continuously adjust your reasoning based on intermediate results and reflections, "
        "adapting your strategy as you progress. Regularly evaluate progress using <reflection> tags. Be critical "
        "and honest about your reasoning process. Assign a quality score between 0.0 and 1.0 using <reward> tags "
        "after each reflection. Use this to guide your approach: 0.8+: Continue current approach 0.5-0.7: Consider "
        "minor adjustments Below 0.5: Seriously consider backtracking and trying a different approach If unsure or "
        "if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> "
        "tags. For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed "
        "proofs. Explore multiple solutions individually if possible, comparing approaches in reflections. Use thoughts "
        "as a scratchpad, writing out all calculations and reasoning explicitly. Synthesize the final answer within "
        "<answer> tags, providing a clear, concise summary. Conclude with a final reflection on the overall solution, "
        "discussing effectiveness, challenges, and solutions. Assign a final reward score."
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

    print(json.dumps(session_messages, indent=2))

    return session_messages