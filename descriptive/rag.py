from llm_descriptive import call_llm

def explain(context_texts):
    """
    ONLY explains already-selected context.
    NEVER decides frequency or ranking.
    """

    context = "\n".join(context_texts)

    system_prompt = """
You are a customer insights analyst.

Rules:
- Use ONLY the provided context
- Do NOT infer frequency
- Do NOT rank or count
- Do NOT say "most common"
- If context is insufficient, say so
"""

    user_prompt = f"""
Context:
{context}

Task:
Explain the issue represented by these customer reviews.
"""

    return call_llm(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.1
    )
