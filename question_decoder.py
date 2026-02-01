import json
from llm import call_llm

def decompose_question(llm, question: str):
    prompt = f"""
You are a query decomposition and routing engine.

Your job:
1. Break the question into atomic sub-questions
2. Identify analytical needs
3. Decide the SINGLE best execution route

IMPORTANT RULES:
- Do NOT invent filters or segments
- Only include a segment IF the question explicitly mentions
  departments, user groups, categories, or comparisons
- If the question is global, segment MUST be null

Allowed routes:
- SQL
- CLUSTER_COUNT_SQL_RAG
- RAG_ONLY

Routing rules:
- SQL → structured, numeric, column-based
- CLUSTER_COUNT_SQL_RAG → any semantic grouping, issues, complaints, frequency,major,minor,most,least or explanation over text
- RAG_ONLY → conceptual explanation without using database data

Return STRICT JSON in this format:

{{
  "sub_questions": [],
  "needs": {{
    "aggregation": false,
    "explanation": false,
    "comparison": false,
    "trend": false,
    "clustering": false
  }},
  "filters": {{
    "time": null,
    "segment": null
  }},
  "route": ""
}}

DEFINITIONS:
- segment = grouping or filtering by an explicit category
  (e.g., IT vs Finance, premium users, region-based groups)
- If no explicit grouping is requested, segment MUST be null

OUTPUT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown

Question:
{question}
"""


    response_text = llm(
        prompt=prompt,
        system_prompt="You output only valid JSON.",
        temperature=0.0
    )

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON:\n{response_text}"
        ) from e


result = decompose_question(
    llm=call_llm,
    question="comment on the issue"
)

print(result)