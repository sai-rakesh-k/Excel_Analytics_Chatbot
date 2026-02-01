import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3:8b"

def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    stream: bool = False
) -> str:
    """
    Generic LLM caller for Ollama.
    Returns ONLY the model text response.
    """

    payload = {
        "model": model,
        "stream": stream,
        "messages": [],
        "options": {
            "temperature": temperature
        }
    }

    if system_prompt:
        payload["messages"].append({
            "role": "system",
            "content": system_prompt
        })

    payload["messages"].append({
        "role": "user",
        "content": prompt
    })

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()

    if "message" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    return data["message"]["content"].strip()
