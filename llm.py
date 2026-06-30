"""
Phase B — LLM client. Talks to Ollama via its OpenAI-compatible endpoint,
so swapping to a hosted API or cloud later is just a base_url + key change.

Prereqs on your machine:
    1. Install Ollama (ollama.com), then pull a model, e.g.:
         ollama pull llama3.1:8b      # if you have an 8GB GPU
         ollama pull llama3.2:3b      # smaller / lower-VRAM fallback
    2. pip install openai
"""

from openai import OpenAI

# Ollama exposes an OpenAI-compatible server locally. api_key is unused locally.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL = "llama3.1:8b"   # MUST match the model you pulled in Ollama


def generate(messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,     # low = factual, sticks to context
    )
    return resp.choices[0].message.content
