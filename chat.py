"""
Phase B — Chat CLI. The full loop: retrieve -> assemble prompt -> Llama -> answer.
This is where the bot actually answers questions.

Run (after Ollama is running and a model is pulled):
    python chat.py
"""

from retrieve import connect, retrieve
from prompt import build_messages
from llm import generate


def answer(db, question: str, history: list[dict]) -> tuple[str, list[dict]]:
    chunks, _filters, mode = retrieve(db, question)
    # List queries enumerate the catalogue, so give the model ALL matches.
    # Semantic queries only need the few most relevant passages.
    top = chunks[:30] if mode == "list" else chunks[:6]
    messages = build_messages(question, top, history)
    reply = generate(messages)
    return reply, top


def main():
    db = connect()
    history: list[dict] = []
    print("Real-estate assistant. Ask a question (Ctrl+C to quit).\n")
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            reply, used = answer(db, q, history)
            print(f"\nBot: {reply}")
            sources = sorted({
                f"listing {c['listing_id']}" if c['listing_id'] else "general info"
                for c in used
            })
            print(f"(sources: {', '.join(sources)})\n")
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": reply})
            history = history[-6:]          # keep the last few turns for follow-ups
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
