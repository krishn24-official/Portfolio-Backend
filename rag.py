"""RAG core: retrieve relevant chunks from Chroma, then ask Groq's free
Llama endpoint to answer using only those chunks."""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "portfolio"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

TOP_K = 8
MAX_ANSWER_TOKENS = 300

SYSTEM_PROMPT_TEMPLATE = """You are the AI Portfolio Assistant for {name}, a {role}.
Answer ONLY using the CONTEXT below. The context is the complete set of facts
you are allowed to use about {name} — treat anything not stated there as
unknown, even if it sounds plausible.

Rules:
- If the question asks about multiple items (e.g. 'projects', 'skills', plural nouns),
  mention ALL relevant items found in the CONTEXT, not just one — don't stop after
  the first match.
- If the answer isn't in the context, say you don't have that information and
  suggest the visitor email {name} directly at {email}.
- Never invent projects, skills, dates, employers, or contact details.
- Keep answers short (2-4 sentences) and friendly, written in third person
  about {name}.
- If asked something unrelated to {name}'s background (general knowledge,
  other people, coding help unrelated to their work), politely decline and
  redirect to what you can help with.

CONTEXT:
{context}
"""

_client = None
_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        try:
            _collection.count()
            return _collection
        except Exception:
            _collection = None

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    _collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    return _collection


def _get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and put it in your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top_k most relevant chunks using Chroma's embedding function."""
    collection = _get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "title": meta["title"], "section": meta["section"]})
    return chunks


def answer_question(question: str, profile_name: str, profile_role: str, profile_email: str) -> dict:
    """Full RAG turn: retrieve context, then generate a grounded answer."""
    chunks = retrieve(question)
    context = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        name=profile_name, role=profile_role, email=profile_email, context=context
    )

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    sources = sorted({c["title"] for c in chunks})
    return {"answer": answer, "sources": sources}
