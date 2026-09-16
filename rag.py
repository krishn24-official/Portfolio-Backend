"""RAG core: retrieve relevant chunks from Chroma, then ask Groq's free
Llama endpoint to answer using only those chunks."""

import os
from pathlib import Path

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "portfolio"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

TOP_K = 4
MAX_ANSWER_TOKENS = 300

SYSTEM_PROMPT_TEMPLATE = """You are the portfolio assistant for {name}, a {role}.
Answer ONLY using the CONTEXT below. The context is the complete set of facts
you are allowed to use about {name} — treat anything not stated there as
unknown, even if it sounds plausible.

Rules:
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

_model = None
_client = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
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
    """Embed the question and return the top_k most relevant chunks."""
    collection = _get_collection()
    query_embedding = _get_model().encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

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
