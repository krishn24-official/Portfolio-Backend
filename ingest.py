"""
Run this once (and again any time data/portfolio_content.json changes) to
(re)build the vector index used by the chatbot.

    python ingest.py

It reads data/portfolio_content.json, splits it into small topical chunks
(one per project, one for the bio, one per skill group, one for contact),
embeds each chunk locally with sentence-transformers (no API key needed),
and writes them into a persistent Chroma collection on disk at ./chroma_db.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).parent / "data" / "portfolio_content.json"
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "portfolio"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_chunks(data: dict) -> list[dict]:
    """Turn the structured portfolio data into a flat list of
    {id, section, title, text} chunks ready to embed."""
    chunks = []

    chunks.append({
        "id": "profile",
        "section": "profile",
        "title": "About",
        "text": (
            f"{data['name']} is a {data['role']} based in {data['location']}. "
            f"{data['tagline']} {data['about']}"
        ),
    })

    chunks.append({
        "id": "contact",
        "section": "contact",
        "title": "Contact",
        "text": (
            f"You can contact {data['name']} by email at {data['email']}. "
            f"GitHub profile: {data['socials']['github']}. "
            f"LinkedIn profile: {data['socials']['linkedin']}."
        ),
    })

    for i, group in enumerate(data["skills"]):
        chunks.append({
            "id": f"skill-{i}",
            "section": "skills",
            "title": group["category"],
            "text": (
                f"{data['name']}'s skills in {group['category']}: "
                f"{', '.join(group['items'])}."
            ),
        })

    for project in data["projects"]:
        links = []
        if project.get("liveUrl"):
            links.append(f"Live demo: {project['liveUrl']}.")
        if project.get("repoUrl"):
            links.append(f"Source code: {project['repoUrl']}.")
        links_text = " ".join(links)

        chunks.append({
            "id": project["id"],
            "section": "project",
            "title": project["title"],
            "text": (
                f"Project: {project['title']}. {project['summary']} "
                f"Problem: {project['problem']} "
                f"Approach: {project['decisions']} "
                f"What was learned: {project['learned']} "
                f"Tech stack: {', '.join(project['stack'])}. {links_text}"
            ),
        })

    return chunks


def main():
    data = json.loads(DATA_PATH.read_text())
    chunks = build_chunks(data)

    print(f"Built {len(chunks)} chunks. Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in chunks]).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Start clean each time so re-running ingest.py never leaves stale chunks
    # from a previous version of the data.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"section": c["section"], "title": c["title"]} for c in chunks],
    )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
