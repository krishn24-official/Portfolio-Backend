# Portfolio chatbot backend

FastAPI service that answers visitor questions about Krishna using RAG:
local embeddings (sentence-transformers) → Chroma vector search → Groq's
free Llama endpoint for generation.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Get a free Groq API key at https://console.groq.com/keys and paste it into
`.env` as `GROQ_API_KEY`.

## 2. Build the vector index

Run this once, and again any time `data/portfolio_content.json` changes:

```bash
python ingest.py
```

This creates a `chroma_db/` folder on disk — that's your vector database,
no server required.

`data/portfolio_content.json` is a copy of what's in the React app's
`src/data/portfolioData.js`. Keep the two in sync by hand for now — if the
bio, skills, or projects change on the site, update this JSON and re-run
`ingest.py`, or write a small build step that generates the JSON from the
JS file automatically.

## 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What backend frameworks does Krishna know?"}'
```

You should get back `{"answer": "...", "sources": [...]}`.

## 4. Deploying

Render (matches the rest of your stack): create a new Web Service pointing
at this folder, build command `pip install -r requirements.txt && python
ingest.py`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`,
and set `GROQ_API_KEY` + `FRONTEND_ORIGIN` (your deployed portfolio URL) as
environment variables.

Note: Render's free tier filesystem is ephemeral on redeploy, so the build
command re-runs `ingest.py` every deploy to rebuild `chroma_db/` — that's
fine here since indexing takes seconds, not something you'd want for a
large document store.

## Notes on the guardrails

- The system prompt in `rag.py` restricts answers to retrieved context only,
  and tells the model to say "I don't have that information" rather than
  guess — test this with a few off-topic questions before you ship it.
- `/chat` is rate-limited to 10 requests/minute per IP (`slowapi`) so a free
  Groq key can't be exhausted by one visitor hammering the endpoint.
- Messages are capped at 300 characters — enough for a real question,
  short enough to keep prompt-injection room limited and token costs low.

## Next step

The frontend chat widget (React component matching your portfolio's
"blueprint" theme) calls `POST /chat` with `{"message": "..."}` and renders
`answer`. Ask for that whenever you're ready to build it.
