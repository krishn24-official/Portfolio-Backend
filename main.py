import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
import resend
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag import answer_question

# Portfolio Chatbot Backend - updated TOP_K=8
logger = logging.getLogger(__name__)

load_dotenv()

resend.api_key = os.environ["RESEND_API_KEY"]

DATA_PATH = Path(__file__).parent / "data" / "portfolio_content.json"
PROFILE = json.loads(DATA_PATH.read_text())

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

configured_origins = [
    origin.strip().rstrip("/")
    for origin in FRONTEND_ORIGIN.split(",")
    if origin.strip()
]
for default_origin in ["http://localhost:5173", "http://127.0.0.1:5173"]:
    if default_origin not in configured_origins:
        configured_origins.append(default_origin)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Portfolio Chatbot API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=300)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class ContactRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    message: str = Field(min_length=1, max_length=1000)


class ContactResponse(BaseModel):
    status: str = "sent"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest):
    result = answer_question(
        question=body.message,
        profile_name=PROFILE["name"],
        profile_role=PROFILE["role"],
        profile_email=PROFILE["email"],
    )
    return result


@app.post("/contact", response_model=ContactResponse)
@limiter.limit("3/hour")
def contact(request: Request, body: ContactRequest):
    # Sanitize name to prevent email header injection
    sanitized_name = re.sub(r"[\r\n]+", " ", body.name).strip()

    try:
        resend.Emails.send({
            "from": "Portfolio Contact <onboarding@resend.dev>",
            "to": [os.environ["OWNER_EMAIL"]],
            "reply_to": body.email,
            "subject": f"Portfolio contact: message from {sanitized_name}",
            "text": f"From: {sanitized_name} <{body.email}>\n\n{body.message}",
        })
    except Exception as e:
        logger.error(f"Resend delivery failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to send message. Please try again later.",
        )

    return {"status": "sent"}

