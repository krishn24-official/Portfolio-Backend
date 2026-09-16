import json
import logging
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag import answer_question

# Portfolio Chatbot Backend - updated TOP_K=8
logger = logging.getLogger(__name__)

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "portfolio_content.json"
PROFILE = json.loads(DATA_PATH.read_text())

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Portfolio Chatbot API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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
    owner_email = os.environ.get("OWNER_EMAIL")
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not all([owner_email, gmail_address, gmail_app_password]):
        logger.error("Missing Gmail SMTP configuration in environment variables.")
        raise HTTPException(
            status_code=502,
            detail="Failed to send message. Please try again later.",
        )

    # Sanitize name and email to prevent email header injection
    clean_name = re.sub(r"[\r\n]+", " ", body.name).strip()
    clean_email = re.sub(r"[\r\n]+", "", str(body.email)).strip()

    msg = EmailMessage()
    msg["Subject"] = f"Portfolio contact: message from {clean_name}"
    msg["From"] = gmail_address
    msg["To"] = owner_email
    msg["Reply-To"] = clean_email
    msg.set_content(
        f"You received a new message from your portfolio contact form:\n\n"
        f"Name: {clean_name}\n"
        f"Email: {clean_email}\n\n"
        f"Message:\n{body.message}\n"
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(gmail_address, gmail_app_password)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"SMTP delivery failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to send message. Please try again later.",
        )

    return {"status": "sent"}

