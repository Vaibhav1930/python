import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# NLP
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PostgreSQL
import psycopg2
from psycopg2.pool import SimpleConnectionPool


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NODE_ALERT_URL = os.getenv("NODE_ALERT_URL")

if not DATABASE_URL:
    raise Exception("❌ DATABASE_URL is missing from environment variables")


# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------
app = FastAPI(title="ClairKey NLP Monitoring Service")


# ------------------------------------------------------------
# Load Lightweight Abuse Model
# ------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    print("✅ NLP Model loaded")
except Exception as e:
    print("❌ Failed to load NLP model:", e)
    raise

LABELS = ["not-offensive", "offensive", "abusive"]


# ------------------------------------------------------------
# PostgreSQL Connection Pool
# ------------------------------------------------------------
try:
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=DATABASE_URL
    )
    print("✅ PostgreSQL connection pool created")
except Exception as e:
    print("❌ Database connection error:", e)
    raise


def get_conn():
    try:
        return db_pool.getconn()
    except:
        return None


def release_conn(conn):
    if conn:
        db_pool.putconn(conn)


# ------------------------------------------------------------
# Request Model
# ------------------------------------------------------------
class ChatMessage(BaseModel):
    sender_id: str
    receiver_id: str
    content: Optional[str] = None
    attachment_url: Optional[str] = None


# ------------------------------------------------------------
# Abuse Detector
# ------------------------------------------------------------
def detect_abuse(text: str):
    text = (text or "").strip()

    if not text:
        return False, 0.0, "neutral"

    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

        idx = int(torch.argmax(probs))
        label = LABELS[idx]
        score = float(probs[idx])

        abuse = label in ["offensive", "abusive"] and score >= 0.70

        return abuse, round(score * 100, 2), label

    except Exception as e:
        print("⚠️ NLP error:", e)
        return False, 0.0, "error"


# ------------------------------------------------------------
# Save Message in DB
# ------------------------------------------------------------
def save_message(sender, receiver, content, url, abuse, score, label):
    conn = get_conn()
    if not conn:
        return False

    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO messages (
                sender_id, receiver_id, content, attachment_url,
                abuse_detected, abuse_score, sentiment, sentiment_score, is_read
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE)
        """

        cur.execute(sql, (
            sender, receiver, content, url,
            abuse, score, label, score
        ))

        conn.commit()
        cur.close()
        return True

    except Exception as e:
        print("❌ DB Insert Error:", e)
        return False

    finally:
        release_conn(conn)


# ------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------
@app.post("/api/messages")
def monitor_message(chat: ChatMessage):

    if not chat.content and not chat.attachment_url:
        raise HTTPException(
            status_code=400,
            detail="Message or attachment is required."
        )

    abuse, score, label = detect_abuse(chat.content or "")

    saved = save_message(
        chat.sender_id,
        chat.receiver_id,
        chat.content,
        chat.attachment_url,
        abuse,
        score,
        label,
    )

    if not saved:
        raise HTTPException(status_code=500, detail="Database error")

    # Send alert to Node server
    if abuse and NODE_ALERT_URL:
        try:
            requests.post(
                f"{NODE_ALERT_URL}/api/alerts",
                json={
                    "sender_id": chat.sender_id,
                    "receiver_id": chat.receiver_id,
                    "content": chat.content,
                    "abuse_detected": True,
                    "abuse_score": score,
                    "abuse_label": label,
                },
                timeout=3
            )
        except Exception as e:
            print("⚠️ Alert forwarding failed:", e)

    return {
        "sender_id": chat.sender_id,
        "receiver_id": chat.receiver_id,
        "content": chat.content,
        "abuse_detected": abuse,
        "abuse_score": score,
        "abuse_label": label
    }
