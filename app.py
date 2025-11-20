import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import requests
import torch
import torch.nn.functional as F

# HF transformers (lightweight)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------
# Load ENV
# ---------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NODE_ALERT_URL = os.getenv("NODE_ALERT_URL")

if not DATABASE_URL:
    raise Exception("❌ DATABASE_URL missing in environment variables")


# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------
app = FastAPI(title="NLP Monitoring Service (Lightweight)")


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    print("✅ Lightweight NLP model loaded")
except Exception as e:
    print("❌ Model load failed:", e)
    raise


# Cardiff NLP mapping
LABELS = ["not-offensive", "offensive", "abusive"]  # model dependent


# ---------------------------------------------------
# DB Pool
# ---------------------------------------------------
try:
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=DATABASE_URL
    )
    print("✅ PostgreSQL pool ready")
except Exception as e:
    print("❌ PostgreSQL connection error:", e)
    raise


def get_conn():
    try:
        return db_pool.getconn()
    except:
        return None


def release_conn(conn):
    if conn:
        db_pool.putconn(conn)


# ---------------------------------------------------
# Request Body
# ---------------------------------------------------
class ChatMessage(BaseModel):
    sender_id: str
    receiver_id: str
    content: Optional[str] = None
    attachment_url: Optional[str] = None


# ---------------------------------------------------
# Abuse Detection
# ---------------------------------------------------
def detect_abuse(message: str):
    message = (message or "").strip()
    if not message:
        return False, 0.0, "neutral"

    try:
        inputs = tokenizer(message, return_tensors="pt").to(device)
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

        # get highest class
        idx = int(torch.argmax(probs))
        label = LABELS[idx]
        score = float(probs[idx])

        # final flag
        abuse = label in ["offensive", "abusive"] and score >= 0.70

        return abuse, round(score * 100, 2), label

    except Exception as e:
        print("⚠ NLP error:", e)
        return False, 0.0, "error"


# ---------------------------------------------------
# Save Message
# ---------------------------------------------------
def save_message(sender, receiver, content, url, abuse, score, label):
    conn = get_conn()
    if not conn:
        return False

    try:
        cur = conn.cursor()
        qry = """
            INSERT INTO messages
            (sender_id, receiver_id, content, attachment_url,
             abuse_detected, abuse_score, sentiment, sentiment_score, is_read)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE)
        """

        cur.execute(qry, (
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


# ---------------------------------------------------
# API Endpoint
# ---------------------------------------------------
@app.post("/api/messages")
def monitor(chat: ChatMessage):

    if not chat.content and not chat.attachment_url:
        raise HTTPException(
            status_code=400,
            detail="Either message or attachment is required."
        )

    abuse_detected, score, label = detect_abuse(chat.content or "")

    saved = save_message(
        chat.sender_id,
        chat.receiver_id,
        chat.content,
        chat.attachment_url,
        abuse_detected,
        score,
        label
    )

    if not saved:
        raise HTTPException(status_code=500, detail="Database save failed.")

    # send alert if abuse
    if abuse_detected and NODE_ALERT_URL:
        try:
            requests.post(
                f"{NODE_ALERT_URL}/api/alerts",
                json={
                    "sender_id": chat.sender_id,
                    "receiver_id": chat.receiver_id,
                    "content": chat.content,
                    "abuse_detected": True,
                    "abuse_score": score,
                    "abuse_label": label
                },
                timeout=3
            )
        except Exception as e:
            print("⚠ Alert send failed:", e)

    return {
        "sender_id": chat.sender_id,
        "receiver_id": chat.receiver_id,
        "content": chat.content,
        "abuse_detected": abuse_detected,
        "abuse_score": score,
        "abuse_label": label
    }


# ---------------------------------------------------
# Render Port Bind
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
