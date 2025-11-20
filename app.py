import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import requests

# Lightweight model imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NODE_ALERT_URL = os.getenv("NODE_ALERT_URL")

if not DATABASE_URL:
    raise Exception("❌ DATABASE_URL missing in environment variables")


# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------
app = FastAPI(title="NLP Monitoring Service (Lightweight)")


# ---------------------------------------------------
# Load Lightweight Model (SAFE for Render FREE)
# ---------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("✅ Lightweight NLP model loaded")
except Exception as e:
    print("❌ Failed to load lightweight model:", e)
    raise


# ---------------------------------------------------
# DB Connection Pool
# ---------------------------------------------------
try:
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=DATABASE_URL,
        sslmode="require"
    )
    print("✅ PostgreSQL connection pool created")
except Exception as e:
    print("❌ DB connection pool error:", e)
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
# Request body
# ---------------------------------------------------
class ChatMessage(BaseModel):
    sender_id: str
    receiver_id: str
    content: Optional[str] = None
    attachment_url: Optional[str] = None


# ---------------------------------------------------
# NLP abuse detection (Lightweight)
# ---------------------------------------------------
def detect_abuse(message: str):
    if not message.strip():
        return False, 0, "neutral"

    try:
        inputs = tokenizer(message, return_tensors="pt")
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)

        offensive_score = float(scores[0][2])  # class index 2 = offensive
        abuse_detected = offensive_score >= 0.70

        return abuse_detected, round(offensive_score * 100, 2), "offensive"

    except Exception as e:
        print("⚠ NLP error:", e)
        return False, 0, "error"


# ---------------------------------------------------
# Save message to DB
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
        print("❌ Error saving message:", e)
        return False

    finally:
        release_conn(conn)


# ---------------------------------------------------
# API Endpoint
# ---------------------------------------------------
@app.post("/api/messages")
def monitor(chat: ChatMessage):

    if not chat.content and not chat.attachment_url:
        raise HTTPException(status_code=400, detail="Message or attachment is required.")

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
            print("⚠ Failed to send alert:", e)

    return {
        "sender_id": chat.sender_id,
        "receiver_id": chat.receiver_id,
        "content": chat.content,
        "abuse_detected": abuse_detected,
        "abuse_score": score,
        "abuse_label": label
    }


# ---------------------------------------------------
# Render Start Command Port Binding
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
