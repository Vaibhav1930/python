from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import psycopg2
import requests
import os
from dotenv import load_dotenv
from typing import Optional

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NODE_ALERT_URL = os.getenv("NODE_ALERT_URL")

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(title="Chat Monitoring NLP Service")

# ------------------------------
# Abuse Detection Model
# ------------------------------
abuse_classifier = pipeline("text-classification", model="unitary/toxic-bert")

# ------------------------------
# PostgreSQL Connection
# ------------------------------
def get_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        return conn
    except Exception as e:
        print("❌ Database connection error:", e)
        return None

# ------------------------------
# Request Body Model
# ------------------------------
class ChatMessage(BaseModel):
    sender_id: str
    receiver_id: str
    content: Optional[str] = None
    attachment_url: Optional[str] = None

# ------------------------------
# Save Chat to Database
# ------------------------------
def save_chat(sender_id, receiver_id, content, attachment_url,
              abuse_detected, abuse_score, abuse_label):

    try:
        conn = get_connection()
        if conn is None:
            return False

        cursor = conn.cursor()
        query = """
        INSERT INTO messages
        (sender_id, receiver_id, content, attachment_url, abuse_detected, 
        abuse_score, sentiment, sentiment_score, is_read)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE)
        """

        cursor.execute(query, (
            sender_id, receiver_id, content, attachment_url,
            abuse_detected, abuse_score, abuse_label, abuse_score
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print("❌ Error saving chat:", e)
        return False

# ------------------------------
# NLP Abuse Detection
# ------------------------------
THRESHOLD = 0.85

def detect_abuse(message):
    try:
        result = abuse_classifier(message)[0]
        label = result["label"]
        score = result["score"]

        abuse_detected = (label.lower() in ["toxic", "abuse", "offensive"]
                          and score >= THRESHOLD)

        return abuse_detected, round(score * 100, 2), label

    except Exception as e:
        print("⚠ NLP error:", e)
        return False, 0, "neutral"

# ------------------------------
# API: Monitor & Save Chat
# ------------------------------
@app.post("/api/messages")
def monitor_chat(chat: ChatMessage):

    if not chat.content and not chat.attachment_url:
        raise HTTPException(status_code=400, detail="Message or attachment required")

    abuse_detected, abuse_score, abuse_label = detect_abuse(chat.content or "")

    saved = save_chat(
        chat.sender_id,
        chat.receiver_id,
        chat.content,
        chat.attachment_url,
        abuse_detected,
        abuse_score,
        abuse_label
    )

    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save chat in database.")

    # If abusive → send alert to Node backend
    if abuse_detected and NODE_ALERT_URL:
        try:
            response = requests.post(f"{NODE_ALERT_URL}/api/alerts", json={
                "sender_id": chat.sender_id,
                "receiver_id": chat.receiver_id,
                "content": chat.content,
                "abuse_detected": True,
                "abuse_score": abuse_score,
                "abuse_label": abuse_label
            })
            print("📩 Alert sent:", response.status_code)
        except Exception as e:
            print("⚠️ Failed to send alert:", e)

    return {
        "sender_id": chat.sender_id,
        "receiver_id": chat.receiver_id,
        "content": chat.content,
        "abuse_detected": abuse_detected,
        "abuse_score": abuse_score,
        "abuse_label": abuse_label
    }

# ------------------------------
# Render Port Binding
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
