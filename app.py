import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import psycopg2
import requests

load_dotenv()

DATABASE_URL = "postgresql://almanet_uk_b3pg_user:hmWLlk6zUlnKpSSoJfo9X6TUSbJJqwn9@dpg-d46v5rc9c44c738p0p8g-a.oregon-postgres.render.com/almanet_uk_b3pg"

app = FastAPI(title="Chat Monitoring NLP Service")

# Lazy load NLP model (prevents Out of Memory)
abuse_classifier = None
THRESHOLD = 0.85


def load_nlp_model():
    global abuse_classifier
    if abuse_classifier is None:
        from transformers import pipeline
        abuse_classifier = pipeline("text-classification", model="unitary/toxic-bert")
    return abuse_classifier


def get_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        return conn
    except Exception as e:
        print("❌ DB Error:", e)
        return None


class ChatMessage(BaseModel):
    sender_id: str
    receiver_id: str
    content: Optional[str] = None
    attachment_url: Optional[str] = None


def save_chat(sender_id, receiver_id, content, attachment_url, abuse_detected, abuse_score, abuse_label):
    try:
        conn = get_connection()
        if not conn:
            return False

        cursor = conn.cursor()
        query = """
        INSERT INTO messages
        (sender_id, receiver_id, content, attachment_url, abuse_detected, abuse_score, sentiment, sentiment_score, is_read)
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
        print("❌ Error saving:", e)
        return False


def detect_abuse(message):
    try:
        classifier = load_nlp_model()
        result = classifier(message)[0]

        label = result["label"]
        score = result["score"]
        abuse_detected = (label.lower() in ["toxic", "abuse", "offensive"] and score >= THRESHOLD)

        return abuse_detected, round(score * 100, 2), label

    except Exception as e:
        print("⚠ NLP error:", e)
        return False, 0, "neutral"


@app.post("/api/messages")
def monitor_chat(chat: ChatMessage):

    if not chat.content and not chat.attachment_url:
        raise HTTPException(status_code=400, detail="Message or attachment required")

    abuse_detected, abuse_score, abuse_label = detect_abuse(chat.content or "")

    saved = save_chat(
        chat.sender_id, chat.receiver_id, chat.content,
        chat.attachment_url, abuse_detected, abuse_score, abuse_label
    )

    if not saved:
        raise HTTPException(status_code=500, detail="DB save failed")

    # Alert Node backend ONLY if abuse detected
    if abuse_detected:
        try:
            requests.post(
                "https://your-node-backend-url.onrender.com/api/alerts",
                json={
                    "sender_id": chat.sender_id,
                    "receiver_id": chat.receiver_id,
                    "content": chat.content,
                    "abuse_detected": True,
                    "abuse_score": abuse_score,
                    "abuse_label": abuse_label
                }
            )
        except Exception as e:
            print("⚠️ Alert sending failed:", e)

    return {
        "sender_id": chat.sender_id,
        "receiver_id": chat.receiver_id,
        "content": chat.content,
        "abuse_detected": abuse_detected,
        "abuse_score": abuse_score,
        "abuse_label": abuse_label
    }


# Render Startup (DO NOT REMOVE)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
