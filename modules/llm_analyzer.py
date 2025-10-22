# modules/llm_analyzer.py
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

def _safe_parse_json(text: str):
    """
    Try to extract JSON from LLM text. Handles plain JSON, text + JSON, or fallback.
    """
    text = text.strip()
    # direct json
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    # fallback: return as tags
    return {"emotion": None, "mindset": None, "interest_tags": [text]}

def analyze_mood_mistral(user_input: str):
    """
    Call Mistral chat completions to extract psychological traits.
    Returns a dict: {emotion, mindset, interest_tags}
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set in environment")

    prompt = (
        "Analyze the psychological state of the following user text and return a JSON ONLY.\n\n"
        f"User text: \"{user_input}\"\n\n"
        "Return a JSON object with keys:\n"
        " - emotion: single-word primary emotion (e.g., anxiety, sadness, joy) or null\n"
        " - mindset: short phrase describing mindset (e.g., 'seeking growth', 'burnout') or null\n"
        " - interest_tags: an array of 3-6 short tags the user would like to read about (genres, topics, themes)\n\n"
        "Example output:\n"
        '{"emotion": "anxiety", "mindset": "career stress", "interest_tags": ["mindfulness", "career clarity", "self-help"]}\n\n'
        'Do not include any extra commentary—output JSON only.'

    )

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are a psychology-aware book recommender assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    res = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=30)
    if res.status_code != 200:
        # surface the error for debugging
        raise RuntimeError(f"Mistral API error {res.status_code}: {res.text}")

    data = res.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        # fallback
        content = res.text

    parsed = _safe_parse_json(content)

    # ensure fields exist
    return {
        "emotion": parsed.get("emotion") if parsed.get("emotion") else None,
        "mindset": parsed.get("mindset") if parsed.get("mindset") else None,
        "interest_tags": parsed.get("interest_tags") if parsed.get("interest_tags") else [user_input]
    }
