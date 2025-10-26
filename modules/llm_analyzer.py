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
    Extract JSON safely from LLM response.
    Handles:
      - direct JSON
      - text containing JSON
      - fallback to user text if JSON parsing fails
    """
    text = text.strip()
    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    # Fallback: treat entire text as a single interest tag
    return {"emotion": None, "mindset": None, "interest_tags": [text]}

def analyze_mood_mistral(user_input: str):
    """
    Calls Mistral API to analyze psychological traits from user input.
    Returns dict with keys: emotion, mindset, interest_tags
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set in environment")

    prompt = (
        "Analyze the psychological state of the following user text and return JSON ONLY.\n\n"
        f"User text: \"{user_input}\"\n\n"
        "Return a JSON object with keys:\n"
        " - emotion: single-word primary emotion (e.g., anxiety, sadness, joy) or null\n"
        " - mindset: short phrase describing mindset (e.g., 'seeking growth', 'burnout') or null\n"
        " - interest_tags: an array of 3-6 short tags the user would like to read about (genres, topics, themes)\n\n"
        "Example output:\n"
        '{"emotion": "anxiety", "mindset": "career stress", "interest_tags": ["mindfulness", "career clarity", "self-help"]}\n\n'
        "Do not include any extra commentary—JSON ONLY."
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

    try:
        res = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
    except requests.RequestException as e:
        # Return fallback with user_input as interest_tag
        return {"emotion": None, "mindset": None, "interest_tags": [user_input]}

    data = res.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _safe_parse_json(content)

    # Ensure required fields exist
    return {
        "emotion": parsed.get("emotion") or None,
        "mindset": parsed.get("mindset") or None,
        "interest_tags": parsed.get("interest_tags") or [user_input]
    }
