# modules/llm_analyzer.py
import os
import json
import requests

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

def analyze_mood_mistral(user_input: str):
    """
    Call Mistral API to extract psychological traits from user input.
    Returns JSON: {emotion, mindset, interest_tags}
    """
    prompt = f"""
    Analyze the psychological state of this user input: "{user_input}"
    and return a valid JSON with the following keys:
    - emotion: main emotion
    - mindset: current mindset or outlook
    - interest_tags: list of interest-based tags (genres, topics, etc.)
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    response = requests.post(MISTRAL_URL, headers=headers, json=payload)
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return {
            "emotion": "unknown",
            "mindset": "neutral",
            "interest_tags": [user_input]
        }
