import requests, os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
url = "https://api.mistral.ai/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}"}
payload = {
    "model": "mistral-tiny",
    "messages": [{"role": "user", "content": "Say hi from Mistral!"}]
}

response = requests.post(url, headers=headers, json=payload)

print("🔹 Status Code:", response.status_code)
print("🔹 Raw Response:\n", response.text)
