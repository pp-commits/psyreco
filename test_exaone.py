import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")  # public version
prompt = "Write a friendly message from EXAONE."
print(client.text_generation(prompt, max_new_tokens=50))
