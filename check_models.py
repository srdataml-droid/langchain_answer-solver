import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")

headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get("https://ollama.com", headers=headers)

if response.status_code == 200:
    models = [m['name'] for m in response.json().get('models', [])]
    print("Available model tags on your host:")
    for model in models:
        print(f" -> {model}")
else:
    print(f"Failed to fetch tags. Status: {response.status_code}, Response: {response.text}")
