import requests
import os
from dotenv import load_dotenv

load_dotenv("ui_D/.env")

api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key loaded: {api_key[:20]}..." if api_key else "No API key found")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Airline Assistant Test",
}

# Try a simple free model
payload = {
    "model": "mistralai/mistral-7b-instruct:free",
    "messages": [
        {"role": "user", "content": "Say hello in one word"}
    ],
    "max_tokens": 10,
}

print("\nTesting OpenRouter API...")
print(f"Model: {payload['model']}")

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print("✅ SUCCESS!")
        print(f"Response: {data['choices'][0]['message']['content']}")
    else:
        print(f"❌ ERROR: {resp.status_code}")
        print(f"Response: {resp.text}")

except Exception as e:
    print(f"❌ Exception: {e}")
