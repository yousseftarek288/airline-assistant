"""
Test affordable (non-free suffix) models on OpenRouter
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("ui_D/.env")

api_key = os.getenv("OPENROUTER_API_KEY")

print("=" * 70)
print("Testing Affordable Models (without :free suffix)")
print("=" * 70)
print()

# Affordable models without :free suffix
AFFORDABLE_MODELS = [
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-2-9b-it",
    "microsoft/phi-3-mini-128k-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "nousresearch/hermes-3-llama-3.1-8b",
    "google/gemini-flash-1.5",
    "anthropic/claude-3-haiku",
    "openai/gpt-3.5-turbo",
]

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Test Script",
}

working_models = []

for model_id in AFFORDABLE_MODELS:
    print(f"Testing: {model_id}")

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Say OK"}
        ],
        "max_tokens": 5,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            response_text = data['choices'][0]['message']['content']
            print(f"  ✅ WORKS! Response: {response_text}")
            working_models.append(model_id)
        else:
            try:
                error_data = resp.json()
                error_msg = error_data.get('error', {}).get('message', resp.text[:80])
            except:
                error_msg = resp.text[:80]
            print(f"  ❌ Failed ({resp.status_code}): {error_msg}")

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:80]}")

    print()

print("=" * 70)
print("WORKING MODELS:")
print("=" * 70)

if working_models:
    for i, model in enumerate(working_models, 1):
        print(f"  {i}. {model}")

    if len(working_models) >= 3:
        print()
        print("Recommendation: Use these 3 in your app:")
        for i, model in enumerate(working_models[:3], 1):
            print(f"  {i}. {model}")
else:
    print("No working models found. You may need to add credits to OpenRouter.")
