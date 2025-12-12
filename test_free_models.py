"""
Test which free models are available on OpenRouter with your API key
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

if not api_key:
    print("❌ No API key found")
    exit(1)

print("=" * 70)
print("Testing Free Models on OpenRouter")
print("=" * 70)
print()

# List of free models to test
FREE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "google/gemma-7b-it:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "liquid/lfm-40b:free",
    "meta-llama/llama-3.1-8b-instruct:free",
]

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Test Script",
}

working_models = []
failed_models = []

for model_id in FREE_MODELS:
    print(f"Testing: {model_id}")

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 5,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)

        if resp.status_code == 200:
            print(f"  ✅ WORKS!")
            working_models.append(model_id)
        else:
            error_msg = resp.text[:100]
            print(f"  ❌ Failed: {resp.status_code} - {error_msg}")
            failed_models.append((model_id, resp.status_code))

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        failed_models.append((model_id, "Exception"))

    print()

print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

if working_models:
    print(f"✅ WORKING MODELS ({len(working_models)}):")
    print("-" * 70)
    for model in working_models:
        print(f"  • {model}")
else:
    print("❌ No working models found")

print()
print(f"❌ FAILED MODELS ({len(failed_models)}):")
print("-" * 70)
for model, status in failed_models:
    print(f"  • {model} ({status})")

print()
print("=" * 70)
print("RECOMMENDATION FOR YOUR APP:")
print("=" * 70)

if len(working_models) >= 3:
    print("Use these 3 models in your app:")
    for i, model in enumerate(working_models[:3], 1):
        print(f"  {i}. {model}")
else:
    print(f"Only {len(working_models)} working models found.")
    print("You may need to add credits to your OpenRouter account for more models.")
