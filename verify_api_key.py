"""
Script to verify OpenRouter API key and test model access
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

print("=" * 60)
print("OpenRouter API Key Verification")
print("=" * 60)

if not api_key:
    print("❌ ERROR: No API key found in .env file")
    exit(1)

print(f"✓ API Key found: {api_key[:20]}...{api_key[-10:]}")
print()

# Test 1: Check if the key format is valid
if not api_key.startswith("sk-or-v1-"):
    print("⚠️  WARNING: API key doesn't start with 'sk-or-v1-'")
    print("   This might not be a valid OpenRouter API key")
print()

# Test 2: Try to get account information
print("Testing API key with OpenRouter...")
print("-" * 60)

url = "https://openrouter.ai/api/v1/auth/key"
headers = {
    "Authorization": f"Bearer {api_key}",
}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print("✅ API KEY IS VALID!")
        print(f"\nAccount Info:")
        print(f"  - Label: {data.get('data', {}).get('label', 'N/A')}")
        print(f"  - Usage: ${data.get('data', {}).get('usage', 0)}")
        print(f"  - Limit: ${data.get('data', {}).get('limit', 'N/A')}")
        print(f"  - Rate Limit: {data.get('data', {}).get('rate_limit', 'N/A')}")
    else:
        print(f"❌ API KEY VALIDATION FAILED")
        print(f"Response: {resp.text}")

except Exception as e:
    print(f"❌ Error checking API key: {e}")

print()
print("-" * 60)

# Test 3: Try a simple model call
print("\nTesting with a free model (mistralai/mistral-7b-instruct:free)...")
print("-" * 60)

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Test Script",
}

payload = {
    "model": "mistralai/mistral-7b-instruct:free",
    "messages": [
        {"role": "user", "content": "Say 'API key works!' and nothing else."}
    ],
    "max_tokens": 20,
}

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        message = data['choices'][0]['message']['content']
        print(f"✅ MODEL CALL SUCCESSFUL!")
        print(f"Response: {message}")
    else:
        print(f"❌ MODEL CALL FAILED")
        print(f"Response: {resp.text}")

except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 60)
print("Verification Complete")
print("=" * 60)
