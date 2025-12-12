"""
LLM Model Comparison Script for Airline Assistant
Evaluates Mistral 7B, Llama 3.2 3B, and Gemma 2 9B on flight recommendation tasks
"""
import sys
import io
import os

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
# Load environment variables from ui_D/.env
env_path = os.path.join(os.path.dirname(__file__), "ui_D", ".env")
load_dotenv(env_path)

import time
import json
import requests
from typing import List, Dict, Any
from preprocessing_A.preprocessing import preprocess
from baseline_B.baseline import baseline_retrieve
from embeddings_C.embeddings import embedding_retrieve

# Import the functions from app.py
import importlib.util
spec = importlib.util.spec_from_file_location("app", "ui_D/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

# Models to compare
MODELS = {
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "Llama 3.2 3B": "meta-llama/llama-3.2-3b-instruct",
    "Gemma 2 9B": "google/gemma-2-9b-it",
}

# Test cases with expected characteristics
TEST_CASES = [
    {
        "query": "Find me a business class flight from JFK to LAX with low delay",
        "expected_focus": ["business class", "JFK", "LAX", "low delay"],
        "difficulty": "easy"
    },
    {
        "query": "Which economy flights have the best food quality?",
        "expected_focus": ["economy", "food quality", "high score"],
        "difficulty": "medium"
    },
    {
        "query": "I need a flight with minimal delays and good food, what do you recommend?",
        "expected_focus": ["low delay", "good food", "recommendation"],
        "difficulty": "medium"
    },
]


def call_model_with_metrics(model_id: str, prompt: str) -> Dict[str, Any]:
    """Call a model and measure response time and tokens"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "No API key found in environment", "response_time": 0, "tokens": 0}

    api_key = api_key.strip()

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Model Comparison",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful flight information assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    start_time = time.time()

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        response_time = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            return {
                "response": data["choices"][0]["message"]["content"],
                "response_time": response_time,
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                "error": None
            }
        else:
            return {
                "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
                "response_time": response_time,
                "tokens": 0
            }
    except Exception as e:
        return {
            "error": str(e),
            "response_time": time.time() - start_time,
            "tokens": 0
        }


def evaluate_response_quality(response: str, test_case: Dict) -> Dict[str, Any]:
    """
    Evaluate response quality based on multiple criteria
    Returns scores from 0-5 for each criterion
    """
    response_lower = response.lower()

    # 1. Relevance: Does it mention the expected focus areas?
    relevance_score = 0
    for focus_item in test_case["expected_focus"]:
        if focus_item.lower() in response_lower:
            relevance_score += 1
    relevance_score = min(5, (relevance_score / len(test_case["expected_focus"])) * 5)

    # 2. Completeness: Does it provide recommendations with reasoning?
    completeness_score = 0
    if "flight" in response_lower:
        completeness_score += 1
    if any(word in response_lower for word in ["recommend", "suggest", "best", "good option"]):
        completeness_score += 1.5
    if any(word in response_lower for word in ["because", "due to", "since", "as", "reason"]):
        completeness_score += 1.5
    if any(word in response_lower for word in ["delay", "food", "score", "minutes"]):
        completeness_score += 1
    completeness_score = min(5, completeness_score)

    # 3. Accuracy: Does it avoid hallucination (mentioning impossible data)?
    accuracy_score = 5.0  # Start high, deduct for issues
    if "sorry" in response_lower and "no flight" in response_lower:
        accuracy_score = 3.0  # Acceptable if genuinely no results
    # Check for generic responses without specifics
    if response.count("Flight") < 1 and "no flight" not in response_lower:
        accuracy_score -= 2

    # 4. Conciseness: Not too verbose, not too short
    word_count = len(response.split())
    if 50 <= word_count <= 200:
        conciseness_score = 5.0
    elif 30 <= word_count < 50 or 200 < word_count <= 300:
        conciseness_score = 4.0
    elif word_count < 30:
        conciseness_score = 2.0
    else:
        conciseness_score = 3.0

    # 5. Naturalness: Does it sound human and professional?
    naturalness_score = 4.0  # Base score
    if any(phrase in response_lower for phrase in ["i recommend", "i suggest", "here are", "based on"]):
        naturalness_score += 1
    naturalness_score = min(5, naturalness_score)

    return {
        "relevance": round(relevance_score, 2),
        "completeness": round(completeness_score, 2),
        "accuracy": round(accuracy_score, 2),
        "conciseness": round(conciseness_score, 2),
        "naturalness": round(naturalness_score, 2),
        "overall": round((relevance_score + completeness_score + accuracy_score +
                         conciseness_score + naturalness_score) / 5, 2),
        "word_count": word_count
    }


def run_comparison():
    """Run full model comparison"""
    # Verify API key is loaded
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("Make sure .env file exists at: ui_D/.env")
        return

    print(f"✅ API Key loaded: {api_key[:15]}...{api_key[-10:]}")
    print()

    print("=" * 80)
    print("AIRLINE ASSISTANT - LLM MODEL COMPARISON")
    print("=" * 80)
    print()
    print(f"Models being compared:")
    for name, model_id in MODELS.items():
        print(f"  • {name}: {model_id}")
    print()
    print(f"Test cases: {len(TEST_CASES)}")
    print("=" * 80)
    print()

    results = {model_name: [] for model_name in MODELS.keys()}

    # Run each test case
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}/{len(TEST_CASES)} - Difficulty: {test_case['difficulty'].upper()}")
        print(f"{'=' * 80}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Focus: {', '.join(test_case['expected_focus'])}")
        print()

        # 1. Preprocess query
        intent_res = preprocess(test_case['query'])
        print(f"Intent: {intent_res.intent}")
        print(f"Entities: {intent_res.entities}")
        print()

        # 2. Retrieve context
        base_ctx = baseline_retrieve(intent_res.intent, intent_res.entities)
        emb_ctx = embedding_retrieve(test_case['query'], intent_res, model_name="tfidf", top_k=10)

        # 3. Combine context and build prompt
        combined_rows = app.combine_context(base_ctx, emb_ctx, max_items=15)
        prompt = app.build_llm_prompt(test_case['query'], intent_res, combined_rows)

        print(f"Retrieved {len(combined_rows)} flight options from KG")
        print()

        # 4. Test each model
        for model_name, model_id in MODELS.items():
            print(f"Testing {model_name}...")

            metrics = call_model_with_metrics(model_id, prompt)

            if metrics.get("error"):
                print(f"  ❌ Error: {metrics['error']}")
                results[model_name].append({
                    "test_case": i,
                    "query": test_case['query'],
                    "error": metrics['error'],
                    "response_time": metrics['response_time']
                })
                continue

            # Evaluate response quality
            quality = evaluate_response_quality(metrics['response'], test_case)

            print(f"  ✅ Response time: {metrics['response_time']:.2f}s")
            print(f"  📊 Tokens: {metrics['total_tokens']} (prompt: {metrics['prompt_tokens']}, completion: {metrics['completion_tokens']})")
            print(f"  ⭐ Quality Score: {quality['overall']}/5.0")
            print(f"     - Relevance: {quality['relevance']}/5")
            print(f"     - Completeness: {quality['completeness']}/5")
            print(f"     - Accuracy: {quality['accuracy']}/5")
            print(f"     - Word count: {quality['word_count']}")
            print()

            # Store results
            results[model_name].append({
                "test_case": i,
                "query": test_case['query'],
                "difficulty": test_case['difficulty'],
                "response": metrics['response'],
                "response_time": metrics['response_time'],
                "tokens": metrics['total_tokens'],
                "prompt_tokens": metrics['prompt_tokens'],
                "completion_tokens": metrics['completion_tokens'],
                "quality": quality,
                "error": None
            })

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    for model_name in MODELS.keys():
        model_results = [r for r in results[model_name] if r.get("error") is None]

        if not model_results:
            print(f"\n{model_name}: No successful responses")
            continue

        avg_time = sum(r['response_time'] for r in model_results) / len(model_results)
        avg_tokens = sum(r['tokens'] for r in model_results) / len(model_results)
        avg_quality = sum(r['quality']['overall'] for r in model_results) / len(model_results)

        print(f"\n{model_name}:")
        print(f"  Quantitative Metrics:")
        print(f"    • Avg Response Time: {avg_time:.2f}s")
        print(f"    • Avg Total Tokens: {avg_tokens:.0f}")
        print(f"    • Success Rate: {len(model_results)}/{len(TEST_CASES)} ({len(model_results)/len(TEST_CASES)*100:.0f}%)")
        print(f"  Qualitative Metrics:")
        print(f"    • Avg Overall Quality: {avg_quality:.2f}/5.0")

        # Break down by quality dimensions
        avg_relevance = sum(r['quality']['relevance'] for r in model_results) / len(model_results)
        avg_completeness = sum(r['quality']['completeness'] for r in model_results) / len(model_results)
        avg_accuracy = sum(r['quality']['accuracy'] for r in model_results) / len(model_results)
        avg_conciseness = sum(r['quality']['conciseness'] for r in model_results) / len(model_results)
        avg_naturalness = sum(r['quality']['naturalness'] for r in model_results) / len(model_results)

        print(f"    • Relevance: {avg_relevance:.2f}/5")
        print(f"    • Completeness: {avg_completeness:.2f}/5")
        print(f"    • Accuracy: {avg_accuracy:.2f}/5")
        print(f"    • Conciseness: {avg_conciseness:.2f}/5")
        print(f"    • Naturalness: {avg_naturalness:.2f}/5")

    # Determine best model
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print("=" * 80)

    model_scores = {}
    for model_name in MODELS.keys():
        model_results = [r for r in results[model_name] if r.get("error") is None]
        if model_results:
            # Weighted score: 60% quality, 20% speed, 20% token efficiency
            avg_quality = sum(r['quality']['overall'] for r in model_results) / len(model_results)
            avg_time = sum(r['response_time'] for r in model_results) / len(model_results)
            avg_tokens = sum(r['tokens'] for r in model_results) / len(model_results)

            # Normalize scores (lower is better for time and tokens)
            speed_score = max(0, 5 - (avg_time / 2))  # Assume 10s is worst case
            token_score = max(0, 5 - (avg_tokens / 200))  # Assume 1000 tokens is worst case

            weighted_score = (avg_quality * 0.6) + (speed_score * 0.2) + (token_score * 0.2)
            model_scores[model_name] = weighted_score

    if model_scores:
        best_model = max(model_scores, key=model_scores.get)
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Weighted Score: {model_scores[best_model]:.2f}/5.0")
        print(f"\n   This model offers the best balance of:")
        print(f"   • Answer quality and relevance")
        print(f"   • Response speed")
        print(f"   • Token efficiency (cost-effectiveness)")

    # Save detailed results to JSON
    output_file = "model_comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "models": MODELS,
            "test_cases": TEST_CASES,
            "results": results,
            "summary": model_scores
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Detailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
