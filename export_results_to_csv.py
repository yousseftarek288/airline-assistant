"""
Export model comparison results from JSON to CSV format
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import csv
import os

# Read the JSON results
with open('model_comparison_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create CSV for detailed results
with open('model_comparison_detailed.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([
        'Model',
        'Test Case',
        'Query',
        'Difficulty',
        'Response Time (s)',
        'Total Tokens',
        'Prompt Tokens',
        'Completion Tokens',
        'Relevance (1-5)',
        'Completeness (1-5)',
        'Accuracy (1-5)',
        'Conciseness (1-5)',
        'Naturalness (1-5)',
        'Overall Quality (1-5)',
        'Word Count',
        'Error'
    ])

    # Data rows
    for model_name, results in data['results'].items():
        for result in results:
            if result.get('error'):
                writer.writerow([
                    model_name,
                    result['test_case'],
                    result['query'],
                    result.get('difficulty', 'N/A'),
                    result['response_time'],
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    result['error']
                ])
            else:
                quality = result['quality']
                writer.writerow([
                    model_name,
                    result['test_case'],
                    result['query'],
                    result['difficulty'],
                    round(result['response_time'], 2),
                    result['tokens'],
                    result['prompt_tokens'],
                    result['completion_tokens'],
                    quality['relevance'],
                    quality['completeness'],
                    quality['accuracy'],
                    quality['conciseness'],
                    quality['naturalness'],
                    quality['overall'],
                    quality['word_count'],
                    ''
                ])

print("✅ Detailed results exported to: model_comparison_detailed.csv")

# Create CSV for summary statistics
with open('model_comparison_summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([
        'Model',
        'Avg Response Time (s)',
        'Avg Total Tokens',
        'Success Rate',
        'Avg Overall Quality',
        'Avg Relevance',
        'Avg Completeness',
        'Avg Accuracy',
        'Avg Conciseness',
        'Avg Naturalness',
        'Weighted Score'
    ])

    # Calculate summary for each model
    for model_name, results in data['results'].items():
        successful_results = [r for r in results if not r.get('error')]

        if successful_results:
            avg_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            avg_tokens = sum(r['tokens'] for r in successful_results) / len(successful_results)
            success_rate = f"{len(successful_results)}/{len(results)}"

            avg_overall = sum(r['quality']['overall'] for r in successful_results) / len(successful_results)
            avg_relevance = sum(r['quality']['relevance'] for r in successful_results) / len(successful_results)
            avg_completeness = sum(r['quality']['completeness'] for r in successful_results) / len(successful_results)
            avg_accuracy = sum(r['quality']['accuracy'] for r in successful_results) / len(successful_results)
            avg_conciseness = sum(r['quality']['conciseness'] for r in successful_results) / len(successful_results)
            avg_naturalness = sum(r['quality']['naturalness'] for r in successful_results) / len(successful_results)

            # Calculate weighted score (60% quality, 20% speed, 20% tokens)
            speed_score = max(0, 5 - (avg_time / 2))
            token_score = max(0, 5 - (avg_tokens / 200))
            weighted_score = (avg_overall * 0.6) + (speed_score * 0.2) + (token_score * 0.2)

            writer.writerow([
                model_name,
                round(avg_time, 2),
                round(avg_tokens, 0),
                success_rate,
                round(avg_overall, 2),
                round(avg_relevance, 2),
                round(avg_completeness, 2),
                round(avg_accuracy, 2),
                round(avg_conciseness, 2),
                round(avg_naturalness, 2),
                round(weighted_score, 2)
            ])
        else:
            writer.writerow([
                model_name,
                '',
                '',
                f"0/{len(results)}",
                '',
                '',
                '',
                '',
                '',
                '',
                ''
            ])

print("✅ Summary statistics exported to: model_comparison_summary.csv")

# Create CSV for individual responses
with open('model_comparison_responses.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([
        'Model',
        'Test Case',
        'Query',
        'Response'
    ])

    # Data rows
    for model_name, results in data['results'].items():
        for result in results:
            if not result.get('error'):
                writer.writerow([
                    model_name,
                    result['test_case'],
                    result['query'],
                    result['response']
                ])

print("✅ Model responses exported to: model_comparison_responses.csv")

print("\n" + "="*60)
print("EXPORT COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("1. model_comparison_detailed.csv - All metrics for each test")
print("2. model_comparison_summary.csv - Summary statistics per model")
print("3. model_comparison_responses.csv - Full text responses")
print("\nYou can open these files in Excel or Google Sheets!")
