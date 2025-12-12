# LLM Model Comparison Guide for Airline Assistant

This guide helps you systematically compare the three LLM models in your airline assistant application.

## Models Being Compared

1. **Mistral 7B Instruct** - `mistralai/mistral-7b-instruct`
2. **Meta Llama 3.2 3B** - `meta-llama/llama-3.2-3b-instruct`
3. **Google Gemma 2 9B** - `google/gemma-2-9b-it`

##Test Cases

Use these test queries to evaluate each model. For each query, test all three models and record the metrics below.

### Test Case 1: Simple Business Class Query (Easy)
**Query:** `Find me a business class flight from JFK to LAX with low delay`

**Expected behavior:**
- Should identify business class constraint
- Should filter by JFK → LAX route
- Should prioritize low delay times
- Should recommend 1-3 specific flights with reasoning

### Test Case 2: Food Quality Focus (Medium)
**Query:** `Which economy flights have the best food quality?`

**Expected behavior:**
- Should identify economy class constraint
- Should prioritize high food satisfaction scores
- Should rank flights by food quality
- Should explain why recommended flights have good food

### Test Case 3: Multi-Constraint Recommendation (Medium)
**Query:** `I need a flight with minimal delays and good food, what do you recommend?`

**Expected behavior:**
- Should balance two criteria: low delay AND good food
- Should explain the trade-offs
- Should recommend flights that satisfy both constraints

### Test Case 4: Comparison Request (Hard)
**Query:** `Compare first class and business class options for comfort`

**Expected behavior:**
- Should retrieve both first class and business class flights
- Should compare them based on available metrics (food, delay, miles)
- Should provide a structured comparison

### Test Case 5: Reliability Focus (Easy)
**Query:** `What are the most reliable flights with shortest delays?`

**Expected behavior:**
- Should rank by delay time (ascending)
- Should recommend top 3 most reliable options
- Should provide specific delay minutes

## Evaluation Criteria

For each test case and each model, record the following:

### A. Quantitative Metrics

1. **Response Time**: How long did the model take to respond? (seconds)
   - Measure using browser developer tools or observe subjectively
   - Record: _____ seconds

2. **Token Usage** (if available in response):
   - Prompt tokens: _____
   - Completion tokens: _____
   - Total tokens: _____

3. **Success Rate**: Did the model provide a valid response?
   - ✅ Success / ❌ Error

### B. Qualitative Metrics (Rate 1-5 for each)

1. **Relevance** (1-5): Does the response address the user's query?
   - 1 = Completely irrelevant
   - 3 = Partially relevant
   - 5 = Perfectly relevant and on-topic
   - Score: _____

2. **Completeness** (1-5): Does it provide recommendations with reasoning?
   - 1 = No recommendations or reasoning
   - 3 = Basic recommendations without much detail
   - 5 = Detailed recommendations with clear reasoning
   - Score: _____

3. **Accuracy** (1-5): Does it use only the provided context? No hallucination?
   - 1 = Makes up data or contradicts context
   - 3 = Mostly accurate with minor issues
   - 5 = Perfectly accurate, uses only provided data
   - Score: _____

4. **Conciseness** (1-5): Is the response appropriately concise?
   - 1 = Way too verbose or way too short
   - 3 = Acceptable length
   - 5 = Perfect length for the question
   - Score: _____

5. **Naturalness** (1-5): Does it sound human and professional?
   - 1 = Robotic, awkward phrasing
   - 3 = Acceptable but could be more natural
   - 5 = Sounds like a helpful human assistant
   - Score: _____

6. **Overall Quality** (1-5): General impression
   - Score: _____

## Comparison Table Template

Copy this table for each test case:

```
| Metric | Mistral 7B | Llama 3.2 3B | Gemma 2 9B |
|--------|-----------|--------------|------------|
| Response Time (s) | | | |
| Total Tokens | | | |
| Relevance (1-5) | | | |
| Completeness (1-5) | | | |
| Accuracy (1-5) | | | |
| Conciseness (1-5) | | | |
| Naturalness (1-5) | | | |
| **Overall (1-5)** | | | |
```

## Summary Analysis Template

After testing all cases, answer these questions:

### 1. Quantitative Analysis

**Average Response Time:**
- Mistral 7B: _____ seconds
- Llama 3.2 3B: _____ seconds
- Gemma 2 9B: _____ seconds
- **Winner:** _____

**Average Token Usage:**
- Mistral 7B: _____ tokens
- Llama 3.2 3B: _____ tokens
- Gemma 2 9B: _____ tokens
- **Most Efficient:** _____

**Success Rate:**
- Mistral 7B: _____/5 (___%)
- Llama 3.2 3B: _____/5 (___%)
- Gemma 2 9B: _____/5 (___%)
- **Most Reliable:** _____

### 2. Qualitative Analysis

**Average Overall Quality Score:**
- Mistral 7B: _____/5
- Llama 3.2 3B: _____/5
- Gemma 2 9B: _____/5
- **Highest Quality:** _____

**Strengths of Each Model:**

**Mistral 7B:**
- Strengths: _____
- Weaknesses: _____

**Llama 3.2 3B:**
- Strengths: _____
- Weaknesses: _____

**Gemma 2 9B:**
- Strengths: _____
- Weaknesses: _____

### 3. Overall Recommendation

**Best Model for This Use Case:** _____

**Reasoning:**
_____

**Runner-up:** _____

**When to use each model:**
- Use Mistral 7B when: _____
- Use Llama 3.2 3B when: _____
- Use Gemma 2 9B when: _____

## Cost Consideration

Based on OpenRouter pricing (approximate):
- Mistral 7B: Very low cost
- Llama 3.2 3B: Very low cost (smallest model)
- Gemma 2 9B: Low cost

**Cost-Quality Trade-off Analysis:**
_____

## Final Conclusion

Write a 2-3 paragraph summary of your findings, including:
1. Which model performed best overall
2. Key differences between the models
3. Your recommendation for production use
4. Any surprising findings

---

## Example Completed Evaluation

Here's an example of what one test case evaluation might look like:

### Test Case 1: Business Class JFK to LAX

| Metric | Mistral 7B | Llama 3.2 3B | Gemma 2 9B |
|--------|-----------|--------------|------------|
| Response Time (s) | 2.3 | 1.8 | 2.5 |
| Total Tokens | 245 | 198 | 312 |
| Relevance (1-5) | 5 | 4 | 5 |
| Completeness (1-5) | 4 | 3 | 5 |
| Accuracy (1-5) | 5 | 5 | 5 |
| Conciseness (1-5) | 4 | 5 | 3 |
| Naturalness (1-5) | 4 | 3 | 5 |
| **Overall (1-5)** | 4.4 | 4.0 | 4.6 |

**Observations:**
- Gemma 2 9B provided the most complete and natural response but was slightly verbose
- Llama 3.2 3B was fastest but less detailed
- Mistral 7B offered good balance of speed and quality
- All models were accurate and avoided hallucination

---

## Tips for Fair Evaluation

1. **Use the same test queries** for all models
2. **Test in the same order** to account for any caching
3. **Clear context** between tests if needed
4. **Record immediately** - don't rely on memory
5. **Be objective** - focus on measurable criteria
6. **Test multiple times** if results seem inconsistent
7. **Consider your use case** - what matters most for your application?

---

## How to Use This Guide

1. Open your Streamlit app: `streamlit run ui_D/app.py`
2. For each test case:
   - Enter the query
   - Select "Mistral 7B" and click Search
   - Record all metrics
   - Select "Llama 3.2 3B" and click Search
   - Record all metrics
   - Select "Gemma 2 9B" and click Search
   - Record all metrics
3. Complete the comparison tables
4. Write your summary analysis
5. Save this document with your findings

Good luck with your evaluation! 🚀
