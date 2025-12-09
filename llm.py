import time
from openai import OpenAI

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
# ⚠️ PASTE YOUR REAL OPENROUTER KEY HERE
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Initialize the Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Free models to compare
MODELS = {
    "deepseek-r1t2":     "tngtech/deepseek-r1t2-chimera:free",
    "KAT-Coder-Pro V1":  "kwaipilot/kat-coder-pro:free",
    "NemotronNano 12B 2 VL": "nvidia/nemotron-nano-12b-v2-vl:free"
}

# ==============================================================================
# 2. MOCK DATA (Simulate Team Output)
# ==============================================================================
user_query = "Find me a flight from Cairo to Dubai that is comfortable and has low delays."

mock_cypher_results = [
    "Flight: EK924, Airline: Emirates, Route: CAI->DXB, Delay: 12 mins, Aircraft: Boeing 777",
    "Flight: QR230, Airline: Qatar Airways, Route: CAI->DOH->DXB, Delay: 45 mins, Aircraft: Airbus A350",
    "Flight: MS910, Airline: EgyptAir, Route: CAI->DXB, Delay: 0 mins, Aircraft: Boeing 737"
]

mock_vector_results = [
    "Review for EK924: 'Legroom was amazing and the service was top notch. Very smooth flight.'",
    "Review for QR230: 'Long layover and the seats were cramped. Not recommended for sleeping.'",
    "Review for MS910: 'On time but the food was cold and seats were hard.'"
]

# ==============================================================================
# 3. CORE LOGIC (Step 4 Implementation)
# ==============================================================================

def construct_context(cypher_list, vector_list):
    """Merges structured and unstructured data."""
    context_str = "--- FLIGHT DATA (Structured) ---\n"
    for item in cypher_list:
        context_str += f"- {item}\n"
    
    context_str += "\n--- PASSENGER REVIEWS (Semantics) ---\n"
    for item in vector_list:
        context_str += f"- {item}\n"
        
    return context_str

def generate_prompt(query, context):
    """Constructs the prompt with explicit Persona, Context, and Task."""
    prompt = f"""
    ### PERSONA
    You are an expert Airline Assistant for a flight company. 
    Your goal is to help users find the best flights based on their specific needs.
    
    ### CONTEXT DATA
    {context}
    
    ### USER QUESTION 
    "{query}"
    
    ### TASK INSTRUCTIONS
    1. Answer the user's question using **ONLY** the provided CONTEXT DATA above.
    2. If the answer is NOT in the context, say: "I do not have enough information to answer that." (Do not invent facts).
    3. If the user asks for "comfort" or "smoothness", analyze the Passenger Reviews section.
    4. If the user asks for "delays" or "time", analyze the Flight Data section.
    5. Keep your answer professional and concise.
    """
    return prompt

def call_llm(model_id, prompt):
    """Calls API with System Prompt + Retry Logic + Strip."""
    
    # ⭐ RETRY LOGIC: Try 2 times before giving up
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    # ⭐ SYSTEM PROMPT: Enforces the persona globally
                    {"role": "system", "content": "You are a helpful Airline Assistant. strict_context_mode=TRUE. Do not invent facts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            # ⭐ STRIP: Removes empty space from start/end
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"   ⚠️ Timeout/Error on attempt {attempt+1}... Retrying.")
            time.sleep(30) # Wait 10 seconds before retrying

    return "Error: Model failed to respond after retries."

# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================

def run_experiment():
    print(f"🔎 USER QUERY: {user_query}")
    print("="*60)
    
    # 1. Build Context
    final_context = construct_context(mock_cypher_results, mock_vector_results)
    
    # 2. Build Prompt
    final_prompt = generate_prompt(user_query, final_context)
    
    # 3. Test All 3 Models
    results = {}
    
    for friendly_name, model_id in MODELS.items():
        print(f"🤖 Testing Model: {friendly_name}...")
        
        start_time = time.time()
        response = call_llm(model_id, final_prompt)
        end_time = time.time()
        
        duration = round(end_time - start_time, 2)
        results[friendly_name] = {"answer": response, "time": duration}
        print(f"   ✅ Done in {duration} seconds.\n")

    # 4. Final Report
    print("="*60)
    print("📊 EXPERIMENT RESULTS")
    print("="*60)
    
    for name, data in results.items():
        print(f"🔹 MODEL: {name}")
        print(f"⏱️ TIME: {data['time']} sec")
        print(f"📝 ANSWER:\n{data['answer']}")
        print("-" * 40)

if __name__ == "__main__":
    if "sk-or-v1" in OPENROUTER_API_KEY:
        run_experiment()
    else:
        print("⚠️ ERROR: Please paste your real OpenRouter API Key in line 8!")