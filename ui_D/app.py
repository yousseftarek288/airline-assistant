# ui_D/app.py

import sys
import os
from dotenv import load_dotenv
load_dotenv()  

# Make repo root importable (fix for streamlit)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import requests
from typing import List, Dict, Any


from preprocessing_A.preprocessing import preprocess
from baseline_B.baseline import baseline_retrieve
from embeddings_C.embeddings import embedding_retrieve


st.set_page_config(page_title="✈️ Airline Flight Assistant", layout="wide")

st.title("✈️ Airline Assistant")
st.write("Ask about flights, delays, recommendations, classes, food quality, etc.")


# 🔽 LLM model selector (3 OpenRouter models + local baseline)
LLM_OPTIONS = {
    "local_rule (no API)": "local_rule",
    "Mistral: Mistral-7B": "mistralai/mistral-7b-instruct:free",
    "Google: Gemini Flash": "google/gemini-2.0-flash-exp:free",
    "Meta: Llama 3.2": "meta-llama/llama-3.2-3b-instruct:free",
}

llm_label = st.selectbox(
    "LLM model for final answer (Milestone 3.c):",
    list(LLM_OPTIONS.keys()),
    key="llm_selector",
)

llm_model_choice = LLM_OPTIONS[llm_label]   # this is either 'local_rule' or a REAL model id





def combine_context(
    base_ctx,
    emb_ctx,
    max_items: int = 15,
) -> List[Dict[str, Any]]:
    """
    3.a Combine KG results from baseline + embeddings into a unified context.
    - Merge rows from base_ctx.rows and emb_ctx.rows
    - Remove duplicates based on (flight, origin, destination)
    - Mark source: baseline / embedding / both
    - Rank by: low delay, then high food, then similarity if present
    """
    combined: Dict[tuple, Dict[str, Any]] = {}

    # 1) Baseline rows
    for row in (base_ctx.rows or []):
        key = (row.get("flight"), row.get("origin"), row.get("destination"))
        new_row = dict(row)
        new_row["source"] = "baseline"
        combined[key] = new_row

    # 2) Embedding rows
    for row in (emb_ctx.rows or []):
        key = (row.get("flight"), row.get("origin"), row.get("destination"))
        if key in combined:
            combined[key]["source"] = "both"
            if "similarity_score" in row:
                combined[key]["similarity_score"] = row["similarity_score"]
        else:
            new_row = dict(row)
            new_row["source"] = "embedding"
            combined[key] = new_row

    all_rows = list(combined.values())

    def sort_key(r: Dict[str, Any]):
        delay = r.get("avg_delay", r.get("delay", 0.0))
        food = r.get("avg_food", r.get("food_score", 0.0))
        sim = r.get("similarity_score", 0.0)
        return (float(delay), -float(food), -float(sim))

    all_rows.sort(key=sort_key)
    return all_rows[:max_items]


def _format_row_for_context(idx: int, row: Dict[str, Any]) -> str:
    """Format a single row line for LLM context."""
    flight = row.get("flight")
    origin = row.get("origin")
    destination = row.get("destination")
    passenger_class = row.get("passenger_class") or row.get("class") or "Unknown class"
    delay = row.get("avg_delay", row.get("delay"))
    food = row.get("avg_food", row.get("food_score"))
    miles = row.get("avg_miles", row.get("miles"))
    source = row.get("source", "unknown")

    parts = [f"{idx}. Flight {flight} from {origin} to {destination}"]
    parts.append(f"class: {passenger_class}")
    if delay is not None:
        parts.append(f"avg delay: {round(float(delay), 1)} minutes")
    if food is not None:
        parts.append(f"food score: {round(float(food), 1)}/5")
    if miles is not None:
        parts.append(f"miles: {miles}")
    parts.append(f"source: {source}")
    if "similarity_score" in row:
        parts.append(f"similarity: {round(float(row['similarity_score']), 3)}")

    return ", ".join(parts)


def build_llm_prompt(user_query: str, intent_res, combined_rows: List[Dict[str, Any]]) -> str:
    """
    3.b – structured prompt: Persona + Context + Task
    """
    persona = """Persona:
You are a flight information assistant working for an airline company.
You answer questions about flights, delays, and passenger satisfaction.
You must only use the information given in the retrieved context below."""

    user_part = f"User question:\n{user_query}\n"

    intent_part = (
        "Detected interpretation:\n"
        f"- Intent: {intent_res.intent}\n"
        f"- Entities: {intent_res.entities}\n"
    )

    if not combined_rows:
        context_part = (
            "Context (retrieved from the knowledge graph):\n"
            "(No flights found in the knowledge graph for this query.)\n"
        )
    else:
        lines = [
            _format_row_for_context(i, row)
            for i, row in enumerate(combined_rows, start=1)
        ]
        context_part = (
            "Context (retrieved from the knowledge graph):\n"
            + "\n".join(lines)
            + "\n"
        )

    task = """Task:
- Answer the user's question using ONLY the provided context above.
- If there are good candidate flights, recommend 1–3 and explain WHY
  (consider delay, food score, miles, and any other relevant metrics).
- If no flights match the constraints, clearly say that and suggest the closest
  alternatives from the context.
- Do NOT invent flights or data that are not present in the context."""

    return "\n\n".join([persona, user_part, intent_part, context_part, task])


def call_openrouter_model(model_id: str, prompt: str) -> str:
    """
    Call an OpenRouter model by its *direct* ID, e.g. 'mistralai/dev-2512'.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "[Error: OPENROUTER_API_KEY not found in environment]"

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",   # required by OpenRouter
        "X-Title": "Airline Assistant",       # required by OpenRouter
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

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error calling {model_id}: {e}]"




# ----------------------------
# User Input
# ----------------------------
user_query = st.text_input("Enter your query:", "")

embedding_choice = st.selectbox(
    "Embedding model (for semantic matches):",
    ["TF-IDF", "Bag-of-Words"],
)

# map UI choice to internal code
if embedding_choice == "TF-IDF":
    embedding_model_name = "tfidf"
else:
    embedding_model_name = "bow"
    

if st.button("Search") and user_query.strip():
    # 1. Preprocess
    intent_res = preprocess(user_query)

    st.subheader("🧠 Preprocessing (Understanding the query)")
    st.write(f"**Intent:** {intent_res.intent}")
    st.write(f"**Entities Detected:** {intent_res.entities}")

    # 2. Baseline (Cypher)
    base_ctx = baseline_retrieve(intent_res.intent, intent_res.entities)
    st.subheader("📊 Exact Matches (Baseline Neo4j Cypher)")
    if not base_ctx.rows:
        st.warning("No exact matches found")
    else:
        st.dataframe(base_ctx.rows)

    # 3. Embeddings
    emb_ctx = embedding_retrieve(
        user_query,
        intent_res,
        model_name=embedding_model_name,
        top_k=10,
    )
    st.subheader("🔍 Semantic Matches (Embedding-based Similar Flights)")
    if not emb_ctx.rows:
        st.warning("No similar flights found")
    else:
        st.dataframe(emb_ctx.rows)

    # 4. LLM Layer: combine + prompt
    combined_rows = combine_context(base_ctx, emb_ctx, max_items=15)
    prompt = build_llm_prompt(user_query, intent_res, combined_rows)

    # 5. Final answer:
    if llm_model_choice == "local_rule":
        # Simple internal explanation using combined context (no external API)
        if not combined_rows:
            final_answer = (
                "I could not find any flights in the knowledge graph that match your "
                "constraints. Try relaxing class, date, or food quality and search again."
            )
        else:
            # Very simple rule-based explanation
            lines = ["Here are some recommended flights based on the retrieved context:"]
            for i, row in enumerate(combined_rows[:3], start=1):
                lines.append(_format_row_for_context(i, row))
            lines.append(
                "These were chosen by balancing lower delay and higher food scores. "
            )
            final_answer = "\n".join(lines)
    else:
        # Call OpenRouter (3.c)
        final_answer = call_openrouter_model(llm_model_choice, prompt)

    st.subheader(f"🧾 Final Answer ({llm_model_choice})")
    st.write(final_answer)

    # Show the structured prompt for report & model comparison
    with st.expander("Show LLM prompt (context + persona + task)"):
        st.code(prompt, language="markdown")
else:
    st.info("Type your query above and click **Search** to begin.")



