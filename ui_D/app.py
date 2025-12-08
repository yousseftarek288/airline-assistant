# ui_D/app.py

import sys
import os

# Make repo root importable (fix for streamlit)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from preprocessing_A.preprocessing import preprocess
from baseline_B.baseline import baseline_retrieve
from embeddings_C.embeddings import embedding_retrieve


st.set_page_config(page_title="✈️ Airline Flight Assistant", layout="wide")

st.title("✈️ Airline Assistant")
st.write("Ask about flights, delays, recommendations, classes, food quality, etc.")


# ----------------------------
# User Input
# ----------------------------
user_query = st.text_input("Enter your query:", "")

if st.button("Search") and user_query.strip():
    # 1. Preprocess
    intent_res = preprocess(user_query)

    st.subheader("🧠 Preprocessing (Understanding the query)")
    st.write(f"**Intent:** {intent_res.intent}")
    st.write(f"**Entities Detected:** {intent_res.entities}")

    # 2. Baseline (Cypher retrieval)
    base_ctx = baseline_retrieve(intent_res.intent, intent_res.entities)
    st.subheader("📊 Exact Matches (Baseline Neo4j Cypher)")
    if not base_ctx.rows:
        st.warning("No exact matches found")
    else:
        st.dataframe(base_ctx.rows)

    # 3. Embedding-based retrieval
    emb_ctx = embedding_retrieve(user_query, intent_res, top_k=10)
    st.subheader("🔍 Semantic Matches (Embedding-based Similar Flights)")
    if not emb_ctx.rows:
        st.warning("No similar flights found")
    else:
        st.dataframe(emb_ctx.rows)

else:
    st.info("Type your query above and click **Search** to begin.")
