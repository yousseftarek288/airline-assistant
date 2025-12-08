# embeddings_C/embeddings.py
#
# Task C: Embedding-based Retrieval
# - Build text descriptions of journeys from Neo4j
# - Compute TF-IDF embeddings for those descriptions
# - Given a user query, find the most similar journeys
# - Return them as EmbeddingContext

from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from shared.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from shared.types import IntentResult, EmbeddingContext


# ----------------------------
# 1. Globals for the index
# ----------------------------

_VECTOR: Optional[TfidfVectorizer] = None
_DOC_MATRIX: Optional[np.ndarray] = None
_JOURNEY_ROWS: List[Dict[str, Any]] = []


def _run_neo4j_query(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper: run Cypher and return list of dict rows."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]
    finally:
        driver.close()


# ----------------------------
# 2. Build the corpus from Neo4j
# ----------------------------

def _load_journeys() -> List[Dict[str, Any]]:
    """
    Load journeys + flights + airports from Neo4j.
    Each row represents one Journey node with some attributes.
    """
    cypher = """
    MATCH (j:Journey)-[:ON]->(f:Flight)
    MATCH (f)-[:DEPARTS_FROM]->(dep:Airport)
    MATCH (f)-[:ARRIVES_AT]->(arr:Airport)
    RETURN
        id(j) AS journey_id,
        f.flight_number AS flight,
        dep.station_code AS origin,
        arr.station_code AS destination,
        j.passenger_class AS passenger_class,
        j.arrival_delay_minutes AS delay,
        j.food_satisfaction_score AS food_score,
        j.actual_flown_miles AS miles
    LIMIT 2000
    """
    rows = _run_neo4j_query(cypher, {})
    return rows


def _build_text_description(row: Dict[str, Any]) -> str:
    """
    Turn a journey row into a text description for semantic search.
    Example:
      'Flight 500 from LAX to IAX in economy class, delay -10 minutes, food score 4, 2000 miles.'
    """
    flight = row.get("flight")
    origin = row.get("origin")
    destination = row.get("destination")
    passenger_class = row.get("passenger_class", "")
    delay = row.get("delay")
    food_score = row.get("food_score")
    miles = row.get("miles")

    parts = [
        f"Flight {flight}",
        f"from {origin} to {destination}",
    ]

    if passenger_class:
        parts.append(f"in {passenger_class} class")

    if delay is not None:
        parts.append(f"delay {delay} minutes")

    if food_score is not None:
        parts.append(f"food satisfaction {food_score} out of 5")

    if miles is not None:
        parts.append(f"{miles} miles")

    return ", ".join(parts)


def _ensure_index_built():
    """
    Lazy initialization:
    - Load journeys from Neo4j
    - Build text descriptions
    - Fit a TF-IDF vectorizer and compute the document matrix
    """
    global _VECTOR, _DOC_MATRIX, _JOURNEY_ROWS

    if _VECTOR is not None and _DOC_MATRIX is not None and _JOURNEY_ROWS:
        # already built
        return

    print("[Embeddings] Building TF-IDF index from Neo4j journeys...")
    _JOURNEY_ROWS = _load_journeys()

    documents = [_build_text_description(row) for row in _JOURNEY_ROWS]

    _VECTOR = TfidfVectorizer()
    _DOC_MATRIX = _VECTOR.fit_transform(documents)  # shape: (N_docs, N_terms)
    print(f"[Embeddings] Indexed {len(_JOURNEY_ROWS)} journeys.")


# ----------------------------
# 3. Embedding retrieval
# ----------------------------

def _build_query_text(query: str, intent: str, entities: Dict[str, Any]) -> str:
    """
    Build a richer query text by mixing:
    - original user query
    - intent
    - entities (from, to, class, etc.)
    """
    parts = [query]

    parts.append(f"intent {intent}")

    origin = entities.get("from")
    dest = entities.get("to")
    passenger_class = entities.get("class")
    airline = entities.get("airline")
    date = entities.get("date")
    time_of_day = entities.get("time_of_day")

    if origin:
        parts.append(f"from {origin}")
    if dest:
        parts.append(f"to {dest}")
    if passenger_class:
        parts.append(f"in {passenger_class} class")
    if airline:
        parts.append(f"with airline {airline}")
    if date:
        parts.append(f"date {date}")
    if time_of_day:
        parts.append(f"time {time_of_day}")

    return " ".join(parts)


def embedding_retrieve(user_query: str, intent_result: IntentResult, top_k: int = 10) -> EmbeddingContext:
    """
    Main Task C function.

    - Ensures index is built (loads journeys & builds TF-IDF matrix)
    - Builds a query text using user_query + intent + entities
    - Computes similarity to all journeys
    - Applies same filters as baseline when possible:
        * class (economy / business / first)
        * min_food_score / max_food_score
    - Returns top_k most similar as EmbeddingContext(rows=[...])
    """
    _ensure_index_built()

    if _VECTOR is None or _DOC_MATRIX is None or not _JOURNEY_ROWS:
        return EmbeddingContext(rows=[])

    entities = intent_result.entities or {}
    query_text = _build_query_text(user_query, intent_result.intent, entities)

    # Vectorize the query
    query_vec = _VECTOR.transform([query_text])  # shape: (1, N_terms)
    scores = (_DOC_MATRIX @ query_vec.T).toarray().ravel()  # shape: (N_docs,)

    # Sort doc indices by similarity (high to low)
    sorted_indices = np.argsort(-scores)

    # Optional filters
    wanted_class = entities.get("class")           # e.g. "business"
    min_food = entities.get("min_food_score")
    max_food = entities.get("max_food_score")

    results: List[Dict[str, Any]] = []

    for idx in sorted_indices:
        row = dict(_JOURNEY_ROWS[idx])  # copy

        # ----- Apply filters -----
        # Class filter (case-insensitive)
        if wanted_class:
            row_class = (row.get("passenger_class") or "").lower()
            if row_class != wanted_class.lower():
                continue

        # Food score filters
        food_score = row.get("food_score")
        if food_score is not None:
            if min_food is not None and food_score < min_food:
                continue
            if max_food is not None and food_score > max_food:
                continue

        # If it passes filters, keep it
        row["similarity_score"] = float(scores[idx])
        results.append(row)

        if len(results) >= top_k:
            break

    return EmbeddingContext(rows=results)



# ----------------------------
# 4. Manual test (run this file)
# ----------------------------

if __name__ == "__main__":
    # Fake an IntentResult like the one from Task A
    fake_entities = {
        "from": "LAX",
        "to": "IAX",
        "class": "economy",
    }
    fake_intent = "recommendation"
    fake_intent_result = IntentResult(intent=fake_intent, entities=fake_entities, query_embedding=None)

    test_query = "best economy flights from LAX to IAX with low delay and good food"
    ctx = embedding_retrieve(test_query, fake_intent_result, top_k=5)

    print("\n[Embedding Results]")
    for row in ctx.rows:
        print(row)
