# embeddings_C/embeddings.py
#
# Task C: Embedding-based Retrieval
# - Build text descriptions of journeys from Neo4j
# - Compute embeddings with TWO models:
#       1) TF-IDF (default)
#       2) Bag-of-Words (CountVectorizer)
# - Given a user query, find the most similar journeys
# - Apply filters (class, min/max food score)
# - Return them as EmbeddingContext

from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

from shared.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from shared.types import IntentResult, EmbeddingContext


# ----------------------------
# 1. Globals for the indexes
# ----------------------------

_VECTOR_TFIDF: Optional[TfidfVectorizer] = None
_DOC_MATRIX_TFIDF: Optional[np.ndarray] = None

_VECTOR_BOW: Optional[CountVectorizer] = None
_DOC_MATRIX_BOW: Optional[np.ndarray] = None

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
    return _run_neo4j_query(cypher, {})


def _build_text_description(row: Dict[str, Any]) -> str:
    """
    Turn a journey row into a text description for semantic search.
    Example:
      'Flight 500 from LAX to IAX in economy class, delay -10 minutes, 
       food satisfaction 4 out of 5, 2000 miles.'
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
    - Fit TF-IDF vectorizer and Bag-of-Words vectorizer
    """
    global _VECTOR_TFIDF, _DOC_MATRIX_TFIDF
    global _VECTOR_BOW, _DOC_MATRIX_BOW
    global _JOURNEY_ROWS

    if (
        _VECTOR_TFIDF is not None
        and _DOC_MATRIX_TFIDF is not None
        and _VECTOR_BOW is not None
        and _DOC_MATRIX_BOW is not None
        and _JOURNEY_ROWS
    ):
        # already built
        return

    print("[Embeddings] Building indexes from Neo4j journeys...")
    _JOURNEY_ROWS = _load_journeys()
    documents = [_build_text_description(row) for row in _JOURNEY_ROWS]

    # Model A: TF-IDF
    _VECTOR_TFIDF = TfidfVectorizer()
    _DOC_MATRIX_TFIDF = _VECTOR_TFIDF.fit_transform(documents)

    # Model B: Bag-of-Words (CountVectorizer)
    _VECTOR_BOW = CountVectorizer()
    _DOC_MATRIX_BOW = _VECTOR_BOW.fit_transform(documents)

    print(f"[Embeddings] Indexed {len(_JOURNEY_ROWS)} journeys "
          f"with TF-IDF and Bag-of-Words.")


def _get_matrix_and_vector(model_name: str):
    """
    Return the (matrix, vectorizer) pair for a given model name.
    model_name: 'tfidf' or 'bow'
    """
    if model_name == "bow":
        return _DOC_MATRIX_BOW, _VECTOR_BOW
    # default to tfidf
    return _DOC_MATRIX_TFIDF, _VECTOR_TFIDF


# ----------------------------
# 3. Build query text
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


# ----------------------------
# 4. Embedding retrieval with filters
# ----------------------------

def embedding_retrieve(
    user_query: str,
    intent_result: IntentResult,
    model_name: str = "tfidf",
    top_k: int = 10,
) -> EmbeddingContext:
    """
    Embedding-based retrieval with SAFE fallback:

    Pass 1: try to respect filters (class + min_food_score/max_food_score).
    Pass 2: if nothing found, relax food filters but keep class.
    Pass 3: if still nothing, ignore all filters and just return the
            top-k most similar journeys.

    This guarantees that, as long as there are journeys in the KG,
    the embeddings section will NOT be empty – even for "weird"
    queries like:
      'best business flights next week with high food quality'
    """

    # Make sure we have journeys + vectorizers
    _ensure_index_built()

    if not _JOURNEY_ROWS:
        # Truly no data in the KG
        return EmbeddingContext(rows=[])

    entities = intent_result.entities or {}
    query_text = _build_query_text(user_query, intent_result.intent, entities)

    # Select matrix + vectorizer
    doc_matrix, vectorizer = _get_matrix_and_vector(model_name)
    if doc_matrix is None or vectorizer is None:
        doc_matrix, vectorizer = _get_matrix_and_vector("tfidf")

    # Vectorize query
    query_vec = vectorizer.transform([query_text])       # (1, N_terms)
    scores = (doc_matrix @ query_vec.T).toarray().ravel()  # (N_docs,)

    # Sort by similarity (descending)
    sorted_indices = np.argsort(-scores)

    wanted_class = entities.get("class")
    min_food = entities.get("min_food_score")
    max_food = entities.get("max_food_score")

    def _collect_with_filters(
        enforce_class: bool,
        enforce_food: bool,
    ) -> List[Dict[str, Any]]:
        """Helper: collect rows with chosen filters."""
        results: List[Dict[str, Any]] = []

        for idx in sorted_indices:
            row = dict(_JOURNEY_ROWS[idx])  # copy

            # ----- Class filter -----
            if enforce_class and wanted_class:
                row_class = (row.get("passenger_class") or "").lower()
                if row_class != wanted_class.lower():
                    continue

            # ----- Food filters -----
            if enforce_food:
                food_score = row.get("food_score")
                if food_score is not None:
                    # soft-ish: allow one point slack
                    if min_food is not None and food_score < (min_food - 1):
                        continue
                    if max_food is not None and food_score > (max_food + 1):
                        continue

            # If we reach here, row is accepted
            row["similarity_score"] = float(scores[idx])
            results.append(row)

            if len(results) >= top_k:
                break

        return results

    # ---- PASS 1: strictest (class + food) ----
    results = _collect_with_filters(enforce_class=True, enforce_food=True)

    # ---- PASS 2: only class, ignore food if PASS 1 empty ----
    if not results:
        results = _collect_with_filters(enforce_class=True, enforce_food=False)

    # ---- PASS 3: ignore all filters, pure semantic top-k ----
    if not results:
        results = []
        for idx in sorted_indices[:top_k]:
            row = dict(_JOURNEY_ROWS[idx])
            row["similarity_score"] = float(scores[idx])
            results.append(row)

    return EmbeddingContext(rows=results)



# ----------------------------
# 5. Manual test (run this file)
# ----------------------------

if __name__ == "__main__":
    fake_entities = {
        "from": "LAX",
        "to": "IAX",
        "class": "economy",
    }
    fake_intent = "recommendation"
    fake_intent_result = IntentResult(intent=fake_intent, entities=fake_entities, query_embedding=None)

    test_query = "best economy flights from LAX to IAX with low delay and good food"

    for model in ["tfidf", "bow"]:
        print("\n=== Testing model:", model, "===")
        ctx = embedding_retrieve(test_query, fake_intent_result, model_name=model, top_k=5)
        for row in ctx.rows:
            print(row)
