# baseline_B/baseline.py
#
# Task B: Baseline Retrieval using Cypher (no embeddings).
# - Connects to your Milestone 2 Neo4j database
# - Uses intent + entities (from Task A) to run multiple graph query templates
# - Returns results wrapped in BaselineContext

from neo4j import GraphDatabase
from typing import Dict, Any, List

from shared.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from shared.types import BaselineContext


def _run_query(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to run a Cypher query and return list of dict rows."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]
    finally:
        driver.close()


def baseline_retrieve(intent: str, entities: Dict[str, Any]) -> BaselineContext:
    """
    Main function for Task B.

    Parameters:
        intent: string from Task A (e.g. 'delay_query', 'recommendation', 'satisfaction_analysis', ...)
        entities: dict from Task A (e.g. {'from': 'LAX', 'to': 'IAX', 'class': 'economy', ...})

    Returns:
        BaselineContext(rows=[{...}, {...}, ...])
    """
    intent = intent or "general_flight_query"
    rows: List[Dict[str, Any]] = []

    origin = entities.get("from")
    dest = entities.get("to")
    cabin_class = entities.get("class")      # e.g. "economy", "business"
    airline = entities.get("airline")        # e.g. "Emirates"
    min_food = entities.get("min_food_score")
    max_food = entities.get("max_food_score")

    # ----------------------------
    # 1) DELAY QUERIES (D1, D2, D3)
    # ----------------------------
    if intent == "delay_query":
        # D1: Route-specific delay stats (from + to)
        if origin and dest:
            cypher = """
            MATCH (dep:Airport {station_code: $origin})
                  <-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->
                  (arr:Airport {station_code: $dest})
            MATCH (j:Journey)-[:ON]->(f)
            RETURN
                f.flight_number AS flight,
                dep.station_code AS origin,
                arr.station_code AS destination,
                avg(j.arrival_delay_minutes) AS avg_delay,
                count(j) AS journey_count
            ORDER BY avg_delay ASC
            LIMIT 10
            """
            rows = _run_query(cypher, {"origin": origin, "dest": dest})

        # D2: Origin-only delay stats
        elif origin:
            cypher = """
            MATCH (dep:Airport {station_code: $origin})
                  <-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(arr:Airport)
            MATCH (j:Journey)-[:ON]->(f)
            RETURN
                f.flight_number AS flight,
                dep.station_code AS origin,
                arr.station_code AS destination,
                avg(j.arrival_delay_minutes) AS avg_delay,
                count(j) AS journey_count
            ORDER BY avg_delay ASC
            LIMIT 10
            """
            rows = _run_query(cypher, {"origin": origin})

        # D3: Global delay ranking
        else:
            cypher = """
            MATCH (j:Journey)-[:ON]->(f:Flight)
            MATCH (f)-[:DEPARTS_FROM]->(dep:Airport)
            MATCH (f)-[:ARRIVES_AT]->(arr:Airport)
            RETURN
                f.flight_number AS flight,
                dep.station_code AS origin,
                arr.station_code AS destination,
                avg(j.arrival_delay_minutes) AS avg_delay,
                count(j) AS journey_count
            ORDER BY avg_delay ASC
            LIMIT 10
            """
            rows = _run_query(cypher, {})

    # ----------------------------
    # 2) RECOMMENDATION QUERIES (R1–R4)
    # ----------------------------
    elif intent == "recommendation":
        params: Dict[str, Any] = {}
        where_clauses: List[str] = []

        # Route filters
        if origin and dest:
            where_clauses.append("dep.station_code = $origin AND arr.station_code = $dest")
            params["origin"] = origin
            params["dest"] = dest
        elif origin:
            # R3: origin-only recommendation
            where_clauses.append("dep.station_code = $origin")
            params["origin"] = origin

        # Class filter (case-insensitive)
        if cabin_class:
            where_clauses.append("toLower(j.passenger_class) = $class")
            params["class"] = cabin_class.lower()

        # Food-score preference (from preprocessing)
        if min_food is not None:
            where_clauses.append("j.food_satisfaction_score >= $min_food")
            params["min_food"] = min_food
        if max_food is not None:
            where_clauses.append("j.food_satisfaction_score <= $max_food")
            params["max_food"] = max_food

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        # R1–R4 are covered by different combinations of where_clauses:
        # - origin+dest+class
        # - origin+dest
        # - origin only
        # - nothing (global)
        cypher = f"""
        MATCH (dep:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(arr:Airport)
        MATCH (j:Journey)-[:ON]->(f)
        {where_clause}
        RETURN
            f.flight_number AS flight,
            dep.station_code AS origin,
            arr.station_code AS destination,
            avg(j.arrival_delay_minutes) AS avg_delay,
            avg(j.food_satisfaction_score) AS avg_food,
            avg(j.actual_flown_miles) AS avg_miles,
            count(j) AS journey_count
        ORDER BY avg_delay ASC, avg_food DESC
        LIMIT 10
        """
        rows = _run_query(cypher, params)

    # ----------------------------
    # 3) SATISFACTION ANALYSIS (S1, S2)
    # ----------------------------
    elif intent == "satisfaction_analysis":
        params: Dict[str, Any] = {}
        where_clauses: List[str] = []

        # S1: Route-based satisfaction (from + to)
        if origin:
            where_clauses.append("dep.station_code = $origin")
            params["origin"] = origin
        if dest:
            where_clauses.append("arr.station_code = $dest")
            params["dest"] = dest

        # S2: Airline-focused satisfaction
        if airline:
            where_clauses.append("f.fleet_type_description CONTAINS $airline")
            params["airline"] = airline

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        cypher = f"""
        MATCH (dep:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(arr:Airport)
        MATCH (j:Journey)-[:ON]->(f)
        {where_clause}
        RETURN
            f.flight_number AS flight,
            dep.station_code AS origin,
            arr.station_code AS destination,
            avg(j.food_satisfaction_score) AS avg_food,
            avg(j.arrival_delay_minutes) AS avg_delay,
            count(j) AS journey_count
        ORDER BY avg_food DESC
        LIMIT 10
        """
        rows = _run_query(cypher, params)

    # ----------------------------
    # 4) GENERAL / FALLBACK QUERIES (G1, G2)
    # ----------------------------
    else:
        # G1: Route summary if from+to given
        if origin and dest:
            cypher = """
            MATCH (dep:Airport {station_code: $origin})
                  <-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->
                  (arr:Airport {station_code: $dest})
            MATCH (j:Journey)-[:ON]->(f)
            RETURN
                f.flight_number AS flight,
                dep.station_code AS origin,
                arr.station_code AS destination,
                avg(j.arrival_delay_minutes) AS avg_delay,
                avg(j.food_satisfaction_score) AS avg_food,
                count(j) AS journey_count
            ORDER BY journey_count DESC
            LIMIT 10
            """
            rows = _run_query(cypher, {"origin": origin, "dest": dest})

        # G2: Global popular flights
        else:
            cypher = """
            MATCH (dep:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(arr:Airport)
            MATCH (j:Journey)-[:ON]->(f)
            RETURN
                f.flight_number AS flight,
                dep.station_code AS origin,
                arr.station_code AS destination,
                avg(j.arrival_delay_minutes) AS avg_delay,
                count(j) AS journey_count
            ORDER BY journey_count DESC
            LIMIT 10
            """
            rows = _run_query(cypher, {})

    return BaselineContext(rows=rows)


# ----------------------------
# Manual quick test
# ----------------------------

if __name__ == "__main__":
    example_intent = "recommendation"
    example_entities = {
        "from": "LAX",
        "to": "IAX",
        "class": "economy",
    }

    print("Testing baseline_retrieve with:")
    print("Intent:", example_intent)
    print("Entities:", example_entities)

    context = baseline_retrieve(example_intent, example_entities)

    print("\nReturned rows:")
    for row in context.rows:
        print(row)
