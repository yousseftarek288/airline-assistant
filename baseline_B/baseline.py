# baseline_B/baseline.py
#
# Task B: Baseline Retrieval using Cypher (no embeddings).
# - Connects to your Milestone 2 Neo4j database
# - Uses intent + entities (from Task A) to run exact graph queries
# - Returns results wrapped in BaselineContext

from neo4j import GraphDatabase
from typing import Dict, Any, List

from shared.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from shared.types import BaselineContext


def _run_query(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Helper to run a Cypher query and return list of dict rows.
    We create and close the driver inside this function to avoid
    global driver cleanup issues at Python shutdown.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]
    finally:
        driver.close()


# ----------------------------
# Baseline retrieval logic
# ----------------------------

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

    # ----------------------------
    # Intent: delay_query
    # ----------------------------
    if intent == "delay_query":
        if origin and dest:
            cypher = """
            MATCH (dep:Airport {station_code: $origin})
                  <-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(arr:Airport {station_code: $dest})
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
    # Intent: recommendation
    # ----------------------------
    elif intent == "recommendation":
        base_where = []
        params: Dict[str, Any] = {}

        if origin and dest:
            base_where.append("dep.station_code = $origin AND arr.station_code = $dest")
            params["origin"] = origin
            params["dest"] = dest

        if cabin_class:
            # compare in lowercase so 'Economy' in DB matches 'economy' from Task A
            base_where.append("toLower(j.passenger_class) = $class")
            params["class"] = cabin_class.lower()


        where_clause = ""
        if base_where:
            where_clause = "WHERE " + " AND ".join(base_where)

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
    # Intent: satisfaction_analysis
    # ----------------------------
    elif intent == "satisfaction_analysis":
        params: Dict[str, Any] = {}
        where_clauses = []

        if origin:
            where_clauses.append("dep.station_code = $origin")
            params["origin"] = origin
        if dest:
            where_clauses.append("arr.station_code = $dest")
            params["dest"] = dest
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
    # Fallback: general_flight_query
    # ----------------------------
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
# Manual test
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
