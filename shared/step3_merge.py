from typing import List, Dict, Any
from .types import BaselineContext, EmbeddingContext, FinalContext

# ------------------------------
# 1. Normalize Cypher rows
# ------------------------------
def normalize_baseline_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Cypher row → unified dict format.
    Expected Neo4j structure:
        row = {
            "flight_no": "...",
            "from": "...",
            "to": "...",
            "delay": ...,
            "satisfaction": ...
        }
    """

    #If journey is a node:
    if "journey" in row:
        j = row["journey"]
    else:
        j = row  # assume already flat

    return {
        "journey_id": j.get("journey_id") or j.get("id") or j.get("flight_no"),
        "flight_no": j.get("flight_no"),
        "from": j.get("from") or j.get("from_code"),
        "to": j.get("to") or j.get("to_code"),
        "delay": j.get("delay"),
        "satisfaction": j.get("satisfaction"),
        "baseline_score": 1.0,
        "embedding_score": 0.0,
        "source": "baseline"
    }


# ------------------------------
# 2. Normalize embedding rows
# ------------------------------
def normalize_embedding_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert embedding search result → unified dict format.
    Expected shape:
        {
            "metadata": {...},
            "similarity": 0.78
        }
    """

    meta = row.get("metadata", {})
    sim = row.get("similarity") or row.get("score") or 0.0

    return {
        "journey_id": meta.get("journey_id") or meta.get("flight_no"),
        "flight_no": meta.get("flight_no"),
        "from": meta.get("from") or meta.get("from_code"),
        "to": meta.get("to") or meta.get("to_code"),
        "delay": meta.get("delay"),
        "satisfaction": meta.get("satisfaction"),
        "baseline_score": 0.0,
        "embedding_score": float(sim),
        "source": "embedding"
    }


# ------------------------------
# 3. MERGE LOGIC (Step 3 core)
# ------------------------------
def merge_contexts(
    baseline: BaselineContext,
    embedding: EmbeddingContext,
    w_baseline: float = 0.6,
    w_embedding: float = 0.4
) -> FinalContext:
    """
    Step 3:
     - Normalize baseline + embedding
     - Merge by journey_id
     - Compute final score
     - Return FinalContext(rows=[...])
    """

    #1. Normalization
    baseline_norm = [normalize_baseline_row(r) for r in baseline.rows]
    embedding_norm = [normalize_embedding_row(r) for r in embedding.rows]

    #2. We Start merged dict
    merged = {item["journey_id"]: item for item in baseline_norm}

    #3. We Merge embedding items
    for item in embedding_norm:
        jid = item["journey_id"]

        if jid in merged:
            existing = merged[jid]
            existing["embedding_score"] = max(existing["embedding_score"], item["embedding_score"])
            existing["source"] = "hybrid"

            #fill missing Neo4j data if embedding has it
            if existing.get("delay") is None:
                existing["delay"] = item.get("delay")

            if existing.get("satisfaction") is None:
                existing["satisfaction"] = item.get("satisfaction")

        else:
            merged[jid] = item

    #4. We Compute final weighted score
    for item in merged.values():
        item["final_score"] = (
            w_baseline * item.get("baseline_score", 0.0)
            + w_embedding * item.get("embedding_score", 0.0)
        )

    #5. We Sort results
    final_rows = sorted(
        merged.values(),
        key=lambda x: (
            -x["final_score"],
            x["delay"] if x["delay"] is not None else 9999,
            -(x["satisfaction"] if x["satisfaction"] is not None else 0.0)
        )
    )

    #6. Return standardized final context
    return FinalContext(rows=final_rows)


# ------------------------------
# 4. Build LLM context for Step 4
# ------------------------------
def build_llm_context(intent: str, entities: Dict[str, Any],
                      final: FinalContext, top_k: int = 10) -> str:
    """
    Convert merged results → LLM-safe text context.
    """

    lines = []
    lines.append(f"Intent: {intent}")
    lines.append("Entities:")
    for k, v in entities.items():
        lines.append(f"  - {k}: {v}")

    lines.append("")
    lines.append("Flights from graph:")

    for row in final.rows[:top_k]:
        lines.append(
            f"- {row['flight_no']} {row['from']} → {row['to']} | "
            f"delay={row['delay']} | satisfaction={row['satisfaction']} | "
            f"source={row['source']} | "
            f"baseline_score={row['baseline_score']:.2f} | "
            f"embedding_score={row['embedding_score']:.2f}"
        )

    return "\n".join(lines)
