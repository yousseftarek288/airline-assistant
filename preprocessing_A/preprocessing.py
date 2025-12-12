# preprocessing_A/preprocessing.py
#
# Task A: Preprocessing / Understanding the user query
# - Detect intent
# - Extract entities (airports, airline, date, class, time_of_day)
# - Produce a simple dummy embedding (Member C will replace later)

from shared.types import IntentResult
import re

# ----------------------------
# 1. Base lists (airlines, helpers)
# ----------------------------

# Airlines you might see in queries or fleet descriptions
AIRLINES = [
    "Emirates",
    "EgyptAir",
    "Qatar",
    "Saudia",
    "Etihad",
    "Turkish",
    "Turkish Airlines",
    "British Airways",
    "Lufthansa",
    "Air France",
]

# Common economy/business/first keywords
CLASS_KEYWORDS = {
    "economy": ["economy", "eco"],
    "business": ["business", "biz"],
    "first": ["first class", "first"],
    "premium_economy": ["premium economy", "premium eco"],
}


# ----------------------------
# 2. Detect Intent
# ----------------------------

def detect_intent(query: str) -> str:
    """
    Decide what the user wants:
    - delay_query
    - recommendation
    - satisfaction_analysis
    - general_flight_query (default)
    """
    q = query.lower()

    # Recommendation-related FIRST
    if any(word in q for word in [
        "best", "recommend", "suggest", "smooth", "nice",
        "comfortable", "no delay", "good timings", "ideal", "which flight should i take"
    ]):
        return "recommendation"

    # Delay-related NEXT
    if any(word in q for word in [
        "delay", "delayed", "late", "on time", "punctual","on-time","ontime"
        "early arrival", "ahead of time"
    ]):
        return "delay_query"
   

    # Satisfaction / experience-related
    if any(word in q for word in [
        "satisfaction", "happy", "unhappy", "angry", "bad experience",
        "complaint", "bad food", "good food", "comfortable seats",
        "seat space", "leg room", "tight seats"
    ]):
        return "satisfaction_analysis"

    # If nothing fits
    return "general_flight_query"


# ----------------------------
# 3. Detect Airport Codes (from & to)
# ----------------------------

def detect_airports(query: str) -> dict:
    """
    Detect station codes in the text.
    Your dataset uses 3-letter uppercase synthetic codes like LAX, IAX, EWX, etc.
    We use a simple rule:
      - any token of 3 uppercase letters is considered an airport code.
    We assign:
      - first found → 'from'
      - second found → 'to'
    """
    entities = {}

    # Find all 3-letter uppercase tokens (e.g. LAX, IAX, EWX)
    candidates = re.findall(r"\b[A-Z]{3}\b", query)

    # Assign first as origin, second as destination if available
    if len(candidates) >= 1:
        entities["from"] = candidates[0]
    if len(candidates) >= 2:
        entities["to"] = candidates[1]

    return entities


# ----------------------------
# 4. Detect Airline Name (if mentioned)
# ----------------------------

def detect_airline(query: str) -> dict:
    q_lower = query.lower()
    for airline in AIRLINES:
        if airline.lower() in q_lower:
            return {"airline": airline}
    return {}


# ----------------------------
# 5. Detect Date / Time Expressions
# ----------------------------

def detect_date_expression(query: str) -> dict:
    """
    We don't convert to real calendar dates.
    We just tag what user said: 'today', 'tomorrow', 'next_week', etc.
    """
    q = query.lower()

    if "today" in q:
        return {"date": "today"}
    if "tomorrow" in q or "tmrw" in q:
        return {"date": "tomorrow"}
    if "next week" in q:
        return {"date": "next_week"}
    if "this week" in q:
        return {"date": "this_week"}
    if "next month" in q:
        return {"date": "next_month"}
    if "this month" in q:
        return {"date": "this_month"}
    if "weekend" in q or "this weekend" in q:
        return {"date": "weekend"}

    # Specific day names
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for d in days:
        if d in q:
            return {"date": d}

    return {}


def detect_class(query: str) -> dict:
    """
    Detect flight class: economy, business, first, premium economy.
    """
    q = query.lower()

    for class_name, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                # normalize premium_economy to "premium economy" in output if you like
                if class_name == "premium_economy":
                    return {"class": "premium economy"}
                return {"class": class_name}

    return {}


def detect_time_of_day(query: str) -> dict:
    """
    Detect time of day: morning, afternoon, evening, night.
    """
    q = query.lower()

    if any(word in q for word in ["early morning", "morning", "sunrise"]):
        return {"time_of_day": "morning"}

    if any(word in q for word in ["afternoon", "midday"]):
        return {"time_of_day": "afternoon"}

    if any(word in q for word in ["evening", "sunset"]):
        return {"time_of_day": "evening"}

    if any(word in q for word in ["night", "late night"]):
        return {"time_of_day": "night"}

    return {}

def detect_food_preference(query: str) -> dict:
    q = query.lower()
    # very simple rules
    if "high food quality" in q or "very good food" in q or "best food" in q:
        return {"min_food_score": 4}
    if "good food" in q:
        return {"min_food_score": 3}
    if "bad food" in q or "terrible food" in q:
        return {"max_food_score": 2}
    return {}



# ----------------------------
# 6. Placeholder Embedding (Member C will replace later)
# ----------------------------

def dummy_embedding(query: str):
    """
    This is just a placeholder.
    Member C will later replace this with a real embedding model.
    """
    # You can leave this simple; the shape matters more than the values.
    return [0.1, 0.2, 0.3]


# ----------------------------
# 7. Final Preprocessing Function (main function for Task A)
# ----------------------------

def preprocess(query: str) -> IntentResult:
    """
    Main function used by the rest of the system.
    Takes a raw user query string and returns:
      - intent  (str)
      - entities (dict)
      - query_embedding (list of floats)
    """
    intent = detect_intent(query)

    airports = detect_airports(query)
    airline = detect_airline(query)
    date_info = detect_date_expression(query)
    class_info = detect_class(query)
    time_info = detect_time_of_day(query)
    food_info = detect_food_preference(query)

    # Merge all detected entities into one dictionary
    entities = {
        **airports,
        **airline,
        **date_info,
        **class_info,
        **time_info,
        **food_info,
    }

    embedding = dummy_embedding(query)

    return IntentResult(
        intent=intent,
        entities=entities,
        query_embedding=embedding
    )


# ----------------------------
# 8. Manual test (you can run this file alone)
# ----------------------------

if __name__ == "__main__":
    test_queries = [
        "show me best smooth night flights from LAX to IAX tomorrow in economy",
        "recommend business flights next week from EWX to DEX with low delay",
        "what is the satisfaction on Emirates flights this weekend",
        "are there on time flights from ORX to SFX early morning",
    ]

    for q in test_queries:
        print("=" * 60)
        print("QUERY:", q)
        res = preprocess(q)
        print("Intent:", res.intent)
        print("Entities:", res.entities)
        print("Embedding:", res.query_embedding)
