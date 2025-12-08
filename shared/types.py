# shared/types.py

from typing import List, Dict, Optional

class IntentResult:
    def __init__(self, intent: str, entities: Dict[str, str], query_embedding: Optional[List[float]] = None):
        self.intent = intent
        self.entities = entities
        self.query_embedding = query_embedding

class BaselineContext:
    def __init__(self, rows: List[Dict]):
        self.rows = rows

class EmbeddingContext:
    def __init__(self, rows: List[Dict]):
        self.rows = rows

class FinalContext:
    def __init__(self, rows: List[Dict]):
        self.rows = rows
