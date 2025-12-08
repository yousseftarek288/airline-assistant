# main.py
from preprocessing_A.preprocessing import preprocess
from baseline_B.baseline import baseline_retrieve
from embeddings_C.embeddings import embedding_retrieve

if __name__ == "__main__":
    print("✈️ Airline Assistant")
    print("---------------------")
    user_query = input("Ask about flights: ")

    intent_res = preprocess(user_query)
    print("\n[Preprocessing]")
    print("Intent:", intent_res.intent)
    print("Entities:", intent_res.entities)

    base_ctx = baseline_retrieve(intent_res.intent, intent_res.entities)
    print("\n[Baseline Cypher Results]")
    if not base_ctx.rows:
        print("(No baseline flights found)")
    else:
        for row in base_ctx.rows:
            print(row)

    emb_ctx = embedding_retrieve(user_query, intent_res, top_k=5)
    print("\n[Embedding-based Similar Flights]")
    if not emb_ctx.rows:
        print("(No similar journeys found)")
    else:
        for row in emb_ctx.rows:
            print(row)
