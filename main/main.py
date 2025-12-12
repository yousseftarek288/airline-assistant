# main.py
from preprocessing_A.preprocessing import preprocess
from baseline_B.baseline import baseline_retrieve

if __name__ == "__main__":
    user_query = input("Ask about flights: ")

    intent_res = preprocess(user_query)
    print("\n[Preprocessing]")
    print("Intent:", intent_res.intent)
    print("Entities:", intent_res.entities)

    base_ctx = baseline_retrieve(intent_res.intent, intent_res.entities)
    print("\n[Baseline Cypher Results]")
    for row in base_ctx.rows:
        print(row)
