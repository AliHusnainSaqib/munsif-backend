"""
One-time ingestion script.

Run from the project root (with .venv active):
    python -m app.scripts.ingest_data

Re-running is safe: it wipes training_examples and intents first,
then re-inserts everything from the JSON files.

Data files expected at:
    data/sbp_intent_responses.json   — {intent_name: response_text, ...}
    data/sbp_chatbot_dataset.json    — [{id, intent, user_query, bot_response, entities, source}, ...]
"""

import json
import pathlib
from sentence_transformers import SentenceTransformer

from app.db import get_supabase_client

# ---------------------------------------------------------------------------
# Paths (relative to project root; run with `python -m app.scripts.ingest_data`)
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]   # sbp-chatbot-backend/
INTENT_RESPONSES_PATH = ROOT / "data" / "sbp_intent_responses.json"
DATASET_PATH = ROOT / "data" / "sbp_chatbot_dataset.json"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    client = get_supabase_client()

    # ------------------------------------------------------------------
    # 1. Load JSON files
    # ------------------------------------------------------------------
    print("Loading JSON files …")
    with open(INTENT_RESPONSES_PATH, encoding="utf-8") as f:
        intent_responses: dict[str, str] = json.load(f)

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    print(f"  Intents in response file : {len(intent_responses)}")
    print(f"  Rows in dataset          : {len(dataset)}")

    # ------------------------------------------------------------------
    # 2. Wipe existing data (clean-slate re-run)
    #    Delete child table first to respect FK constraints.
    # ------------------------------------------------------------------
    print("\nWiping existing data …")
    # Delete all rows by matching a condition that is always true
    # neq against the nil UUID is always true for any real row (uuid columns are
    # never the zero UUID in practice), giving us a clean "delete all" without
    # a type mismatch against the actual uuid primary key.
    NIL_UUID = "00000000-0000-0000-0000-000000000000"
    client.table("training_examples").delete().neq("id", NIL_UUID).execute()
    client.table("intents").delete().neq("id", NIL_UUID).execute()
    print("  Done.")

    # ------------------------------------------------------------------
    # 3. Upsert intents
    # ------------------------------------------------------------------
    print("\nUpserting intents …")
    intent_rows = [
        {"intent_name": name, "response_text": text}
        for name, text in intent_responses.items()
    ]
    client.table("intents").upsert(
        intent_rows, on_conflict="intent_name"
    ).execute()
    print(f"  Upserted {len(intent_rows)} intents.")

    # ------------------------------------------------------------------
    # 4. Fetch inserted intents to build name → id map
    # ------------------------------------------------------------------
    result = client.table("intents").select("id, intent_name").execute()
    intent_id_map: dict[str, str] = {  # intent_name → uuid string
        row["intent_name"]: row["id"] for row in result.data
    }

    # ------------------------------------------------------------------
    # 5. Encode all user queries in one batch
    # ------------------------------------------------------------------
    print(f"\nLoading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    queries = [row["user_query"] for row in dataset]
    print(f"Encoding {len(queries)} queries (this may take a minute) …")
    embeddings = model.encode(queries, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # 6. Insert training examples
    # ------------------------------------------------------------------
    print("\nInserting training examples …")
    skipped = 0
    training_rows = []

    for row, embedding in zip(dataset, embeddings):
        intent_name = row["intent"]
        intent_id = intent_id_map.get(intent_name)
        if intent_id is None:
            # The dataset references an intent that has no response entry —
            # skip rather than insert an orphaned row.
            skipped += 1
            continue

        training_rows.append({
            "intent_id": intent_id,
            "query_text": row["user_query"],
            "source": row.get("source", ""),
            "embedding": embedding.tolist(),  # pgvector accepts a list of floats
        })

    # Insert in batches of 200 to stay within Supabase request-size limits
    BATCH_SIZE = 200
    inserted = 0
    for i in range(0, len(training_rows), BATCH_SIZE):
        batch = training_rows[i : i + BATCH_SIZE]
        client.table("training_examples").insert(batch).execute()
        inserted += len(batch)
        print(f"  Inserted {inserted}/{len(training_rows)} …", end="\r")

    print()  # newline after progress line

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n=== Ingestion complete ===")
    print(f"  Intents inserted         : {len(intent_rows)}")
    print(f"  Training examples inserted: {inserted}")
    if skipped:
        print(f"  Rows skipped (no matching intent in intent_responses): {skipped}")


if __name__ == "__main__":
    main()
