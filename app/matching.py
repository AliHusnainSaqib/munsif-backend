"""
Matching engine — core decision logic.

The SentenceTransformer model is NOT loaded here; it is loaded once at
FastAPI startup (lifespan) and injected via the `model` parameter.
This keeps the matching logic fully testable without a running app.

Supabase RPC expected:
    match_training_examples(query_embedding vector, match_count int)
    → table(query_text text, intent_name text, score float)
    (ordered best-to-worst by cosine similarity)
"""

import logging
from typing import Any, Optional

from app.db import get_supabase_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds — change here, nowhere else
# ---------------------------------------------------------------------------
CONFIDENT_THRESHOLD: float = 0.55   # below this → no_match
CLEAR_WINNER_MARGIN: float = 0.08   # gap required to skip clarification


def decide(
    message: str,
    model: Any,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> dict:
    """
    Core decision function.  Replicates the Streamlit prototype logic exactly.

    Parameters
    ----------
    message : str
        Raw user input.
    model : SentenceTransformer
        The already-loaded embedding model (passed from app state).
    user_id : str, optional
        The authenticated user's UUID.  When provided, it is included in
        unanswered_queries log rows for attribution.
    chat_id : str, optional
        The active chat session UUID.  When provided, it is included in
        unanswered_queries log rows for traceability.

    Returns
    -------
    dict with key "action" in {"no_match", "answer", "clarify"}.
    """

    # ------------------------------------------------------------------
    # Step 1 — empty / whitespace guard
    # ------------------------------------------------------------------
    if not message or not message.strip():
        return {
            "action": "no_match",
            "response": "Could you type your question?",
        }

    # ------------------------------------------------------------------
    # Step 2 — encode query
    # ------------------------------------------------------------------
    embedding = model.encode(message, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # Step 3 — vector similarity search via Supabase RPC
    # ------------------------------------------------------------------
    client = get_supabase_client()
    rpc_result = (
        client.rpc(
            "match_training_examples",
            {
                "query_embedding": embedding.tolist(),
                "match_count": 30,
            },
        )
        .execute()
    )
    rows = rpc_result.data  # list of {query_text, intent_name, score}

    # ------------------------------------------------------------------
    # Step 4 — group by intent, keep highest score per intent,
    #           preserve the best-to-worst order from the RPC result.
    #           Take top 3 distinct intents.
    # ------------------------------------------------------------------
    seen: dict[str, dict] = {}   # intent_name → best row so far
    ordered_intents: list[str] = []  # insertion order = best-first

    for row in rows:
        intent = row["intent_name"]
        if intent not in seen:
            seen[intent] = row
            ordered_intents.append(intent)

    # ranked: list of (intent_name, score, matched_query_text), top 3
    ranked = [
        (seen[intent]["intent_name"], seen[intent]["score"], seen[intent]["query_text"])
        for intent in ordered_intents[:3]
    ]

    # Edge-case: RPC returned no rows at all
    if not ranked:
        return {
            "action": "no_match",
            "response": (
                "I'm not confident I understood that. Could you rephrase, "
                "or contact your bank branch / the SBP helpline directly?"
            ),
        }

    top_intent, top_score, top_query = ranked[0]

    # ------------------------------------------------------------------
    # Step 6 — below confidence threshold → no_match + log unanswered
    # ------------------------------------------------------------------
    if top_score < CONFIDENT_THRESHOLD:
        try:
            unanswered_payload: dict = {
                "query_text": message,
                "top_match_intent": top_intent,
                "top_score": top_score,
            }
            if user_id is not None:
                unanswered_payload["user_id"] = user_id
            if chat_id is not None:
                unanswered_payload["chat_id"] = chat_id
            client.table("unanswered_queries").insert(unanswered_payload).execute()
        except Exception:
            logger.exception("Failed to log unanswered query to Supabase.")

        return {
            "action": "no_match",
            "response": (
                "I'm not confident I understood that. Could you rephrase, "
                "or contact your bank branch / the SBP helpline directly?"
            ),
        }

    # ------------------------------------------------------------------
    # Step 7 — clear winner: only one ranked result OR top beats 2nd
    #           by at least CLEAR_WINNER_MARGIN
    # ------------------------------------------------------------------
    is_clear_winner = len(ranked) == 1 or (
        top_score - ranked[1][1] >= CLEAR_WINNER_MARGIN
    )

    if is_clear_winner:
        # Fetch the canonical response text for this intent
        intent_row = (
            client.table("intents")
            .select("response_text")
            .eq("intent_name", top_intent)
            .single()
            .execute()
        )
        response_text = intent_row.data["response_text"]

        return {
            "action": "answer",
            "intent": top_intent,
            "response": response_text,
            "confidence": top_score,
        }

    # ------------------------------------------------------------------
    # Step 8 — confident but close competitors → ask for clarification
    # ------------------------------------------------------------------
    candidates = [
        {
            "intent": intent_name,
            "score": score,
            "example_question": example_query,
        }
        for intent_name, score, example_query in ranked
    ]

    return {
        "action": "clarify",
        "candidates": candidates,
    }
