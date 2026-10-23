"""
app/routers/chats.py — Authenticated chat session and messaging endpoints.

Security model
--------------
This backend uses the SERVICE ROLE Supabase client on every query, which
bypasses Row Level Security entirely.  Ownership enforcement is therefore
handled 100% in application code:

  * Every endpoint that references a ``chat_id`` first fetches that chat row
    and verifies ``chats.user_id == current_user["user_id"]``.
  * If the chat row doesn't exist OR belongs to a different user, we return
    the SAME 404 "Chat not found" — never 403 — so callers cannot probe
    whether a chat_id exists for another user.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth import get_current_user
from app.db import get_supabase_client
from app.matching import decide

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    matched_intent: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime


class MessageIn(BaseModel):
    content: str


class ChatRenameIn(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# Internal helper — ownership-verified chat lookup
# ---------------------------------------------------------------------------

def _get_owned_chat(chat_id: str, user_id: str) -> dict:
    """
    Fetch the chat row by ``chat_id`` and assert ownership.

    Raises ``HTTPException(404)`` if:
    - The chat doesn't exist, OR
    - The chat belongs to a different user.

    Treating both cases identically prevents callers from enumerating
    chat IDs that belong to other users.
    """
    client = get_supabase_client()
    result = (
        client
        .table("chats")
        .select("*")
        .eq("id", chat_id)
        .maybe_single()
        .execute()
    )
    chat = result.data
    if chat is None or chat["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


# ---------------------------------------------------------------------------
# POST /chats — create a new chat session
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatOut, status_code=201)
def create_chat(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatOut:
    """
    Create a new chat session for the authenticated user.
    The title defaults to "New chat" and may be auto-updated after the first
    message is sent, or renamed explicitly via PATCH /chats/{chat_id}.
    """
    client = get_supabase_client()
    result = (
        client
        .table("chats")
        .insert({"user_id": current_user["user_id"], "title": "New chat"})
        .execute()
    )
    return ChatOut(**result.data[0])


# ---------------------------------------------------------------------------
# GET /chats — list all chats for the current user
# ---------------------------------------------------------------------------

@router.get("", response_model=List[ChatOut])
def list_chats(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[ChatOut]:
    """
    Return all chat sessions owned by the current user, newest first.
    """
    client = get_supabase_client()
    result = (
        client
        .table("chats")
        .select("*")
        .eq("user_id", current_user["user_id"])
        .order("updated_at", desc=True)
        .execute()
    )
    return [ChatOut(**row) for row in result.data]


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/messages — fetch message history
# ---------------------------------------------------------------------------

@router.get("/{chat_id}/messages", response_model=List[MessageOut])
def get_messages(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[MessageOut]:
    """
    Return all messages in a chat, ordered chronologically (oldest first).
    Returns 404 if the chat doesn't exist or belongs to another user.
    """
    _get_owned_chat(chat_id, current_user["user_id"])

    client = get_supabase_client()
    result = (
        client
        .table("messages")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [MessageOut(**row) for row in result.data]


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/messages — send a message and get a response
# ---------------------------------------------------------------------------

@router.post("/{chat_id}/messages")
def send_message(
    chat_id: str,
    body: MessageIn,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """
    Send a user message and receive an assistant response.

    Workflow
    --------
    1. Verify chat ownership (404 if missing or foreign).
    2. Persist the user message.
    3. Run the matching engine.
    4. Build the assistant content string.
    5. Persist the assistant message (with matched_intent / confidence_score
       when action is "answer").
    6. Update the chat's updated_at; auto-title from the first message if the
       chat title is still "New chat".
    7. Return ``{action, message: MessageOut, candidates?}``.

    Response shape
    --------------
    - ``action``: one of ``"answer"``, ``"no_match"``, ``"clarify"``
    - ``message``: the assistant ``MessageOut`` row just inserted
    - ``candidates``: list of clarification candidates (only when ``action == "clarify"``)
    """
    chat = _get_owned_chat(chat_id, current_user["user_id"])
    client = get_supabase_client()

    # ------------------------------------------------------------------
    # 1. Persist user message
    # ------------------------------------------------------------------
    client.table("messages").insert(
        {
            "chat_id": chat_id,
            "role": "user",
            "content": body.content,
        }
    ).execute()

    # ------------------------------------------------------------------
    # 2. Run matching engine
    # ------------------------------------------------------------------
    model = request.app.state.model
    decision = decide(
        message=body.content,
        model=model,
        user_id=current_user["user_id"],
        chat_id=chat_id,
    )

    action: str = decision["action"]

    # ------------------------------------------------------------------
    # 3. Build assistant content string
    # ------------------------------------------------------------------
    if action == "answer":
        assistant_content = decision["response"]
    elif action == "no_match":
        assistant_content = decision["response"]
    else:  # "clarify"
        examples = [
            f"'{c['example_question']}'"
            for c in decision.get("candidates", [])
        ]
        if len(examples) > 1:
            joined = ", ".join(examples[:-1]) + f", or {examples[-1]}"
        else:
            joined = examples[0] if examples else ""
        assistant_content = (
            f"I found a few possible matches — did you mean: {joined}?"
        )

    # ------------------------------------------------------------------
    # 4. Persist assistant message
    # ------------------------------------------------------------------
    assistant_row: dict = {
        "chat_id": chat_id,
        "role": "assistant",
        "content": assistant_content,
        "matched_intent": decision.get("intent") if action == "answer" else None,
        "confidence_score": decision.get("confidence") if action == "answer" else None,
    }
    assistant_result = (
        client.table("messages").insert(assistant_row).execute()
    )
    assistant_message = MessageOut(**assistant_result.data[0])

    # ------------------------------------------------------------------
    # 5. Update chat's updated_at; auto-title on first real message
    # ------------------------------------------------------------------
    chat_update: dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if chat.get("title") == "New chat":
        raw_title = body.content.strip()
        if len(raw_title) > 40:
            chat_update["title"] = raw_title[:40].rstrip() + "..."
        else:
            chat_update["title"] = raw_title

    client.table("chats").update(chat_update).eq("id", chat_id).execute()

    # ------------------------------------------------------------------
    # 6. Build and return response
    # ------------------------------------------------------------------
    response: dict = {
        "action": action,
        "message": assistant_message,
    }
    if action == "clarify":
        response["candidates"] = decision.get("candidates")

    return response


# ---------------------------------------------------------------------------
# PATCH /chats/{chat_id} — rename a chat
# ---------------------------------------------------------------------------

@router.patch("/{chat_id}", response_model=ChatOut)
def rename_chat(
    chat_id: str,
    body: ChatRenameIn,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatOut:
    """
    Update the title of a chat session.
    Returns 404 if the chat doesn't exist or belongs to another user.
    """
    _get_owned_chat(chat_id, current_user["user_id"])

    client = get_supabase_client()
    result = (
        client
        .table("chats")
        .update({"title": body.title})
        .eq("id", chat_id)
        .execute()
    )
    return ChatOut(**result.data[0])


# ---------------------------------------------------------------------------
# DELETE /chats/{chat_id} — delete a chat (messages cascade)
# ---------------------------------------------------------------------------

@router.delete("/{chat_id}")
def delete_chat(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """
    Permanently delete a chat and all its messages (cascade delete via FK).
    Returns 404 if the chat doesn't exist or belongs to another user.
    """
    _get_owned_chat(chat_id, current_user["user_id"])

    client = get_supabase_client()
    client.table("chats").delete().eq("id", chat_id).execute()
    return {"deleted": True}
