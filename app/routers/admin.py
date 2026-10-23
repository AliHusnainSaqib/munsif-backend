"""
app/routers/admin.py — Admin-only endpoints for dataset management.

Access control
--------------
Every route in this file is protected by ``require_admin``, which enforces:
  1. A valid Supabase JWT (same as get_current_user).
  2. ``profile.is_admin == True``.
  3. Session AAL == "aal2" (MFA challenge completed in the current session).

No route here may be changed to ``get_current_user`` — the AAL2 enforcement
is intentional and must stay.

Model access
------------
Embedding a new training example requires the SentenceTransformer model that
is loaded once at startup into ``app.state.model``.  We access it via the
injected ``Request`` object, exactly as ``chats.py`` does for the matching
engine.  We do NOT load a second model instance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth import require_admin, require_super_admin
from app.db import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class UnansweredQueryOut(BaseModel):
    id: str
    query_text: str
    top_match_intent: Optional[str] = None
    top_score: Optional[float] = None
    resolved: Optional[bool] = None
    admin_note: Optional[str] = None
    created_at: Any          # datetime or str — accept whatever Supabase returns
    user_id: Optional[str] = None
    chat_id: Optional[str] = None


class UnansweredQueryUpdate(BaseModel):
    resolved: Optional[bool] = None
    admin_note: Optional[str] = None


class IntentOut(BaseModel):
    id: str
    intent_name: str
    response_text: str


class IntentCreate(BaseModel):
    intent_name: str
    response_text: str


class IntentUpdate(BaseModel):
    response_text: str


class TrainingExampleCreate(BaseModel):
    query_text: str


class TrainingExampleOut(BaseModel):
    id: str
    query_text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_intent_or_404(intent_id: str) -> dict:
    """Fetch an intent row by id, raise 404 if it doesn't exist."""
    client = get_supabase_client()
    result = (
        client
        .table("intents")
        .select("*")
        .eq("id", intent_id)
        .maybe_single()
        .execute()
    )
    if result.data is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    return result.data


# ---------------------------------------------------------------------------
# GET /admin/unanswered-queries
# ---------------------------------------------------------------------------

@router.get("/unanswered-queries", response_model=List[UnansweredQueryOut])
def list_unanswered_queries(
    resolved: Optional[bool] = None,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> List[UnansweredQueryOut]:
    """
    List unanswered queries, optionally filtered by their resolved status.

    - ``?resolved=false`` → only unresolved rows (the admin backlog)
    - ``?resolved=true``  → only rows already marked as resolved
    - (no param)          → all rows

    Ordered newest-first so the most recent failures surface at the top.
    """
    client = get_supabase_client()
    query = (
        client
        .table("unanswered_queries")
        .select("*")
        .order("created_at", desc=True)
    )
    if resolved is not None:
        query = query.eq("resolved", resolved)

    result = query.execute()
    return [UnansweredQueryOut(**row) for row in result.data]


# ---------------------------------------------------------------------------
# PATCH /admin/unanswered-queries/{query_id}
# ---------------------------------------------------------------------------

@router.patch(
    "/unanswered-queries/{query_id}",
    response_model=UnansweredQueryOut,
)
def update_unanswered_query(
    query_id: str,
    body: UnansweredQueryUpdate,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> UnansweredQueryOut:
    """
    Partially update a single unanswered query row.

    Only the fields explicitly provided in the request body are updated —
    omitted fields are left as-is in the database.  This prevents accidentally
    overwriting ``resolved`` or ``admin_note`` with null.

    Returns 404 if the query_id does not exist.
    """
    client = get_supabase_client()

    # Verify the row exists first so we can return a clean 404.
    existing = (
        client
        .table("unanswered_queries")
        .select("id")
        .eq("id", query_id)
        .maybe_single()
        .execute()
    )
    if existing.data is None:
        raise HTTPException(status_code=404, detail="Unanswered query not found")

    # Build the update payload from only the supplied (non-None) fields.
    update_payload: dict = {}
    if body.resolved is not None:
        update_payload["resolved"] = body.resolved
    if body.admin_note is not None:
        update_payload["admin_note"] = body.admin_note

    if not update_payload:
        # Nothing to update — fetch and return the current state.
        current = (
            client
            .table("unanswered_queries")
            .select("*")
            .eq("id", query_id)
            .single()
            .execute()
        )
        return UnansweredQueryOut(**current.data)

    result = (
        client
        .table("unanswered_queries")
        .update(update_payload)
        .eq("id", query_id)
        .execute()
    )
    return UnansweredQueryOut(**result.data[0])


# ---------------------------------------------------------------------------
# GET /admin/intents
# ---------------------------------------------------------------------------

@router.get("/intents", response_model=List[IntentOut])
def list_intents(
    current_user: Dict[str, Any] = Depends(require_admin),
) -> List[IntentOut]:
    """
    Return all intent rows, sorted alphabetically by intent_name.
    """
    client = get_supabase_client()
    result = (
        client
        .table("intents")
        .select("*")
        .order("intent_name", desc=False)
        .execute()
    )
    return [IntentOut(**row) for row in result.data]


# ---------------------------------------------------------------------------
# POST /admin/intents
# ---------------------------------------------------------------------------

@router.post("/intents", response_model=IntentOut, status_code=201)
def create_intent(
    body: IntentCreate,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> IntentOut:
    """
    Create a new intent.

    Returns 409 Conflict if an intent with the same ``intent_name`` already
    exists (unique-constraint violation), rather than surfacing a raw database
    error.
    """
    client = get_supabase_client()

    # Check for existing intent_name before attempting the insert so we can
    # return a meaningful 409 instead of a raw postgrest error.
    existing = (
        client
        .table("intents")
        .select("id")
        .eq("intent_name", body.intent_name)
        .maybe_single()
        .execute()
    )
    if existing.data is not None:
        raise HTTPException(
            status_code=409,
            detail=f"An intent named '{body.intent_name}' already exists.",
        )

    result = (
        client
        .table("intents")
        .insert({"intent_name": body.intent_name, "response_text": body.response_text})
        .execute()
    )
    return IntentOut(**result.data[0])


# ---------------------------------------------------------------------------
# PATCH /admin/intents/{intent_id}
# ---------------------------------------------------------------------------

@router.patch("/intents/{intent_id}", response_model=IntentOut)
def update_intent(
    intent_id: str,
    body: IntentUpdate,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> IntentOut:
    """
    Update the ``response_text`` of an existing intent.

    Returns 404 if the intent does not exist.
    """
    _get_intent_or_404(intent_id)  # raises 404 if missing

    client = get_supabase_client()
    result = (
        client
        .table("intents")
        .update({"response_text": body.response_text})
        .eq("id", intent_id)
        .execute()
    )
    return IntentOut(**result.data[0])


# ---------------------------------------------------------------------------
# POST /admin/intents/{intent_id}/training-examples
# ---------------------------------------------------------------------------

@router.post(
    "/intents/{intent_id}/training-examples",
    response_model=TrainingExampleOut,
    status_code=201,
)
def add_training_example(
    intent_id: str,
    body: TrainingExampleCreate,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> TrainingExampleOut:
    """
    Add a single training example to an existing intent.

    The text is encoded using the SentenceTransformer model that was loaded
    at startup into ``app.state.model`` — the same model instance and same
    ``normalize_embeddings=True`` flag used by ``ingest_data.py`` and the
    matching engine.  No second copy of the model is loaded.

    Returns 404 if the intent does not exist.
    """
    _get_intent_or_404(intent_id)  # raises 404 if missing

    # Access the singleton model from app state — same pattern as chats.py.
    model = request.app.state.model
    embedding = model.encode(body.query_text, normalize_embeddings=True)

    client = get_supabase_client()
    result = (
        client
        .table("training_examples")
        .insert(
            {
                "intent_id": intent_id,
                "query_text": body.query_text,
                "source": "admin_added",
                "embedding": embedding.tolist(),
            }
        )
        .execute()
    )
    row = result.data[0]
    return TrainingExampleOut(id=row["id"], query_text=row["query_text"])


# ---------------------------------------------------------------------------
# Pydantic schema — user management (super-admin only)
# ---------------------------------------------------------------------------

class UserOut(BaseModel):
    id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    is_admin: Optional[bool] = None
    is_super_admin: Optional[bool] = None


# ---------------------------------------------------------------------------
# Internal helper — user management
# ---------------------------------------------------------------------------

def _get_user_or_404(user_id: str) -> dict:
    """Fetch a profile row by id, raise 404 if it doesn't exist."""
    client = get_supabase_client()
    result = (
        client
        .table("profiles")
        .select("id, email, display_name, is_admin, is_super_admin")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    if result.data is None:
        raise HTTPException(status_code=404, detail="User not found")
    return result.data


# ---------------------------------------------------------------------------
# GET /admin/users
# ---------------------------------------------------------------------------

@router.get("/users", response_model=List[UserOut])
def list_users(
    current_user: Dict[str, Any] = Depends(require_super_admin),
) -> List[UserOut]:
    """
    Return all rows from ``profiles``, ordered by email ascending.

    Fields returned: id, email, display_name, is_admin, is_super_admin.

    Protected by ``require_super_admin`` (is_super_admin=True + aal2).
    ``is_super_admin`` itself cannot be modified via the API — that is a
    deliberate, direct-database-only operation.
    """
    client = get_supabase_client()
    result = (
        client
        .table("profiles")
        .select("id, email, display_name, is_admin, is_super_admin")
        .order("email", desc=False)
        .execute()
    )
    return [UserOut(**row) for row in result.data]


# ---------------------------------------------------------------------------
# POST /admin/users/{user_id}/grant-admin
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/grant-admin", response_model=UserOut)
def grant_admin(
    user_id: str,
    current_user: Dict[str, Any] = Depends(require_super_admin),
) -> UserOut:
    """
    Set ``is_admin = true`` for the given profile id.

    Returns the updated profile row.
    Returns 404 if the user_id does not exist.
    """
    _get_user_or_404(user_id)  # raises 404 if missing

    client = get_supabase_client()
    result = (
        client
        .table("profiles")
        .update({"is_admin": True})
        .eq("id", user_id)
        .execute()
    )
    row = result.data[0]
    return UserOut(
        id=row["id"],
        email=row.get("email"),
        display_name=row.get("display_name"),
        is_admin=row.get("is_admin"),
        is_super_admin=row.get("is_super_admin"),
    )


# ---------------------------------------------------------------------------
# POST /admin/users/{user_id}/revoke-admin
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/revoke-admin", response_model=UserOut)
def revoke_admin(
    user_id: str,
    current_user: Dict[str, Any] = Depends(require_super_admin),
) -> UserOut:
    """
    Set ``is_admin = false`` for the given profile id.

    Returns the updated profile row.
    Returns 404 if the user_id does not exist.
    Returns 400 if the target user has ``is_super_admin = True`` — revoking
    admin access from a super admin would leave them in an inconsistent state.
    Remove ``is_super_admin`` directly in the database first.
    """
    profile = _get_user_or_404(user_id)  # raises 404 if missing

    if profile.get("is_super_admin") is True:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot revoke admin access from a super admin — "
                "remove super admin status directly in the database first"
            ),
        )

    client = get_supabase_client()
    result = (
        client
        .table("profiles")
        .update({"is_admin": False})
        .eq("id", user_id)
        .execute()
    )
    row = result.data[0]
    return UserOut(
        id=row["id"],
        email=row.get("email"),
        display_name=row.get("display_name"),
        is_admin=row.get("is_admin"),
        is_super_admin=row.get("is_super_admin"),
    )


# ===========================================================================
# CNIC Change-Request admin endpoints
# ===========================================================================

class CnicRequestAdminOut(BaseModel):
    id: str
    user_id: str
    reason: Optional[str] = None
    status: str
    admin_note: Optional[str] = None
    reviewed_by: Optional[str] = None
    created_at: Any
    reviewed_at: Optional[Any] = None


class CnicRequestReview(BaseModel):
    status: str           # must be "approved" or "rejected"
    admin_note: Optional[str] = None


def _check_cnic_request_access(requester_user_id: str, current_user: Dict[str, Any]) -> None:
    """
    Enforce role-based access for reviewing/acting on a CNIC change request.

    - Regular user request (is_admin is false): any admin may act on it.
    - Admin user request (is_admin is true or is_super_admin is true):
      only a super-admin may act on it.  If the caller is not a super-admin,
      raise 403 "Only a super admin can review another admin's request".
    """
    client = get_supabase_client()
    res = (
        client
        .table("profiles")
        .select("id, is_admin, is_super_admin")
        .eq("id", requester_user_id)
        .maybe_single()
        .execute()
    )
    requester_profile = res.data or {}
    requester_is_admin = bool(
        requester_profile.get("is_admin") or requester_profile.get("is_super_admin")
    )

    caller_is_super = bool(current_user.get("profile", {}).get("is_super_admin"))

    if requester_is_admin and not caller_is_super:
        raise HTTPException(
            status_code=403,
            detail="Only a super admin can review another admin's request",
        )


# ---------------------------------------------------------------------------
# GET /admin/cnic-requests
# ---------------------------------------------------------------------------

@router.get("/cnic-requests", response_model=List[CnicRequestAdminOut])
def list_cnic_requests(
    status: Optional[str] = None,
    request_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> List[CnicRequestAdminOut]:
    """
    List all CNIC change requests.

    Optional ?status=pending|approved|rejected filter.
    Returns all requests when the query param is omitted.

    proof_image_path is intentionally excluded here — admins retrieve a
    short-lived signed URL via the dedicated proof-url endpoint.

    Role-based visibility:
    - Super admins can view all requests.
    - Regular admins can only view requests from regular users (non-admins).
      Requests from admins are filtered out since only super admins may act on them.
    - If a specific ?request_id= is queried, enforces role-based check directly (403 if admin request).
    """
    client = get_supabase_client()
    query = (
        client
        .table("cnic_change_requests")
        .select("id, user_id, reason, status, admin_note, reviewed_by, created_at, reviewed_at")
        .order("created_at", desc=True)
    )
    if status is not None:
        query = query.eq("status", status)
    if request_id is not None:
        query = query.eq("id", request_id)

    result = query.execute()
    rows = result.data or []

    if request_id is not None:
        if not rows:
            return []
        _check_cnic_request_access(rows[0]["user_id"], current_user)
        return [CnicRequestAdminOut(**row) for row in rows]

    caller_is_super = bool(current_user.get("profile", {}).get("is_super_admin"))
    if caller_is_super:
        return [CnicRequestAdminOut(**row) for row in rows]

    if not rows:
        return []

    # For plain admins: filter out requests submitted by admins or super-admins
    requester_ids = list({row["user_id"] for row in rows})
    profiles_res = (
        client
        .table("profiles")
        .select("id, is_admin, is_super_admin")
        .in_("id", requester_ids)
        .execute()
    )
    admin_user_ids = {
        p["id"]
        for p in (profiles_res.data or [])
        if p.get("is_admin") is True or p.get("is_super_admin") is True
    }

    filtered_rows = [row for row in rows if row["user_id"] not in admin_user_ids]
    return [CnicRequestAdminOut(**row) for row in filtered_rows]


# ---------------------------------------------------------------------------
# GET /admin/cnic-requests/{request_id}
# ---------------------------------------------------------------------------

@router.get("/cnic-requests/{request_id}", response_model=CnicRequestAdminOut)
def get_cnic_request(
    request_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> CnicRequestAdminOut:
    """
    Get a single CNIC change request by ID.

    Enforces role-based access:
    - If the requester is an admin and the caller is not a super admin,
      returns 403 "Only a super admin can review another admin's request".
    """
    client = get_supabase_client()
    result = (
        client
        .table("cnic_change_requests")
        .select("id, user_id, reason, status, admin_note, reviewed_by, created_at, reviewed_at")
        .eq("id", request_id)
        .maybe_single()
        .execute()
    )
    if result.data is None:
        raise HTTPException(status_code=404, detail="CNIC change request not found.")

    _check_cnic_request_access(result.data["user_id"], current_user)

    return CnicRequestAdminOut(**result.data)


# ---------------------------------------------------------------------------
# GET /admin/cnic-requests/{request_id}/proof-url
# ---------------------------------------------------------------------------

@router.get("/cnic-requests/{request_id}/proof-url")
def get_proof_signed_url(
    request_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> dict:
    """
    Generate a short-lived (5-minute) signed URL for the proof document
    attached to a CNIC change request.

    The "cnic-proofs" bucket is private; this endpoint is the only way
    for an admin to view a proof file without making the bucket public.

    Enforces role-based access:
    - If the requester is an admin and the caller is not a super admin,
      returns 403 "Only a super admin can review another admin's request".

    Returns {"url": "<signed URL>"}.
    Returns 404 if the request_id does not exist.
    """
    client = get_supabase_client()

    row_result = (
        client
        .table("cnic_change_requests")
        .select("user_id, proof_image_path")
        .eq("id", request_id)
        .maybe_single()
        .execute()
    )
    if row_result.data is None:
        raise HTTPException(status_code=404, detail="CNIC change request not found.")

    _check_cnic_request_access(row_result.data["user_id"], current_user)

    storage_path: str = row_result.data["proof_image_path"]

    try:
        signed = client.storage.from_("cnic-proofs").create_signed_url(
            path=storage_path,
            expires_in=300,  # 5 minutes
        )
        # supabase-py returns {"signedURL": "..."} or {"signed_url": "..."}
        signed_url = signed.get("signedURL") or signed.get("signed_url") or ""
    except Exception as exc:
        logger.error(
            "Failed to create signed URL for request %s (path=%s): %s",
            request_id, storage_path, exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate proof URL. Please try again.",
        ) from exc

    return {"url": signed_url}


# ---------------------------------------------------------------------------
# PATCH /admin/cnic-requests/{request_id}
# ---------------------------------------------------------------------------

@router.patch("/cnic-requests/{request_id}", response_model=CnicRequestAdminOut)
def review_cnic_request(
    request_id: str,
    body: CnicRequestReview,
    current_user: Dict[str, Any] = Depends(require_admin),
) -> CnicRequestAdminOut:
    """
    Approve or reject a pending CNIC change request.

    Rules:
    - body.status must be "approved" or "rejected" (400 otherwise).
    - Enforces role-based access: If the requester is an admin and the caller
      is not a super admin, returns 403 "Only a super admin can review another admin's request".
    - The request must currently have status "pending" — already-decided
      requests cannot be re-reviewed (400).
    - Sets status, admin_note (optional), reviewed_by (admin's user_id),
      and reviewed_at (server timestamp via now()).

    Returns 404 if the request_id does not exist.
    """
    if body.status not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400,
            detail="status must be 'approved' or 'rejected'.",
        )

    client = get_supabase_client()

    existing = (
        client
        .table("cnic_change_requests")
        .select("id, user_id, status")
        .eq("id", request_id)
        .maybe_single()
        .execute()
    )
    if existing.data is None:
        raise HTTPException(status_code=404, detail="CNIC change request not found.")

    _check_cnic_request_access(existing.data["user_id"], current_user)

    if existing.data["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=(
                f"This request has already been decided (status: '{existing.data['status']}')."
                " Only pending requests can be reviewed."
            ),
        )

    admin_user_id = current_user["user_id"]
    update_payload: dict = {
        "status": body.status,
        "reviewed_by": admin_user_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if body.admin_note is not None:
        update_payload["admin_note"] = body.admin_note

    result = (
        client
        .table("cnic_change_requests")
        .update(update_payload)
        .eq("id", request_id)
        .execute()
    )
    return CnicRequestAdminOut(**result.data[0])


