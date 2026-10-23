"""
app/routers/auth_test.py — Authenticated user-profile endpoints.

Routes
------
GET  /me                — Return the current user's profile + computed age.
PATCH /me               — Update safe profile fields (display_name, avatar_url,
                          language_preference).  cnic and date_of_birth are
                          silently ignored even if supplied — use PATCH /me/cnic.
PATCH /me/cnic          — Re-authentication-gated CNIC change flow.
POST  /me/cnic-requests — Submit a CNIC change request with proof document.
GET   /me/cnic-requests — List the current user's own CNIC change requests.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.auth import get_current_user
from app.db import get_supabase_client
from app.validation import validate_cnic_format

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth-test"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_age(dob: date | None) -> int | None:
    """
    Compute age in completed years from *dob* to today.

    Returns None if dob is None.
    """
    if dob is None:
        return None
    today = date.today()
    # Subtract 1 if the birthday hasn't occurred yet this calendar year.
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


def _enrich_profile(profile: dict) -> dict:
    """
    Attach the computed ``age`` field to a profiles dict before returning it.
    Leaves all other fields untouched.
    """
    dob_raw = profile.get("date_of_birth")
    if dob_raw:
        try:
            dob = date.fromisoformat(str(dob_raw))
        except (ValueError, TypeError):
            dob = None
    else:
        dob = None
    return {**profile, "age": _compute_age(dob)}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    language_preference: Optional[str] = None
    # cnic and date_of_birth are intentionally NOT listed here.
    # Any attempt to pass them via PATCH /me will be silently ignored.


class CnicChangeRequest(BaseModel):
    cnic: str
    current_password: str


class CnicRequestOut(BaseModel):
    id: str
    status: str
    reason: Optional[str] = None
    admin_note: Optional[str] = None
    created_at: Any
    reviewed_at: Optional[Any] = None
    consumed: bool = False


# ---------------------------------------------------------------------------
# Allowed MIME types and max size for proof document uploads
# ---------------------------------------------------------------------------
_ALLOWED_PROOF_TYPES = {"image/jpeg", "image/png", "application/pdf"}
_MAX_PROOF_BYTES = 5 * 1024 * 1024  # 5 MB
_CNIC_PROOFS_BUCKET = "cnic-proofs"


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------

@router.get("/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Returns the decoded current-user context, with a computed ``age`` field
    added to the profile based on ``date_of_birth``.
    """
    # Enrich the profile with the computed age before returning.
    enriched = dict(current_user)
    enriched["profile"] = _enrich_profile(current_user["profile"])
    return enriched


# ---------------------------------------------------------------------------
# PATCH /me
# ---------------------------------------------------------------------------

@router.patch("/me")
def update_me(
    body: ProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Partially update the current user's row in ``profiles``.

    Only the fields explicitly provided in the request body are updated.

    Note: ``cnic`` and ``date_of_birth`` are silently ignored here — they
    are managed by dedicated, gated endpoints.
    """
    user_id = current_user["user_id"]
    client = get_supabase_client()

    # Build the update payload from only the supplied (non-None) fields.
    update_payload: dict = {}
    if body.display_name is not None:
        update_payload["display_name"] = body.display_name
    if body.avatar_url is not None:
        update_payload["avatar_url"] = body.avatar_url
    if body.language_preference is not None:
        update_payload["language_preference"] = body.language_preference

    if not update_payload:
        current = (
            client
            .table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
        return _enrich_profile(current.data)

    result = (
        client
        .table("profiles")
        .update(update_payload)
        .eq("id", user_id)
        .execute()
    )
    return _enrich_profile(result.data[0])


# ---------------------------------------------------------------------------
# PATCH /me/cnic
# ---------------------------------------------------------------------------

@router.patch("/me/cnic")
def update_cnic(
    body: CnicChangeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Re-authentication-gated CNIC change endpoint.

    Security gates (in order):
    0. Approved change-request gate: user must have an approved, unconsumed
       row in cnic_change_requests.  A single approval permits exactly one
       CNIC change (the row is marked consumed on success).
    1. Validate new CNIC format.
    2. Re-verify the caller's current password via a fresh Supabase sign-in.
    3. Server-side MFA check: if the user has at least one *verified* MFA
       factor enrolled (via supabase.auth.admin.mfa.list_factors), require
       the current session to be aal2.  If no verified factor is enrolled,
       skip this check.
    4. Attempt the DB update; translate uniqueness violations to 409.

    Returns the updated profile (with computed age) on success.
    """
    user_id = current_user["user_id"]
    user_email = current_user["email"]
    current_aal = current_user["aal"]

    service_client = get_supabase_client()

    # ------------------------------------------------------------------
    # Gate 0 — Approved change-request check (skipped for super-admins)
    # ------------------------------------------------------------------
    approved_request_id: Optional[str] = None
    is_super_admin = current_user.get("profile", {}).get("is_super_admin") is True

    if not is_super_admin:
        approval_result = (
            service_client
            .table("cnic_change_requests")
            .select("id")
            .eq("user_id", user_id)
            .eq("status", "approved")
            .eq("consumed", False)
            .limit(1)
            .execute()
        )
        if not approval_result.data:
            raise HTTPException(
                status_code=403,
                detail="You need an approved CNIC change request before you can update your CNIC.",
            )
        approved_request_id = approval_result.data[0]["id"]

    # ------------------------------------------------------------------
    # Gate 1 — Validate the new CNIC format
    # ------------------------------------------------------------------
    try:
        normalised = validate_cnic_format(body.cnic)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Gate 2 — Re-verify the password via a fresh sign-in
    # ------------------------------------------------------------------
    if not user_email:
        raise HTTPException(
            status_code=400,
            detail="Cannot verify password: no email address on this account.",
        )

    from app.db import get_supabase_anon_client  # local import to avoid circular
    anon_client = get_supabase_anon_client()
    try:
        sign_in_resp = anon_client.auth.sign_in_with_password(
            {"email": user_email, "password": body.current_password}
        )
        if sign_in_resp is None or sign_in_resp.user is None:
            raise HTTPException(status_code=401, detail="Incorrect password.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Incorrect password.")

    # ------------------------------------------------------------------
    # Gate 3 — MFA check (server-side, conditional on enrollment)
    #
    # We call supabase.auth.admin.mfa.list_factors({"user_id": user_id})
    # to retrieve the user's enrolled factors.  This method exists in
    # supabase-py 2.31.0 as SyncGoTrueAdminMFAAPI.list_factors and maps
    # to GET /admin/users/{user_id}/factors on the GoTrue API.
    #
    # We only enforce aal2 when the user has at least one *verified*
    # factor, because an unverified factor means 2FA setup was started
    # but never completed — the user can't satisfy aal2 in that state.
    # ------------------------------------------------------------------
    try:
        factors_resp = service_client.auth.admin.mfa.list_factors(
            {"user_id": user_id}
        )
        # factors_resp is List[Factor]; each Factor has a .status attribute
        has_verified_factor = any(
            getattr(f, "status", None) == "verified" for f in factors_resp
        )
    except Exception as exc:
        logger.error(
            "Failed to retrieve MFA factors for user %s: %s — aborting CNIC change.",
            user_id, exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to verify account security status, please try again.",
        ) from exc

    if has_verified_factor and current_aal != "aal2":
        raise HTTPException(
            status_code=403,
            detail="Please complete two-factor verification first.",
        )

    # ------------------------------------------------------------------
    # Gate 4 — Update the DB; catch uniqueness violations
    # Note: cnic_normalized is a generated column in Postgres, so it must
    # not be included in the update payload.
    # ------------------------------------------------------------------
    try:
        result = (
            service_client
            .table("profiles")
            .update({"cnic": body.cnic})
            .eq("id", user_id)
            .execute()
        )
    except Exception as exc:
        exc_str = str(exc)
        # PostgREST / Postgres uniqueness violation codes/messages
        if "23505" in exc_str or "unique" in exc_str.lower():
            raise HTTPException(
                status_code=409,
                detail="This CNIC or B-Form number is already registered.",
            ) from exc
        logger.error("Failed to update CNIC for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while updating your CNIC.",
        ) from exc

    if not result.data:
        raise HTTPException(status_code=404, detail="Profile not found.")

    # ------------------------------------------------------------------
    # Mark the approval as consumed — one approval = one change
    # ------------------------------------------------------------------
    if approved_request_id is not None:
        try:
            service_client.table("cnic_change_requests").update(
                {"consumed": True}
            ).eq("id", approved_request_id).execute()
        except Exception as exc:
            # Log but don't fail — the CNIC was already updated successfully.
            logger.error(
                "Failed to mark cnic_change_request %s as consumed: %s",
                approved_request_id, exc,
            )

    return _enrich_profile(result.data[0])


# ---------------------------------------------------------------------------
# POST /me/cnic-requests   (PART 1)
# ---------------------------------------------------------------------------

@router.post("/me/cnic-requests", status_code=201)
async def create_cnic_request(
    reason: str = File(...),
    proof_image: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """
    Submit a CNIC change request with a supporting proof document.

    Validation:
    - At most one pending request per user (409 if one already exists).
    - proof_image must be image/jpeg, image/png, or application/pdf, max 5 MB.

    The proof file is uploaded to the private "cnic-proofs" storage bucket
    at path  {user_id}/{uuid}-{original_filename}.

    Returns: {id, status}
    """
    user_id = current_user["user_id"]
    if current_user.get("profile", {}).get("is_super_admin") is True:
        raise HTTPException(
            status_code=400,
            detail="Super admins can change their CNIC directly, no request needed.",
        )

    client = get_supabase_client()

    # ------------------------------------------------------------------
    # Check for an existing pending request
    # ------------------------------------------------------------------
    pending = (
        client
        .table("cnic_change_requests")
        .select("id")
        .eq("user_id", user_id)
        .eq("status", "pending")
        .limit(1)
        .execute()
    )
    if pending.data:
        raise HTTPException(
            status_code=409,
            detail="You already have a pending request.",
        )

    # ------------------------------------------------------------------
    # Validate the uploaded file
    # ------------------------------------------------------------------
    content_type = proof_image.content_type or ""
    if content_type not in _ALLOWED_PROOF_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Allowed types: image/jpeg, image/png, application/pdf."
            ),
        )

    file_bytes = await proof_image.read()
    if len(file_bytes) > _MAX_PROOF_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Proof document exceeds the 5 MB size limit.",
        )

    # ------------------------------------------------------------------
    # Upload to the private storage bucket
    # ------------------------------------------------------------------
    original_name = proof_image.filename or "proof"
    storage_path = f"{user_id}/{uuid.uuid4()}-{original_name}"

    try:
        client.storage.from_(_CNIC_PROOFS_BUCKET).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type},
        )
    except Exception as exc:
        logger.error("Failed to upload CNIC proof for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to upload proof document. Please try again.",
        ) from exc

    # ------------------------------------------------------------------
    # Insert the change-request row
    # ------------------------------------------------------------------
    insert_result = (
        client
        .table("cnic_change_requests")
        .insert({
            "user_id": user_id,
            "reason": reason,
            "proof_image_path": storage_path,
            "status": "pending",
        })
        .execute()
    )

    row = insert_result.data[0]
    return {"id": row["id"], "status": row["status"]}


# ---------------------------------------------------------------------------
# GET /me/cnic-requests   (PART 2)
# ---------------------------------------------------------------------------

@router.get("/me/cnic-requests", response_model=List[CnicRequestOut])
def list_cnic_requests(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[CnicRequestOut]:
    """
    Return the current user's CNIC change requests, newest first.

    proof_image_path is intentionally excluded — the frontend only needs
    to know the status of the request, not the storage path.
    """
    user_id = current_user["user_id"]
    client = get_supabase_client()

    result = (
        client
        .table("cnic_change_requests")
        .select("id, status, reason, admin_note, created_at, reviewed_at, consumed")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return [CnicRequestOut(**row) for row in result.data]

