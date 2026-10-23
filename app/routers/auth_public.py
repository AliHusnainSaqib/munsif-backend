"""
app/routers/auth_public.py — Unauthenticated auth-helper endpoints.

Endpoints here require NO Authorization header.  They are rate-limited by
slowapi (5 requests / minute per client IP) to deter enumeration attacks.

Routes
------
POST /auth/validate-cnic
    Pre-flight check: validate the CNIC format and confirm it is not already
    registered, so the frontend can surface errors before the user finishes
    the full sign-up form.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db import get_supabase_client
from app.validation import validate_cnic_format

# ---------------------------------------------------------------------------
# Rate limiter (shared key function: client IP)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/auth", tags=["auth-public"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ValidateCNICRequest(BaseModel):
    cnic: str


class ValidateCNICResponse(BaseModel):
    valid: bool


# ---------------------------------------------------------------------------
# POST /auth/validate-cnic
# ---------------------------------------------------------------------------

@router.post("/validate-cnic", response_model=ValidateCNICResponse)
@limiter.limit("5/minute")
def validate_cnic(
    request: Request,        # required by slowapi for the IP key function
    body: ValidateCNICRequest,
) -> ValidateCNICResponse:
    """
    Pre-flight CNIC / B-Form validation for the sign-up flow.

    1. Validates the format (exactly 13 digits, no junk sequences).
    2. Checks that the normalised value is not already registered.

    Returns 200 ``{"valid": true}`` if both checks pass.

    Raises
    ------
    400 : CNIC format is invalid (message from validate_cnic_format).
    409 : CNIC is already registered.
    """
    # Step 1 — format validation
    try:
        normalised = validate_cnic_format(body.cnic)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Step 2 — uniqueness check against existing profiles
    service_client = get_supabase_client()
    result = (
        service_client
        .table("profiles")
        .select("id")
        .eq("cnic_normalized", normalised)
        .execute()
    )

    if result and getattr(result, "data", None):
        raise HTTPException(
            status_code=409,
            detail="This CNIC or B-Form number is already registered.",
        )

    return ValidateCNICResponse(valid=True)


class ResolveIdentifierRequest(BaseModel):
    identifier: str

class ResolveIdentifierResponse(BaseModel):
    email: str

@router.post("/resolve-login-identifier", response_model=ResolveIdentifierResponse)
@limiter.limit("5/minute")
def resolve_login_identifier(
    request: Request,
    body: ResolveIdentifierRequest,
) -> ResolveIdentifierResponse:
    """
    Given an email OR CNIC, returns the associated email address.
    Required because Supabase signInWithPassword natively only supports email,
    but we want users to be able to sign in via CNIC.
    """
    client = get_supabase_client()
    ident = body.identifier.strip()
    
    if not ident:
        raise HTTPException(status_code=404, detail="Not Found")

    # 1. Try finding by email directly
    res = client.table("profiles").select("email").eq("email", ident).execute()
    if res and getattr(res, "data", None):
        return ResolveIdentifierResponse(email=res.data[0]["email"])

    # 2. Try finding by CNIC
    try:
        from app.validation import validate_cnic_format
        normalised = validate_cnic_format(ident)
        res_cnic = client.table("profiles").select("email").eq("cnic_normalized", normalised).execute()
        if res_cnic and getattr(res_cnic, "data", None):
            return ResolveIdentifierResponse(email=res_cnic.data[0]["email"])
    except ValueError:
        pass # Not a valid CNIC format, so skip checking it as a CNIC

    raise HTTPException(status_code=404, detail="Not Found")
