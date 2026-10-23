"""
app/auth.py — FastAPI dependency functions for authentication and authorisation.

Design notes
------------
* Token verification is delegated entirely to Supabase's Auth service via
  anon_client.auth.get_user(token).  We never validate the JWT signature
  ourselves; we rely on Supabase's server to do that.

* We decode the JWT a second time with PyJWT — but with signature verification
  DISABLED — solely to read the "aal" claim that auth.get_user() does not
  surface on the returned User object.  This is safe because the token's
  authenticity has already been proven in the step above.

* Profile rows are created by a database trigger (on Supabase's Auth user
  creation event), not by this backend.  If no profile row exists yet we
  return 403 rather than silently creating one.
"""

from typing import Any, Dict

import jwt
from fastapi import Depends, Header, HTTPException

from app.db import get_supabase_anon_client, get_supabase_client


# ---------------------------------------------------------------------------
# Primary dependency — get_current_user
# ---------------------------------------------------------------------------

def get_current_user(authorization: str = Header(None)) -> Dict[str, Any]:
    """
    FastAPI dependency that validates a Supabase JWT sent as
    ``Authorization: Bearer <token>`` and returns a dict with:

    .. code-block:: python

        {
            "user_id": str,         # UUID from Supabase Auth
            "email":   str | None,  # may be None for phone-only accounts
            "profile": dict,        # full row from the public.profiles table
            "aal":     str,         # "aal1" or "aal2"
        }

    Raises
    ------
    HTTPException 401
        Missing / malformed Authorization header, or token rejected by
        Supabase Auth.
    HTTPException 403
        Token is valid but no matching profile row exists yet.
    """

    # ------------------------------------------------------------------
    # 1. Extract bearer token from the Authorization header
    # ------------------------------------------------------------------
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header",
        )

    token = authorization[len("Bearer "):]

    # ------------------------------------------------------------------
    # 2. Verify token with Supabase Auth (anon key client)
    #    UserResponse has shape: { user: User }
    #    User has: .id (str), .email (str | None), ...
    # ------------------------------------------------------------------
    anon_client = get_supabase_anon_client()
    try:
        user_response = anon_client.auth.get_user(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        ) from exc

    if user_response is None or user_response.user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )

    auth_user = user_response.user  # supabase_auth.types.User

    # ------------------------------------------------------------------
    # 3. Fetch the matching profile row via the service-role client
    #    (bypasses RLS so we can always read it regardless of policies)
    # ------------------------------------------------------------------
    service_client = get_supabase_client()
    profile_result = (
        service_client
        .table("profiles")
        .select("*")
        .eq("id", auth_user.id)
        .maybe_single()
        .execute()
    )

    if profile_result.data is None:
        raise HTTPException(
            status_code=403,
            detail="Profile not found — please complete sign-up",
        )

    profile = profile_result.data  # dict of the profiles row

    # ------------------------------------------------------------------
    # 4. Decode JWT payload (NO signature verification — trust already
    #    established above) to read the "aal" claim.
    # ------------------------------------------------------------------
    payload = jwt.decode(
        token,
        options={"verify_signature": False},
        algorithms=["HS256", "RS256"],  # accepted alg list; ignored when verify=False
    )
    aal: str = payload.get("aal", "aal1")

    return {
        "user_id": auth_user.id,
        "email": auth_user.email,
        "profile": profile,
        "aal": aal,
    }


# ---------------------------------------------------------------------------
# Secondary dependency — require_admin
# ---------------------------------------------------------------------------

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency that extends ``get_current_user`` with two admin checks:

    1. The user's profile must have ``is_admin = True``.
    2. The session must have completed an MFA challenge (``aal == "aal2"``).
       This prevents an admin whose account was partially compromised from
       accessing admin resources without re-authenticating with their second
       factor.

    Raises
    ------
    HTTPException 403
        Either the user is not an admin, or the session has not satisfied aal2.
    """
    profile = current_user["profile"]

    if profile.get("is_admin") is not True:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )

    if current_user["aal"] != "aal2":
        raise HTTPException(
            status_code=403,
            detail=(
                "Admin accounts require 2FA verification (aal2) to access "
                "this resource"
            ),
        )

    return current_user


# ---------------------------------------------------------------------------
# Tertiary dependency — require_super_admin
# ---------------------------------------------------------------------------

def require_super_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency that extends ``get_current_user`` with two super-admin
    checks:

    1. The user's profile must have ``is_super_admin = True``.
    2. The session must have completed an MFA challenge (``aal == "aal2"``).
       This prevents a super-admin whose account was partially compromised from
       accessing sensitive resources without re-authenticating with their second
       factor.

    ``is_super_admin`` can only be granted or revoked directly in the database —
    there is intentionally no API route that modifies this flag.

    Raises
    ------
    HTTPException 403
        Either the user is not a super admin, or the session has not satisfied
        aal2.
    """
    profile = current_user["profile"]

    if profile.get("is_super_admin") is not True:
        raise HTTPException(
            status_code=403,
            detail="Super admin access required",
        )

    if current_user["aal"] != "aal2":
        raise HTTPException(
            status_code=403,
            detail=(
                "Super admin accounts require 2FA verification (aal2) to access "
                "this resource"
            ),
        )

    return current_user
