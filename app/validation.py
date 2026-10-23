"""
app/validation.py — Shared validation utilities.

Contains reusable validation functions that are called by multiple routers
to avoid duplicating logic.
"""

import re


def validate_cnic_format(raw: str) -> str:
    """
    Validate and normalise a Pakistani CNIC or B-Form number.

    Processing steps
    ----------------
    1. Strip every non-digit character (hyphens, spaces, etc.).
    2. Require exactly 13 digits.
    3. Reject obviously fake values:
       - All identical digits (e.g. 1111111111111)
       - Simple ascending sequence  (0123456789012)
       - Simple descending sequence (9876543210987)

    Parameters
    ----------
    raw : str
        The raw CNIC string as supplied by the caller.

    Returns
    -------
    str
        The cleaned 13-digit string if valid.

    Raises
    ------
    ValueError
        With a descriptive message if the value fails any check.
    """
    # Step 1 — strip non-digits
    digits = re.sub(r"\D", "", raw)

    # Step 2 — must be exactly 13 digits
    if len(digits) != 13:
        raise ValueError(
            "CNIC / B-Form number must contain exactly 13 digits "
            "(format: XXXXX-XXXXXXX-X)."
        )

    # Step 3a — reject all-identical-digit sequences
    if len(set(digits)) == 1:
        raise ValueError(
            "CNIC / B-Form number is not valid (all identical digits)."
        )

    # Step 3b — reject simple ascending sequence (0123456789012 …)
    ascending = "0123456789"
    ascending_doubled = ascending * 2          # cover wrap-around
    if digits[:10] in ascending_doubled:
        raise ValueError(
            "CNIC / B-Form number is not valid (sequential pattern detected)."
        )

    # Step 3c — reject simple descending sequence
    descending = "9876543210"
    descending_doubled = descending * 2
    if digits[:10] in descending_doubled:
        raise ValueError(
            "CNIC / B-Form number is not valid (sequential pattern detected)."
        )

    return digits
