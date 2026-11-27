"""
ISO 6346 Container Number Validation & Parsing.
Standard:
- 4 Letters: 3 uppercase letters (Owner Code) + 1 category identifier (U, J, Z, R)
- 6 Digits: Serial number
- 1 Digit: Check digit calculated with modulo 11 (weights 2^0 to 2^9). Remainder 10 yields 0.
- ISO Size/Type: 1 digit + 3 alphanumeric (e.g., 45G1).
"""
from __future__ import annotations

import re
from typing import NamedTuple

LETTER_MAP = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19, "J": 20,
    "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27, "Q": 28, "R": 29, "S": 30, "T": 31,
    "U": 32, "V": 34, "W": 35, "X": 36, "Y": 37, "Z": 38,
}

VALID_CATEGORIES = {"U", "J", "Z", "R"}
ISO_SIZE_TYPE_PATTERN = re.compile(r"^[0-9][A-Z0-9]{3}$")

class IsoValidationResult(NamedTuple):
    is_valid: bool
    owner_code: str
    category: str
    serial_number: str
    check_digit: int
    expected_check_digit: int | None
    clean_number: str
    error_message: str | None


def calculate_check_digit(container_10_chars: str) -> int:
    """Calculates ISO 6346 check digit for the first 10 characters."""
    total = 0
    for i, char in enumerate(container_10_chars.upper()[:10]):
        if char.isdigit():
            val = int(char)
        elif char in LETTER_MAP:
            val = LETTER_MAP[char]
        else:
            raise ValueError(f"Invalid character in container number: '{char}'")
        weight = 2 ** i
        total += val * weight

    rem = total % 11
    return 0 if rem == 10 else rem


def clean_container_string(text: str) -> str:
    """Cleans up raw OCR text by removing separators and whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned


def validate_iso6346(raw_text: str) -> IsoValidationResult:
    """
    Validates a container number against ISO 6346 specification.
    Returns IsoValidationResult with detailed breakdown.
    """
    cleaned = clean_container_string(raw_text)

    if len(cleaned) < 11:
        return IsoValidationResult(
            is_valid=False, owner_code="", category="", serial_number="",
            check_digit=-1, expected_check_digit=None, clean_number=cleaned,
            error_message=f"Length {len(cleaned)} is less than required 11 characters",
        )

    target = cleaned[:11]
    owner_code = target[:3]
    category = target[3]
    serial = target[4:10]
    check_digit_char = target[10]

    if not owner_code.isalpha():
        return IsoValidationResult(
            is_valid=False, owner_code=owner_code, category=category, serial_number=serial,
            check_digit=-1, expected_check_digit=None, clean_number=target,
            error_message=f"Owner code '{owner_code}' must be 3 letters",
        )

    if category not in VALID_CATEGORIES:
        return IsoValidationResult(
            is_valid=False, owner_code=owner_code, category=category, serial_number=serial,
            check_digit=-1, expected_check_digit=None, clean_number=target,
            error_message=f"Category identifier '{category}' is invalid (expected U, J, Z, or R)",
        )

    if not serial.isdigit():
        return IsoValidationResult(
            is_valid=False, owner_code=owner_code, category=category, serial_number=serial,
            check_digit=-1, expected_check_digit=None, clean_number=target,
            error_message=f"Serial number '{serial}' must be 6 digits",
        )

    if not check_digit_char.isdigit():
        return IsoValidationResult(
            is_valid=False, owner_code=owner_code, category=category, serial_number=serial,
            check_digit=-1, expected_check_digit=None, clean_number=target,
            error_message=f"Check digit '{check_digit_char}' must be a digit",
        )

    actual_check_digit = int(check_digit_char)
    expected_check_digit = calculate_check_digit(target[:10])

    is_valid = actual_check_digit == expected_check_digit
    error_message = (
        None if is_valid else f"Check digit mismatch: actual={actual_check_digit}, expected={expected_check_digit}"
    )

    return IsoValidationResult(
        is_valid=is_valid, owner_code=owner_code, category=category, serial_number=serial,
        check_digit=actual_check_digit, expected_check_digit=expected_check_digit,
        clean_number=target, error_message=error_message,
    )


def validate_iso_size_type(raw_text: str) -> tuple[bool, str, str | None]:
    """
    Validates an ISO 6346 size/type code (4 characters, e.g. "45G1").
    Returns: (is_valid, clean_code, error_message).
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text or "").upper()
    if not cleaned:
        return False, "", "Kode ISO kosong"
    if len(cleaned) != 4:
        return False, cleaned, f"Panjang {len(cleaned)} karakter (harus 4, mis. 45G1)"
    if not ISO_SIZE_TYPE_PATTERN.match(cleaned):
        return False, cleaned, "Format harus 1 digit + 3 karakter alfanumerik (mis. 45G1)"
    return True, cleaned, None