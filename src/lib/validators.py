"""Input validation and sanitization utilities.

Implements DATA-001 through DATA-005.
"""

import re
import logging

logger = logging.getLogger("sfu_library_mcp")

# Maximum query length to prevent abuse
MAX_QUERY_LENGTH = 1000


def validate_isbn(isbn: str) -> tuple[bool, str]:
    """Validate an ISBN-10 or ISBN-13 number.

    Args:
        isbn: ISBN string, may contain hyphens.

    Returns:
        Tuple of (is_valid, cleaned_isbn).
    """
    if not isbn:
        return False, ""

    cleaned = isbn.replace("-", "").replace(" ", "").strip()

    if len(cleaned) == 10:
        return _validate_isbn10(cleaned), cleaned
    elif len(cleaned) == 13:
        return _validate_isbn13(cleaned), cleaned
    else:
        return False, cleaned


def _validate_isbn10(isbn: str) -> bool:
    """Validate ISBN-10 check digit."""
    if not re.match(r"^\d{9}[\dXx]$", isbn):
        return False
    total = 0
    for i, char in enumerate(isbn[:9]):
        total += int(char) * (10 - i)
    check = isbn[9].upper()
    check_val = 10 if check == "X" else int(check)
    total += check_val
    return total % 11 == 0


def _validate_isbn13(isbn: str) -> bool:
    """Validate ISBN-13 check digit."""
    if not isbn.isdigit():
        return False
    total = 0
    for i, char in enumerate(isbn):
        total += int(char) * (1 if i % 2 == 0 else 3)
    return total % 10 == 0


def validate_issn(issn: str) -> tuple[bool, str]:
    """Validate an ISSN number.

    Args:
        issn: ISSN string, may contain hyphen (e.g., '1234-5678').

    Returns:
        Tuple of (is_valid, cleaned_issn).
    """
    if not issn:
        return False, ""

    cleaned = issn.replace("-", "").replace(" ", "").strip()

    if len(cleaned) != 8:
        return False, cleaned

    if not re.match(r"^\d{7}[\dXx]$", cleaned):
        return False, cleaned

    # ISSN check digit validation
    total = 0
    for i, char in enumerate(cleaned[:7]):
        total += int(char) * (8 - i)
    remainder = total % 11
    check_expected = 0 if remainder == 0 else 11 - remainder
    check_char = cleaned[7].upper()
    check_val = 10 if check_char == "X" else int(check_char)

    return check_val == check_expected, cleaned


def sanitize_search_query(query: str) -> str:
    """Sanitize a search query to prevent injection and XSS.

    Backward-compatible wrapper around sanitize_search_query_advanced().

    Args:
        query: Raw user search query.

    Returns:
        Sanitized query string.
    """
    return sanitize_search_query_advanced(query)


# Maximum number of boolean operators Primo supports in a single query
_MAX_BOOLEAN_OPERATORS = 30


def sanitize_search_query_advanced(query: str) -> str:
    """Sanitize a search query while preserving Primo-valid syntax.

    Preserves:
    - Double quotes (phrase search)
    - Boolean operators AND, OR, NOT (uppercase only)
    - Wildcards ? and *

    Still strips:
    - HTML tags
    - Null bytes
    - Stray $$ sequences (Primo subfield delimiters)
    - Collapses whitespace, enforces max length

    Args:
        query: Raw user search query.

    Returns:
        Sanitized query string safe for the Primo API.
    """
    if not query:
        return ""

    # Remove null bytes
    sanitized = query.replace("\x00", "")

    # Strip HTML tags
    sanitized = re.sub(r"<[^>]*>", "", sanitized)

    # Strip stray Primo subfield delimiters ($$X patterns)
    sanitized = re.sub(r"\$\$[A-Za-z]", "", sanitized)

    # Truncate to max length
    if len(sanitized) > MAX_QUERY_LENGTH:
        sanitized = sanitized[:MAX_QUERY_LENGTH]
        logger.warning("Query truncated to %d characters", MAX_QUERY_LENGTH)

    # Collapse multiple whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Limit boolean operators to prevent Primo query overload
    bool_count = len(re.findall(r'\b(?:AND|OR|NOT)\b', sanitized))
    if bool_count > _MAX_BOOLEAN_OPERATORS:
        # Truncate at the Nth boolean operator
        parts = re.split(r'(\b(?:AND|OR|NOT)\b)', sanitized)
        kept: list[str] = []
        op_count = 0
        for part in parts:
            if re.fullmatch(r'AND|OR|NOT', part):
                op_count += 1
                if op_count > _MAX_BOOLEAN_OPERATORS:
                    break
            kept.append(part)
        sanitized = "".join(kept).strip()
        logger.warning("Query truncated to %d boolean operators", _MAX_BOOLEAN_OPERATORS)

    return sanitized


def validate_api_response(response_data: dict, required_keys: list[str] | None = None) -> tuple[bool, list[str]]:
    """Validate an API response has expected structure.

    Args:
        response_data: The parsed JSON response.
        required_keys: Optional list of required top-level keys.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues: list[str] = []

    if not isinstance(response_data, dict):
        return False, ["Response is not a dictionary"]

    if required_keys:
        for key in required_keys:
            if key not in response_data:
                issues.append(f"Missing required key: {key}")

    return len(issues) == 0, issues


def normalize_encoding(text: str) -> str:
    """Handle character encoding issues in text content.

    Args:
        text: Text that may have encoding issues.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Replace common encoding artifacts
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text
