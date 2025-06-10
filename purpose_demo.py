# -*- coding: utf-8 -*-
"""Minimal demonstration of minellm's purpose.

This repository shows how to build a privacy-aware prompt tuning pipeline.
`purpose_demo.py` focuses on the first step: removing personally
identifiable information from text. This standalone demo avoids heavy
dependencies so it can run anywhere.
"""

import re
from typing import List

PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    "SSN": r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}

def sanitize_text(text: str) -> str:
    """Replace known PII patterns with placeholders."""
    sanitized = text
    for label, pattern in PII_PATTERNS.items():
        sanitized = re.sub(pattern, f"[{label}]", sanitized, flags=re.IGNORECASE)
    return sanitized


def demo() -> None:
    """Run a simple PII sanitization demo."""
    samples: List[str] = [
        "Contact me at bob@example.com for details.",
        "Jane's phone number is 555-123-4567.",
        "SSN: 123-45-6789 should never be public.",
    ]
    for original in samples:
        print("Original:  ", original)
        print("Sanitized: ", sanitize_text(original))
        print("-" * 40)


if __name__ == "__main__":
    demo()
