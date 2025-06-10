# -*- coding: utf-8 -*-
"""Streamlit interface for the minellm demo."""

import sys

missing_deps = []
try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None
    missing_deps.append("streamlit")

from purpose_demo import sanitize_text


def main() -> None:
    """Launch a simple UI to sanitize text."""
    st.set_page_config(page_title="minellm Privacy Demo")
    st.title("minellm Privacy Demo")
    st.write(
        "Enter some text below and click *Sanitize* to remove any personally "
        "identifiable information."
    )
    text = st.text_area("Input text", height=200)
    if st.button("Sanitize"):
        sanitized = sanitize_text(text)
        st.text_area("Sanitized text", sanitized, height=200)


if __name__ == "__main__":
    if missing_deps:
        print(
            "This script requires additional dependencies: "
            + ", ".join(missing_deps)
        )
        print("Install them with `pip install streamlit`." )
    else:
        main()
