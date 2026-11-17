"""Versioned prompt templates. Change the file name (v1 -> v2) when a prompt changes meaning."""

from __future__ import annotations

from importlib.resources import files

# v2: the action follows whether the FAQ answers the ticket, not its category (see README).
ASSESS_PROMPT = "assess_v2.md"
VERIFY_PROMPT = "verify_v1.md"


def load_prompt(name: str) -> str:
    return files(__package__).joinpath(name).read_text(encoding="utf-8")
