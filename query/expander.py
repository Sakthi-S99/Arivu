"""
Arivu RAG — Query expansion.
Expands domain-specific acronyms so retrieval matches full terms.
Pure Python, no dependencies, no re-ingest.

ACRONYMS below is a generic starter set. To customize for your own corpus
without publishing your domain vocabulary, drop a JSON file of the same
shape at `config/acronyms.local.json` (gitignored) — entries there are
merged in and take priority over the built-in defaults.
"""

import json
import os

# Acronym → expansion. Query keeps original + appends expansions.
ACRONYMS = {
    "api": "Application Programming Interface",
    "sdk": "Software Development Kit",
    "sso": "Single Sign-On",
    "ootb": "out of the box",
    "faq": "Frequently Asked Questions",
    "rca": "Root Cause Analysis",
}

_LOCAL_OVERRIDE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "acronyms.local.json",
)
if os.path.exists(_LOCAL_OVERRIDE):
    with open(_LOCAL_OVERRIDE, "r") as _f:
        ACRONYMS = {**ACRONYMS, **json.load(_f)}


def expand_query(query: str) -> str:
    """
    Append expansions for any acronym found as a whole word.
    Original query preserved — expansion only adds signal, never replaces.
    """
    words = query.lower().replace("?", " ").replace(",", " ").split()
    additions = []
    for w in words:
        exp = ACRONYMS.get(w)
        if exp and exp.lower() not in query.lower():
            additions.append(exp)

    if not additions:
        return query
    return f"{query} ({'; '.join(additions)})"
