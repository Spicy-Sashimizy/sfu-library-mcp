"""Hot/cold section vocabulary + classifier.

Ported from scripts/eval_sectioned_index.py (the measured 1M-doc eval). The live
corpus has no concepts metadata, so sections are derived from title+abstract
text. Assignment is PRIORITY-ORDERED DISJOINT: a doc lands in the first section
whose vocabulary matches it; `other` catches everything unmatched — mirroring
the OpenSearch `multi_match` OR-of-terms matchers used in the eval (standard
analyzer = lowercase word tokens, no stemming).
"""

from __future__ import annotations

import re

# (name, vocabulary) in priority order — keep in sync with the measured eval.
SECTIONS: list[tuple[str, str]] = [
    ("social_sciences",
     "political politics policy sociology economic economics social society "
     "education psychology law legal governance democracy election cultural "
     "anthropology history philosophy geography business management finance"),
    ("med_bio",
     "patient clinical disease cancer tumor cell gene protein medical health "
     "therapy treatment diagnosis biology biological neural brain immune "
     "infection molecular genetic physiology pharmacology"),
    ("phys_eng",
     "quantum physics material chemical chemistry molecular energy optical "
     "thermal mechanical engineering electron magnetic photon nanoparticle "
     "semiconductor fluid mechanics structural"),
    ("cs_math",
     "algorithm software computational machine learning network data model "
     "computer programming optimization mathematical theorem matrix graph "
     "statistical simulation"),
]

OTHER = "other"
SECTION_NAMES: list[str] = [name for name, _ in SECTIONS] + [OTHER]

# Persona -> HOT sections (kept live with abstract sidecar); everything else is
# COLD (packed zstd-19 artifact archives, no local abstracts).
PERSONAS: dict[str, list[str]] = {
    "political_science": ["social_sciences"],
    "computer_science": ["cs_math"],
    "health_science": ["med_bio"],
    "interdisciplinary_cogsci": ["med_bio", "cs_math", "social_sciences"],
    # Testbed persona: everything live (no cold packing).
    "all_hot": SECTION_NAMES,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SECTION_VOCABS: list[tuple[str, frozenset[str]]] = [
    (name, frozenset(vocab.split())) for name, vocab in SECTIONS
]


def classify_doc(title: str | None, abstract: str | None) -> str:
    """Return the section for a doc: first section whose vocab intersects the
    lowercase word tokens of title+abstract, else `other`."""
    tokens = set(_TOKEN_RE.findall(f"{title or ''} {abstract or ''}".lower()))
    for name, vocab in _SECTION_VOCABS:
        if tokens & vocab:
            return name
    return OTHER


def classify_query(query: str) -> str:
    """Best-effort section hint for a QUERY (used for cold-section unpack
    hints / routing telemetry — never for filtering results)."""
    return classify_doc(query, None)
