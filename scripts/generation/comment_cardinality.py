from __future__ import annotations

import re
from typing import Callable

from query_understanding import norm_text

_COMMENT_WORDS: frozenset[str] = frozenset(
    {
        "commentaire",
        "note",
    }
)

_ALL_SIGNALS: frozenset[str] = frozenset(
    {
        "tous les commentaires",
        "toutes les notes",
        "l ensemble des commentaires",
        "chaque commentaire",
    }
)

_SINGULAR_SIGNALS: frozenset[str] = frozenset(
    {
        "une seule",
        "un seul",
        "seulement un",
        "seulement une",
        "juste un",
        "juste une",
        "1 commentaire",
        "1 note",
        "single comment",
        "le commentaire",
        "ce commentaire",
        "le dernier commentaire",
    }
)

_LISTING_VERBS: frozenset[str] = frozenset(
    {
        "liste",
        "lister",
        "montre",
        "affiche",
        "donne moi",
        "retourne",
        "voir",
    }
)

_LATEST_SIGNALS: frozenset[str] = frozenset(
    {
        "dernier rapport",
        "rapport le plus recent",
        "plus recent",
        "derniere date",
        "date la plus recente",
        "dernier bilan",
        "dernier resultat disponible",
    }
)

_REGEX_SINGLE_VERB = r"(?:liste|lister|montre|affiche|donne moi|retourne|voir)"
_REGEX_SINGLE_ARTICLE = r"(?:un|une)"
_REGEX_SINGLE_OPTIONAL_ONLY = r"(?:\s+(?:seul|seule))?"
_REGEX_SINGLE_COMMENT_WORD = r"(?:commentaire|note)"
_REGEX_SINGLE_PATTERN = (
    rf"\b{_REGEX_SINGLE_VERB}\b\s+"
    rf"{_REGEX_SINGLE_ARTICLE}"
    rf"{_REGEX_SINGLE_OPTIONAL_ONLY}\s+"
    rf"{_REGEX_SINGLE_COMMENT_WORD}\b"
)
_SINGLE_VERB_PATTERN = re.compile(_REGEX_SINGLE_PATTERN)

_REGEX_ALL_LISTING_VERB = r"(?:liste|lister|montre|affiche|donne moi|retourne|voir)"
_REGEX_ALL_OPTIONAL_ARTICLE = r"(?:les|des|tous les|toutes les)"
_REGEX_ALL_PLURAL_COMMENT = r"(?:commentaires|notes)"
_REGEX_ALL_PATTERN = (
    rf"\b{_REGEX_ALL_LISTING_VERB}\b\s+"
    rf"(?:{_REGEX_ALL_OPTIONAL_ARTICLE}\s+)?"
    rf"{_REGEX_ALL_PLURAL_COMMENT}\b"
)
_ALL_LISTING_PATTERN = re.compile(_REGEX_ALL_PATTERN)

_COUNT_SINGLE_VERB = "single_verb"
_COUNT_SINGLE_SIGNAL = "single_signal"
_COUNT_ALL_SIGNAL = "all_signal"
_COUNT_ALL_PATTERN = "all_pattern"
_COUNT_LATEST_SIGNAL = "latest_signal"
_COUNT_UNSPECIFIED = "unspecified"


class CommentCardinality:
    """
    Declarative priority system for comment cardinality resolution.

    Priority order (strict, immutable):
      1. SINGLE      — user wants exactly one comment
      2. LATEST      — user wants the most recent one
      3. ALL         — user wants all comments
      4. UNSPECIFIED — no signal → system default applies

    Rule: a signal of lower rank NEVER overrides a signal of higher rank.
    SINGLE > LATEST > ALL > UNSPECIFIED.

    To change priority: edit _RANK, not the resolution logic.
    To add a new cardinality: add constant + rank entry + rule in _CARDINALITY_RULES.
    """

    SINGLE = "single"
    LATEST = "latest"
    ALL = "all"
    UNSPECIFIED = "unspecified"

    _RANK: dict[str, int] = {
        SINGLE: 1,
        LATEST: 2,
        ALL: 3,
        UNSPECIFIED: 4,
    }

    @classmethod
    def rank(cls, value: str) -> int:
        return cls._RANK.get(value, 99)


def _wants_single_comment(query: str) -> bool:
    """
    Returns True if the user wants exactly ONE comment.

    Signals detected:
    - Explicit singular markers from _SINGULAR_SIGNALS
    - Pattern: [listing_verb] + [un/une] + [seul/seule?] + [comment_word]
    - NEVER returns True if _ALL_SIGNALS present (guard first)
    """

    qn = norm_text(query)
    if any(signal in qn for signal in _ALL_SIGNALS):
        return False
    if any(signal in qn for signal in _SINGULAR_SIGNALS):
        return True
    return bool(_SINGLE_VERB_PATTERN.search(qn))


def _wants_latest_comment(query: str) -> bool:
    """
    Returns True if the user wants the most recent comment specifically.

    Signals detected:
    - Any term from _LATEST_SIGNALS
    - NEVER returns True if _SINGULAR_SIGNALS also present (SINGLE wins)
    """

    qn = norm_text(query)
    if any(signal in qn for signal in _SINGULAR_SIGNALS):
        return False
    return any(signal in qn for signal in _LATEST_SIGNALS)


def _wants_all_comments_listing(query: str) -> bool:
    """
    Returns True if the user wants multiple or all comments.

    Signals detected:
    - Explicit _ALL_SIGNALS markers
    - [listing_verb] + plural comment word (without singular article)

    INVARIANT: if _wants_single_comment(query) is True,
               this function MUST return False.
               This is enforced by calling _wants_single_comment
               as a guard at the start.
    """

    if _wants_single_comment(query):
        return False
    qn = norm_text(query)
    if any(signal in qn for signal in _ALL_SIGNALS):
        return True
    return bool(_ALL_LISTING_PATTERN.search(qn))


_CARDINALITY_RULES: list[tuple[Callable[[str], bool], str]] = [
    (_wants_single_comment, CommentCardinality.SINGLE),
    (_wants_latest_comment, CommentCardinality.LATEST),
    (_wants_all_comments_listing, CommentCardinality.ALL),
]


def detect_comment_cardinality(query: str) -> str:
    """
    Resolves the comment cardinality of a user query.

    Evaluates all rules in _CARDINALITY_RULES and returns
    the highest-priority (lowest rank) matched cardinality.

    Priority: SINGLE > LATEST > ALL > UNSPECIFIED
    Defined in CommentCardinality.

    Args:
        query: raw user query string (French)

    Returns:
        CommentCardinality.SINGLE | LATEST | ALL | UNSPECIFIED
    """

    matches = [card for detector, card in _CARDINALITY_RULES if detector(query)]
    if not matches:
        return CommentCardinality.UNSPECIFIED
    return min(matches, key=CommentCardinality.rank)


_LIMIT_BY_CARDINALITY: dict[str, int | None] = {
    CommentCardinality.SINGLE: 1,
    CommentCardinality.LATEST: 1,
    CommentCardinality.ALL: None,
    CommentCardinality.UNSPECIFIED: None,
}


def resolve_effective_limit(
    query: str,
    *,
    max_display_results: int = 3,
) -> int:
    """
    Returns the effective result limit based on detected cardinality.

    Mapping:
      SINGLE      → 1
      LATEST      → 1
      ALL         → max_display_results
      UNSPECIFIED → max_display_results

    Used directly in run_generation() to replace all if/elif cardinality checks.
    """

    cardinality = detect_comment_cardinality(query)
    fixed_limit = _LIMIT_BY_CARDINALITY.get(cardinality)
    return fixed_limit if fixed_limit is not None else max_display_results

