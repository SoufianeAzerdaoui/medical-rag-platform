from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from comment_cardinality import (  # noqa: E402
    CommentCardinality,
    _wants_all_comments_listing,
    _wants_latest_comment,
    _wants_single_comment,
    detect_comment_cardinality,
    resolve_effective_limit,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("liste une commentaire", True),
        ("liste une seule commentaire", True),
        ("montre un commentaire", True),
        ("affiche un commentaire", True),
        ("donne moi un commentaire", True),
        ("le commentaire", True),
        ("ce commentaire", True),
        ("le dernier commentaire", True),
        ("tous les commentaires", False),
        ("liste les commentaires", False),
    ],
)
def test_wants_single_comment(query: str, expected: bool) -> None:
    assert _wants_single_comment(query) is expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("liste les commentaires", True),
        ("tous les commentaires", True),
        ("montre les commentaires", True),
        ("liste une commentaire", False),
        ("liste une seule commentaire", False),
        ("le commentaire", False),
    ],
)
def test_wants_all_comments_listing(query: str, expected: bool) -> None:
    assert _wants_all_comments_listing(query) is expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("commentaire du dernier rapport", True),
        ("le commentaire le plus recent", True),
        ("liste une commentaire", False),
        ("tous les commentaires", False),
    ],
)
def test_wants_latest_comment(query: str, expected: bool) -> None:
    assert _wants_latest_comment(query) is expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("liste une commentaire", CommentCardinality.SINGLE),
        ("liste une seule commentaire", CommentCardinality.SINGLE),
        ("montre un commentaire", CommentCardinality.SINGLE),
        ("le commentaire de troponine", CommentCardinality.SINGLE),
        ("commentaire du dernier rapport", CommentCardinality.LATEST),
        ("rapport le plus recent commentaire", CommentCardinality.LATEST),
        ("tous les commentaires", CommentCardinality.ALL),
        ("liste les commentaires", CommentCardinality.ALL),
        ("montre les commentaires", CommentCardinality.ALL),
        ("donne moi les résultats biologiques", CommentCardinality.UNSPECIFIED),
    ],
)
def test_detect_comment_cardinality(query: str, expected: str) -> None:
    assert detect_comment_cardinality(query) == expected


@pytest.mark.parametrize(
    ("query", "max_display_results", "expected"),
    [
        ("liste une commentaire", 3, 1),
        ("liste les commentaires", 5, 5),
        ("commentaire dernier rapport", 3, 1),
        ("résultats du rapport", 3, 3),
    ],
)
def test_resolve_effective_limit(query: str, max_display_results: int, expected: int) -> None:
    assert resolve_effective_limit(query, max_display_results=max_display_results) == expected


def test_mutual_exclusivity() -> None:
    queries = [
        "liste une commentaire",
        "liste une seule commentaire",
        "montre un commentaire",
        "affiche une note",
    ]
    for query in queries:
        assert not (_wants_single_comment(query) and _wants_all_comments_listing(query))

