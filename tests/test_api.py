import pytest
from medjev.api import Query, Candidate


def test_ambiguous_candidate_schema_rejected():
    with pytest.raises(ValueError):
        Query("q", "claim", candidates=(Candidate("a", "same"), Candidate("b", "same")))
    with pytest.raises(ValueError):
        Query("q", "claim", candidates=(Candidate("a", "one"), Candidate("a", "two")))
    with pytest.raises(ValueError):
        Query("q", " ")


def test_runtime_candidate_order_preserved():
    q = Query("q", "claim", candidates=(Candidate("z", "not documented"), Candidate("a", "documented")))
    assert [c.id for c in q.candidates] == ["z", "a"]
