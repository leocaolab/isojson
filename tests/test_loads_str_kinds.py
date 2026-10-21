"""`loads` builds each non-ASCII `str` at the kind CPython itself picks.

The equal value is checked against stdlib `json`; the kind (1, 2 or 4 bytes per
character) against `sys.getsizeof`, which differs between kinds for the same
length.
"""

import json
import sys

import pytest

import isojson

# the highest code point of each kind, and the lowest of the next
EDGES = ["\x7f", "\x80", "\xff", "Ā", "߿", "ࠀ", "￿", "\U00010000", "\U0010ffff"]

LENGTHS = [1, 7, 8, 9, 15, 16, 17, 64, 1000]


def cases():
    for ch in EDGES:
        for n in LENGTHS:
            yield ch * n
            yield "a" * n + ch
            yield ch + "a" * n
    for lo in EDGES:
        for hi in EDGES:
            yield lo + "mixed" + hi
            yield hi * 3 + lo * 5
    yield "é日本🔥x" * 50
    yield "éĀ\U00010000"


CASES = list(dict.fromkeys(cases()))


def same_str(got, want):
    # a one-character non-ASCII Latin-1 string must be CPython's shared singleton
    shared = len(want) == 1 and 0x80 <= ord(want) < 0x100
    return (
        got == want
        and type(got) is str
        and sys.getsizeof(got) == sys.getsizeof(want)
        and (got is want) == shared
    )


@pytest.mark.parametrize("s", CASES, ids=lambda s: f"{len(s)}:{max(map(ord, s)):x}")
def test_value_str_has_cpythons_kind(s):
    doc = json.dumps(s, ensure_ascii=False).encode()
    assert same_str(isojson.loads(doc), json.loads(doc))


@pytest.mark.parametrize("s", CASES, ids=lambda s: f"{len(s)}:{max(map(ord, s)):x}")
def test_key_str_has_cpythons_kind(s):
    doc = json.dumps({s: 1}, ensure_ascii=False).encode()
    (got,) = isojson.loads(doc)
    (want,) = json.loads(doc)
    assert same_str(got, want)


@pytest.mark.parametrize("s", ["é", "Ā", "￿", "\U0001f525", "aéb\U0001f525c"])
def test_escaped_str_has_cpythons_kind(s):
    doc = json.dumps(s, ensure_ascii=True).encode()
    assert same_str(isojson.loads(doc), json.loads(doc))
