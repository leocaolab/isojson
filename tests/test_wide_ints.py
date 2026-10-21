"""Integers outside i64::MIN..=u64::MAX in `dumps`: written in full, as `json.dumps` writes
them (design §1a: the Python API is the ground truth). orjson raises `TypeError` for them."""

import json
import sys

import orjson
import pytest

import isojson

I64_MIN, I64_MAX, U64_MAX = -(2**63), 2**63 - 1, 2**64 - 1

INSIDE = [I64_MIN, I64_MIN + 1, -1, 0, I64_MAX, I64_MAX + 1, U64_MAX]
OUTSIDE = [U64_MAX + 1, U64_MAX + 2, I64_MIN - 1, I64_MIN - 2, -U64_MAX - 1, 2**200, -(2**200)]


def stdlib(x):
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode()


@pytest.mark.parametrize("v", INSIDE, ids=str)
def test_inside_64_bits_still_matches_orjson(v):
    assert isojson.dumps(v) == orjson.dumps(v) == stdlib(v)


@pytest.mark.parametrize("v", OUTSIDE, ids=str)
def test_outside_64_bits_is_written_in_full(v):
    with pytest.raises(TypeError, match="Integer exceeds 64-bit range"):
        orjson.dumps(v)
    assert isojson.dumps(v) == stdlib(v)
    assert json.loads(isojson.dumps(v)) == v


def test_inside_containers_and_keys_order():
    doc = {"n": 2**64, "m": [1, -(2**63) - 1, {"x": 10**40}], "s": "after"}
    assert isojson.dumps(doc) == stdlib(doc)
    assert isojson.dumps(doc, option=isojson.OPT_SORT_KEYS) == json.dumps(
        doc, separators=(",", ":"), sort_keys=True
    ).encode()


class Shouty(int):
    def __str__(self):
        return "not the value"

    __repr__ = __str__


def test_int_subclass_writes_its_value_not_its_str():
    assert isojson.dumps([Shouty(2**70)]) == b"[" + str(2**70).encode() + b"]"


def test_strict_integer_still_raises():
    for v in (2**64, -(2**63) - 1):
        with pytest.raises(TypeError, match="Integer exceeds 53-bit range"):
            isojson.dumps(v, option=isojson.OPT_STRICT_INTEGER)


def test_digit_limit_raises_like_json_dumps():
    limit = sys.get_int_max_str_digits()
    at_limit = 10 ** (limit - 1)
    assert isojson.dumps(at_limit) == stdlib(at_limit)
    over = 10**limit
    with pytest.raises(ValueError) as ours:
        isojson.dumps({"n": over})
    with pytest.raises(ValueError) as theirs:
        json.dumps({"n": over})
    assert str(ours.value) == str(theirs.value)
