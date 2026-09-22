"""Differential tests: isojson must produce the same bytes / objects / error
classes as orjson for everything isojson serializes natively."""

import json
import math
import random
import struct

import orjson
import pytest

import isojson

SEEDS = range(20)


def rand_str(r: random.Random) -> str:
    pools = [
        lambda: chr(r.randrange(0x20, 0x7F)),
        lambda: chr(r.randrange(0, 0x20)),  # control chars
        lambda: r.choice('"\\/'),
        lambda: chr(r.randrange(0x80, 0x800)),
        lambda: chr(r.randrange(0x800, 0xD800)),
        lambda: chr(r.randrange(0xE000, 0x10000)),
        lambda: chr(r.randrange(0x10000, 0x110000)),  # non-BMP
    ]
    return "".join(r.choice(pools)() for _ in range(r.randrange(0, 24)))


def rand_float(r: random.Random) -> float:
    while True:
        f = struct.unpack("<d", r.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(f):
            return f


def rand_int(r: random.Random) -> int:
    return r.choice(
        [
            r.randrange(-10, 10),
            r.randrange(-(2**63), 2**63),
            r.randrange(0, 2**64),
            r.choice([0, -1, 2**63 - 1, -(2**63), 2**64 - 1, 2**53, -(2**53), 2**53 - 1]),
        ]
    )


def rand_value(r: random.Random, depth: int = 0):
    kinds = ["int", "float", "str", "bool", "none"]
    if depth < 5:
        kinds += ["list", "tuple", "dict"] * 2
    k = r.choice(kinds)
    if k == "int":
        return rand_int(r)
    if k == "float":
        return r.choice([rand_float(r), r.uniform(-1e6, 1e6), float(r.randrange(-1000, 1000)), r.random()])
    if k == "str":
        return rand_str(r)
    if k == "bool":
        return r.random() < 0.5
    if k == "none":
        return None
    n = r.randrange(0, 6)
    if k == "list":
        return [rand_value(r, depth + 1) for _ in range(n)]
    if k == "tuple":
        return tuple(rand_value(r, depth + 1) for _ in range(n))
    return {rand_str(r): rand_value(r, depth + 1) for _ in range(n)}


OPTION_SETS = [
    None,
    0,
    orjson.OPT_SORT_KEYS,
    orjson.OPT_INDENT_2,
    orjson.OPT_APPEND_NEWLINE,
    orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
]


def test_option_values_match_orjson():
    for name in dir(orjson):
        if name.startswith("OPT_"):
            assert getattr(isojson, name) == getattr(orjson, name), name


@pytest.mark.parametrize("seed", SEEDS)
def test_dumps_random_documents(seed):
    r = random.Random(seed)
    for _ in range(300):
        v = rand_value(r)
        for opt in OPTION_SETS:
            assert isojson.dumps(v, option=opt) == orjson.dumps(v, option=opt), (v, opt)


@pytest.mark.parametrize("seed", SEEDS)
def test_loads_random_documents(seed):
    r = random.Random(seed)
    for _ in range(300):
        v = rand_value(r)
        for doc in (
            orjson.dumps(v),
            orjson.dumps(v, option=orjson.OPT_INDENT_2),
            json.dumps(v, ensure_ascii=True),  # \uXXXX escapes, surrogate pairs
            json.dumps(v, ensure_ascii=False, indent="\t"),
        ):
            assert isojson.loads(doc) == orjson.loads(doc), doc


def test_floats_bit_exact():
    r = random.Random(1234)
    specials = [0.0, -0.0, 1.0, -1.0, 0.1, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 0.0001, 0.00001,
                5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 123456789.123, 1 / 3,
                float("nan"), float("inf"), float("-inf")]
    vals = specials + [rand_float(r) for _ in range(200_000)]
    vals += [r.uniform(-1e20, 1e20) for _ in range(50_000)]
    assert isojson.dumps(vals) == orjson.dumps(vals)
    for f in vals:
        if math.isfinite(f):
            assert isojson.loads(isojson.dumps(f)) == f


def test_subclasses():
    class S(str): pass
    class I(int): pass
    class D(dict): pass
    class L(list): pass
    v = [S("s"), I(3), D(a=1), L([1])]
    assert isojson.dumps(v) == orjson.dumps(v)
    for opt in (orjson.OPT_PASSTHROUGH_SUBCLASS,):
        for x in v:
            with pytest.raises(TypeError):
                orjson.dumps(x, option=opt)
            with pytest.raises(TypeError):
                isojson.dumps(x, option=opt)


def _err(f, *a, **k):
    try:
        f(*a, **k)
    except Exception as e:  # noqa: BLE001
        return type(e), str(e)
    return None


class F(float): pass
class T(tuple): pass


ENCODE_ERRORS = [
    2**64, -(2**63) - 1, {1: 2}, {"a": {1, 2}}, set(), b"x", "\ud800", {"\ud800": 1}, F(1.5), T((1,)), object(),
]


@pytest.mark.parametrize("v", ENCODE_ERRORS, ids=repr)
def test_encode_errors_match(v):
    a, b = _err(isojson.dumps, v), _err(orjson.dumps, v)
    assert a is not None and b is not None
    assert a[0] is b[0] is TypeError
    assert a[1].replace("isojson", "orjson") == b[1]


def test_strict_integer():
    for v in [2**53 - 1, -(2**53) + 1, 2**53, -(2**53), 2**63]:
        assert _err(isojson.dumps, v, option=isojson.OPT_STRICT_INTEGER) == _err(
            orjson.dumps, v, option=orjson.OPT_STRICT_INTEGER
        )


def nest(n):
    x = []
    cur = x
    for _ in range(n - 1):
        m = []
        cur.append(m)
        cur = m
    return x


def test_recursion_limit_matches():
    for n in (253, 254, 255, 256):
        assert _err(isojson.dumps, nest(n)) == _err(orjson.dumps, nest(n)), n
    d = {}
    d["d"] = d
    assert _err(isojson.dumps, d) == _err(orjson.dumps, d)


def test_default():
    class P:
        def __init__(self, x):
            self.x = x

    v = {"p": P(1), "s": {3}}

    def default(o):
        if isinstance(o, P):
            return {"x": o.x}
        if isinstance(o, set):
            return sorted(o)
        raise TypeError

    assert isojson.dumps(v, default=default) == orjson.dumps(v, default=default)
    assert isojson.dumps(v, default) == orjson.dumps(v, default)

    calls = {"iso": 0, "or": 0}

    def forever(key):
        def f(o):
            calls[key] += 1
            return object()
        return f

    assert _err(isojson.dumps, object(), default=forever("iso")) == _err(orjson.dumps, object(), default=forever("or"))
    assert calls["iso"] == calls["or"]

    def boom(o):
        raise ZeroDivisionError("boom")

    with pytest.raises(TypeError) as ei:
        isojson.dumps(object(), default=boom)
    assert isinstance(ei.value.__cause__, ZeroDivisionError)


def test_invalid_opts():
    assert _err(isojson.dumps, 1, option=9999999) == _err(orjson.dumps, 1, option=9999999)
    assert _err(isojson.dumps, 1, option="x")[0] is TypeError
    for opt in (isojson.OPT_NON_STR_KEYS, isojson.OPT_SERIALIZE_NUMPY):
        with pytest.raises(TypeError, match="does not support"):
            isojson.dumps({}, option=opt)


DECODE_OK = [
    b'{"a":1}', "123", "18446744073709551615", "18446744073709551616", "-9223372036854775809",
    "1.5", "1E2", "-0", "0.0", " [1] ", '{"a":1,"a":2}', '"\\ud83d\\ude00"', '"\\/"', "[]", "{}",
    '"a\\u0000b"', "1e-400", "-1e-400", "true", "false", "null",
]


@pytest.mark.parametrize("doc", DECODE_OK, ids=repr)
def test_decode_ok_matches(doc):
    a, b = isojson.loads(doc), orjson.loads(doc)
    assert a == b and type(a) is type(b)


DECODE_BAD = [
    "", " ", "1e400", '"\\ud800"', '"\\udc00"', "[1,]", '{"a":1,}', "NaN", "Infinity", b"\xff", b'"\xed\xa0\x80"',
    '"x"  y', "01", "1.", ".5", "+1", "[", '{"a"}', '{"a":}', "tru", '"\x01"', '"\\x"', '"\\u12"', "[1 2]", "{1:2}",
    b"[\xff]", "\ud800",
]


@pytest.mark.parametrize("doc", DECODE_BAD, ids=repr)
def test_decode_errors(doc):
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(doc)
    with pytest.raises(isojson.JSONDecodeError) as ei:
        isojson.loads(doc)
    assert isinstance(ei.value, json.JSONDecodeError)
    assert isinstance(ei.value, ValueError)


def test_decode_depth_limit():
    for n in (1024, 1025):
        doc = "[" * n + "]" * n
        a = _err(isojson.loads, doc)
        b = _err(orjson.loads, doc)
        assert (a is None) == (b is None), n


def test_decode_input_types():
    for doc in (b"[1]", bytearray(b"[1]"), memoryview(b"[1]"), "[1]"):
        assert isojson.loads(doc) == [1]
    with pytest.raises(isojson.JSONDecodeError):
        isojson.loads(1)


def test_decode_error_position():
    try:
        isojson.loads('{"é": [1, 2,, 3]}')
    except isojson.JSONDecodeError as e:
        assert e.pos == 12 and e.lineno == 1 and e.colno == 13
    else:
        raise AssertionError


def test_loads_float_text_correctly_rounded():
    """The hand-rolled parser (Clinger fast path + Eisel-Lemire + fallback)
    must round every decimal exactly like CPython's float(), which is
    correctly rounded."""
    r = random.Random(99)
    docs = []
    for _ in range(200_000):
        nd = r.randrange(1, 41)
        digits = "".join(r.choice("0123456789") for _ in range(nd)).lstrip("0") or "0"
        cut = r.randrange(0, len(digits) + 1)
        intp, frac = digits[:cut] or "0", digits[cut:]
        s = intp + ("." + frac if frac else ".0")
        if r.random() < 0.6:
            s += r.choice("eE") + r.choice(["", "+", "-"]) + str(r.randrange(0, 330))
        if r.random() < 0.5:
            s = "-" + s
        docs.append(s)
    # halfway / boundary / subnormal cases
    docs += [
        "9007199254740993.0", "9007199254740992.5", "2.2250738585072011e-308",
        "2.2250738585072012e-308", "4.9406564584124654e-324", "2.4703282292062327e-324",
        "2.4703282292062328e-324", "1.7976931348623157e308", "1.7976931348623158e308",
        "0.1000000000000000055511151231257827021181583404541015625",
        "0.1000000000000000055511151231257827021181583404541015624",
        "123456789012345678901234567890e-10", "1e-400", "-1e-400", "0.0e0", "-0.0",
        "7.2057594037927933e16", "3.0540412816652643e-5", "1e23", "8.98846567431158e307",
    ]
    for s in docs:
        try:
            expect = float(s)
        except OverflowError:
            continue
        if expect in (float("inf"), float("-inf")):
            with pytest.raises(isojson.JSONDecodeError):
                isojson.loads(s)
            continue
        got = isojson.loads(s)
        assert type(got) is float and struct.pack("<d", got) == struct.pack("<d", expect), (s, got, expect)


def test_escape_at_every_position():
    """A special character at every offset of strings up to 80 bytes, so each
    position of the 16-byte SIMD block, the 8-byte SWAR step, and the scalar
    tail is hit — for ASCII and multi-byte text."""
    specials = ['"', "\\", "\n", "\x00", "\x1f", "\x7f", "é", "🔥"]
    for fill in ("a", "日"):
        for n in range(0, 81):
            for pos in range(n + 1):
                for sp in specials:
                    s = fill * pos + sp + fill * (n - pos)
                    assert isojson.dumps(s) == orjson.dumps(s), (fill, n, pos, sp)
    dense = "".join(chr(c) for c in range(0, 0x80)) * 50
    assert isojson.dumps(dense) == orjson.dumps(dense)
    assert isojson.dumps({dense: [dense]}) == orjson.dumps({dense: [dense]})


def test_output_growth():
    """Small output first (small size hint), then documents large enough to
    force the output buffer to grow several times, then small again."""
    assert isojson.dumps(1) == b"1"
    for n in (10, 1_000, 100_000, 2_000_000):
        doc = {"k": ["x" * 37, '"quoted"\n' * 3, 1.5, n] * (n // 10 + 1)}
        assert isojson.dumps(doc) == orjson.dumps(doc)
    assert isojson.dumps([1]) == b"[1]"


def test_str_fast_path_is_active():
    """On CPython 3.12-3.14 GIL builds the import-time self-check must pass;
    if it silently failed, isojson would still be correct but slower."""
    import sysconfig
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert isojson.isojson._str_fastpath is True
