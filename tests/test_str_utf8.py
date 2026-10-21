"""`dumps` writes a `str` whose UTF-8 isn't cached by encoding its code points
directly: the bytes match stdlib `json` and orjson, surrogates still raise, and
the caller's `str` is left without a UTF-8 copy of its own.

Every string here is built fresh, so isojson is the first to see it: an
orjson call (or an earlier test) on the same object would cache its UTF-8."""

import json
import random
import sys

import orjson
import pytest

import isojson

SURROGATES = "str is not valid UTF-8: surrogates not allowed"


def fresh(s: str) -> str:
    """A new `str` object equal to `s`, with no UTF-8 cached."""
    return "".join(list(s))


def stdlib(obj, **kw) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), **kw).encode()


CASES = {
    "latin1": "café, naïve, ÿ",
    "latin1_escapes": 'é"\\\n\t\x00\x1f\x7f',
    "ucs2": "多解释器安全的 JSON 序列化",
    "ucs2_escapes": '日\n本"語\\\x01',
    "ucs4": "🔥🐍x𝄞",
    "ucs4_escapes": "🔥\n\x02\"🐍",
    "bmp_edges": "".join(map(chr, (0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF))),
    "astral_edges": "\U00010000\U0010ffff",
    "long": ("日本語\n" * 5000) + ("é" * 3000) + "🔥" * 1000,
}


@pytest.mark.parametrize("name", CASES)
def test_value_bytes_match_stdlib_and_orjson(name):
    s = fresh(CASES[name])
    out = isojson.dumps(s)
    assert out == stdlib(s)
    assert out == orjson.dumps(s)


@pytest.mark.parametrize("name", CASES)
def test_leaves_no_utf8_copy_on_the_str(name):
    s = fresh(CASES[name])
    before = sys.getsizeof(s)
    isojson.dumps([s, {s: s}])
    isojson.dumps({s: 1, "a": 2}, option=isojson.OPT_SORT_KEYS)
    assert sys.getsizeof(s) == before


def test_str_subclass():
    class S(str):
        pass

    for text in CASES.values():
        s = S(fresh(text))
        assert isojson.dumps(s) == stdlib(str(s))
        assert isojson.dumps({s: s}) == stdlib({str(s): str(s)})


def test_keys_in_order_and_indented():
    d = {fresh(k): fresh(v) for k, v in CASES.items()} | {fresh(v): k for k, v in CASES.items()}
    assert isojson.dumps(d) == stdlib(d)
    assert isojson.dumps(d, option=isojson.OPT_INDENT_2) == json.dumps(d, ensure_ascii=False, indent=2).encode()


def test_sorted_keys_mix_ascii_cached_and_uncached():
    cached = fresh("缓存的键")
    orjson.dumps(cached)  # caches its UTF-8
    r = random.Random(7)
    pool = [chr(c) for c in (0x41, 0x7A, 0xE9, 0xFF, 0x100, 0x7FF, 0x4E00, 0xFFFF, 0x10000, 0x1F525)]
    keys = {fresh("".join(r.choice(pool) for _ in range(r.randrange(0, 5)))) for _ in range(300)}
    d = {k: i for i, k in enumerate(keys)} | {cached: -1, "plain": -2}
    out = isojson.dumps(d, option=isojson.OPT_SORT_KEYS)
    assert out == stdlib(d, sort_keys=True)


@pytest.mark.parametrize(
    "obj",
    [
        fresh("\ud800"),
        fresh("a\n\udfff"),  # after an escape grew the reservation
        fresh("日本" + chr(0xD83D) + chr(0xDE00)),  # a surrogate pair is still two surrogates
        fresh("🔥\udc00"),
        [fresh("ok"), fresh("é\ud800")],
        {fresh("\ud800"): 1},
        {fresh("é"): fresh("\ud800")},
    ],
)
def test_surrogates_raise(obj):
    with pytest.raises(TypeError, match=SURROGATES):
        isojson.dumps(obj)


def test_surrogate_key_raises_when_sorting():
    with pytest.raises(TypeError, match=SURROGATES):
        isojson.dumps({fresh("é\ud800"): 1, "a": 2}, option=isojson.OPT_SORT_KEYS)


def test_sorted_keys_report_the_first_bad_key_in_dict_order():
    with pytest.raises(TypeError, match=SURROGATES):
        isojson.dumps({fresh("\ud800"): 1, 5: 2}, option=isojson.OPT_SORT_KEYS)
    with pytest.raises(TypeError, match="Dict key must be str"):
        isojson.dumps({5: 1, fresh("\ud800"): 2}, option=isojson.OPT_SORT_KEYS)


def test_fast_path_survives_a_second_import():
    """Each interpreter's import re-runs the layout self-check; by then a probe
    singleton like "é" has its UTF-8 cached, which must not turn it off."""
    import subprocess
    import sysconfig
    import textwrap

    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("the layout fast path is GIL-build only")
    code = textwrap.dedent(
        """
        from concurrent import interpreters
        import isojson
        interpreters.create().exec("import isojson")
        print(isojson.isojson._str_fastpath)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"
