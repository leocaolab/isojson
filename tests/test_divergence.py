"""E2E-4: one regression per design §1b row where isojson intentionally
differs from orjson 3.12.0. Each proves both halves:

(a) orjson 3.12.0 fails: its wrong bytes or message, or a child killed by a
    signal (POSIX);
(b) the reference: `expected_text`, or for raising rows the Python API's own
    error;
(c) isojson == (b).

Datetime rows: DV-1, 3, 4a, 4b, 12, 15, 17. numpy rows: DV-5…11, 13, 14, 16.
"""

import datetime as dt
import re
import subprocess
import sys
import textwrap
import zoneinfo

import json
import warnings

import numpy as np
import orjson
import pytest
import pytz

import isojson
from oracle.python_api import NoAnswer, expected_text

assert orjson.__version__ == "3.12.0", "the §1b rows were measured on orjson 3.12.0"

NAIVE_UTC = isojson.OPT_NAIVE_UTC
UTC_Z = isojson.OPT_UTC_Z
OMIT = isojson.OPT_OMIT_MICROSECONDS


class Fixed(dt.tzinfo):
    """A tzinfo whose utcoffset() returns whatever it was given."""

    def __init__(self, offset):
        self.offset = offset

    def utcoffset(self, _):
        return self.offset

    def dst(self, _):
        return None

    def tzname(self, _):
        return None


class Raising(dt.tzinfo):
    def utcoffset(self, _):
        raise ValueError("no offset today")


def died_in_child(code):
    """Run `code` in a fresh interpreter; True if a signal killed it."""
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, timeout=60)
    return r.returncode < 0


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="death by signal is POSIX")


# ---- DV-1: offsets with seconds or microseconds ------------------------------

def _tz(**kw):
    return dt.timezone(dt.timedelta(**kw))


DV1 = [
    (dt.datetime(2026, 1, 1, tzinfo=_tz(hours=5, minutes=59, seconds=30)), b'"2026-01-01T00:00:00+05:60"'),
    (dt.datetime(2026, 1, 1, tzinfo=_tz(hours=5, microseconds=1)), b'"2026-01-01T00:00:00+05:00"'),
    (dt.datetime(2026, 1, 1, tzinfo=_tz(seconds=-1)), b'"2026-01-01T00:00:00-00:00"'),
    (dt.datetime(2026, 1, 1, tzinfo=_tz(hours=-24, microseconds=1)), b'"2026-01-01T00:00:00+00:00"'),
    (dt.datetime(1850, 1, 1, tzinfo=zoneinfo.ZoneInfo("America/New_York")), b'"1850-01-01T00:00:00-04:56"'),
]


@pytest.mark.parametrize(("x", "orjson_bytes"), DV1, ids=lambda v: str(v)[-40:])
def test_dv1_offset_seconds(x, orjson_bytes):
    assert orjson.dumps(x) == orjson_bytes  # (a)
    ref = expected_text(x)  # (b)
    assert ref != orjson_bytes
    assert isojson.dumps(x) == ref  # (c)


# ---- DV-3: pytz after arithmetic, not normalized -----------------------------

def test_dv3_pytz_unnormalized():
    x = pytz.timezone("America/New_York").localize(dt.datetime(2026, 3, 7, 12)) + dt.timedelta(days=2)
    assert orjson.dumps(x) == b'"2026-03-09T12:00:00-04:00"'  # (a)
    ref = expected_text(x)
    assert ref == b'"2026-03-09T12:00:00-05:00"'  # (b): x.utcoffset()
    assert isojson.dumps(x) == ref  # (c)


# ---- DV-4a: utcoffset() returns None -----------------------------------------

@pytest.mark.parametrize("opts", [0, UTC_Z, OMIT, UTC_Z | OMIT])
def test_dv4a_none_offset_is_naive(opts):
    x = dt.datetime(2026, 1, 1, 1, 2, 3, 4, tzinfo=Fixed(None))
    o = orjson.dumps(x, option=opts)
    # (a): an invented offset. orjson reads the `None` as a timedelta, so the
    # value depends on the platform: `+00:00` / `Z` on macOS and x86_64
    # Linux, garbage such as `+18:12` on aarch64 Linux (measured in CI).
    assert re.search(rb'([+-]\d\d:\d\d|Z)"$', o), o
    ref = expected_text(x, opts)
    assert not ref.endswith(b'Z"') and b"+" not in ref  # (b): naive to Python
    assert isojson.dumps(x, option=opts) == ref  # (c)


# ---- DV-4b: utcoffset() raises (datetime) / is invalid (time) ----------------

@posix_only
def test_dv4b_datetime_raising_utcoffset():
    assert died_in_child(
        """
        import datetime as dt, orjson
        class Raising(dt.tzinfo):
            def utcoffset(self, _): raise ValueError("x")
        orjson.dumps(dt.datetime(2026, 1, 1, tzinfo=Raising()))
        """
    )  # (a)
    x = dt.datetime(2026, 1, 1, tzinfo=Raising())
    with pytest.raises(ValueError) as ref:  # (b): the Python API raises
        x.utcoffset()
    with pytest.raises(TypeError) as got:  # (c)
        isojson.dumps(x)
    assert str(got.value) == f"datetime.utcoffset() raised ValueError: {ref.value}"
    assert type(got.value.__cause__) is ValueError
    assert str(got.value.__cause__) == str(ref.value)


@pytest.mark.parametrize("offset", [5, dt.timedelta(hours=25)], ids=["int", "25h"])
def test_dv4b_time_invalid_offset(offset):
    x = dt.time(1, tzinfo=Fixed(offset))
    with pytest.raises(TypeError, match="datetime.time must not have tzinfo set"):  # (a)
        orjson.dumps(x)
    with pytest.raises((TypeError, ValueError)) as ref:  # (b)
        x.utcoffset()
    with pytest.raises(TypeError) as got:  # (c)
        isojson.dumps(x)
    assert str(got.value) == f"time.utcoffset() raised {type(ref.value).__name__}: {ref.value}"
    assert type(got.value.__cause__) is type(ref.value)


# ---- DV-12: time with tzinfo -------------------------------------------------

@pytest.mark.parametrize(
    ("x", "want"),
    [
        (dt.time(1, tzinfo=dt.timezone.utc), b'"01:00:00+00:00"'),
        (dt.time(1, 2, 3, 4, tzinfo=_tz(hours=-5)), b'"01:02:03.000004-05:00"'),
        (dt.time(1, tzinfo=zoneinfo.ZoneInfo("Europe/Paris")), b'"01:00:00"'),
    ],
)
def test_dv12_time_with_tzinfo(x, want):
    with pytest.raises(TypeError, match="datetime.time must not have tzinfo set"):  # (a)
        orjson.dumps(x)
    assert expected_text(x) == want  # (b)
    assert isojson.dumps(x) == want  # (c)


# ---- DV-15: utcoffset() returns a non-timedelta or >= 24h --------------------

@pytest.mark.parametrize(
    ("offset", "orjson_bytes"),
    [
        (5, b'"2026-01-01T00:00:00+00:00"'),
        (dt.timedelta(hours=24), b'"2026-01-01T00:00:00+00:00"'),
        (dt.timedelta(hours=25), b'"2026-01-01T00:00:00+01:00"'),
        (dt.timedelta(hours=-25), b'"2026-01-01T00:00:00+23:00"'),
        ("+05:00", None),  # read as a timedelta: garbage that varies per run
    ],
    ids=["int", "24h", "25h", "-25h", "str"],
)
def test_dv15_invalid_datetime_offset(offset, orjson_bytes):
    x = dt.datetime(2026, 1, 1, tzinfo=Fixed(offset))
    with pytest.raises((TypeError, ValueError)) as ref:  # (b): the Python API raises
        x.utcoffset()
    o = orjson.dumps(x)  # (a): no exception, an invented offset
    if orjson_bytes is None:
        assert o.startswith(b'"2026-01-01T00:00:00')
    else:
        assert o == orjson_bytes
    with pytest.raises(TypeError) as got:  # (c)
        isojson.dumps(x)
    assert str(got.value) == f"datetime.utcoffset() raised {type(ref.value).__name__}: {ref.value}"
    assert type(got.value.__cause__) is type(ref.value)


def test_timedelta_subclass_offset_is_valid():
    """Not a divergence (§1b DV-15 note): a timedelta subclass is valid in both."""

    class TD(dt.timedelta):
        pass

    x = dt.datetime(2026, 1, 1, tzinfo=Fixed(TD(hours=3)))
    assert isojson.dumps(x) == expected_text(x) == orjson.dumps(x) == b'"2026-01-01T00:00:00+03:00"'


# ---- DV-17: a time's five-digit microsecond ----------------------------------

@pytest.mark.parametrize("us", [10_000, 75_652, 99_999])
def test_dv17_time_microsecond_leading_zero(us):
    x = dt.time(0, 0, 1, us)
    assert orjson.dumps(x) == f'"00:00:01.{us}"'.encode()  # (a): 0.75652 s, not 0.075652 s
    ref = expected_text(x)
    assert ref == f'"00:00:01.0{us}"'.encode()  # (b)
    assert isojson.dumps(x) == ref  # (c)


# ---- numpy rows --------------------------------------------------------------

NUMPY = isojson.OPT_SERIALIZE_NUMPY
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    NaT = np.datetime64("NaT")
# numpy 2.5 warns on every generic-unit datetime64, NaT included (§15.3)
pytestmark = pytest.mark.filterwarnings("ignore:The 'generic' unit:DeprecationWarning")


def o_err(x, **kw):
    try:
        orjson.dumps(x, option=NUMPY, **kw)
    except TypeError as e:
        return str(e)
    raise AssertionError("orjson did not raise")


def declined(x, message):
    """(b)/(c) for a declined object: without `default` the FR-7 message
    (plus the note); with one, `default` receives the whole object."""
    with pytest.raises(TypeError) as e:
        isojson.dumps(x, option=NUMPY)
    assert str(e.value) == message
    assert e.value.__notes__ and e.value.__notes__[0].startswith("isojson can't write")
    seen = []
    assert isojson.dumps({"x": x}, option=NUMPY, default=lambda a: seen.append(a) or "D") == b'{"x":"D"}'
    assert len(seen) == 1 and seen[0] is x


def both_forms(unit, v):
    """The value as a 1-D array element and as a scalar."""
    a = np.array([v], dtype="i8").view(f"M8[{unit}]") if unit else np.array([v], dtype="i8").view("M8")
    return a, a[0]


def native_equals_reference(x):
    """(b)+(c) for a written array or scalar: each element's reference."""
    xs = [x] if x.ndim == 0 else list(x)
    ref = b"[" + b",".join(expected_text(e) for e in xs) + b"]" if x.ndim else expected_text(x)
    assert isojson.dumps(x, option=NUMPY) == ref
    return ref


def test_dv5_nat_ns():
    for x in both_forms("ns", np.iinfo("i8").min):
        assert orjson.dumps(x, option=NUMPY).strip(b"[]") == b'"1677-09-21T00:12:43.145224"'  # (a)
        assert native_equals_reference(x).strip(b"[]") == b"null"


@pytest.mark.parametrize("unit", ["W", "D", "h", "m"])
def test_dv6_nat_wraps(unit):
    for x in both_forms(unit, np.iinfo("i8").min):
        assert orjson.dumps(x, option=NUMPY).strip(b"[]") == b'"1970-01-01T00:00:00"'  # (a)
        assert native_equals_reference(x).strip(b"[]") == b"null"


@pytest.mark.parametrize("unit", ["Y", "M", "s", "ms", "us", None])
def test_dv7_nat_raises(unit):
    for x in both_forms(unit, np.iinfo("i8").min):
        msg = o_err(x)  # (a)
        assert msg.startswith("unrepresentable") if unit else msg == "unsupported numpy.datetime64 unit: NaT"
        assert native_equals_reference(x).strip(b"[]") == b"null"


@posix_only
@pytest.mark.parametrize(
    "code",
    [
        "np.array([1], dtype='M8[10ms]')",
        "np.array([1], dtype='M8[2D]')",
        "np.datetime64(1, '10ms')",
        "np.array(['1969-12'], dtype='M8[M]')",
    ],
)
def test_dv8_dv9_orjson_dies(code):
    assert died_in_child(
        f"import numpy as np, orjson\norjson.dumps({code}, option=orjson.OPT_SERIALIZE_NUMPY)"
    )  # (a)
    x = eval(code, {"np": np})  # noqa: S307
    native_equals_reference(x)  # (b), (c)


def test_dv8_values():
    assert isojson.dumps(np.array([1], dtype="M8[10ms]"), option=NUMPY) == b'["1970-01-01T00:00:00.010000"]'
    assert isojson.dumps(np.array([1], dtype="M8[2D]"), option=NUMPY) == b'["1970-01-03T00:00:00"]'


def test_dv9_months_before_1970():
    x = np.array(["1969-11"], dtype="M8[M]")
    assert o_err(x) == "unrepresentable numpy.datetime64: -2 months"  # (a)
    assert native_equals_reference(x) == b'["1969-11-01T00:00:00"]'
    assert isojson.dumps(np.array(["1969-12"], dtype="M8[M]"), option=NUMPY) == b'["1969-12-01T00:00:00"]'


@pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us"])
def test_dv10_last_day(unit):
    x = np.array(["9999-12-31"], dtype=f"M8[{unit}]")
    assert o_err(x).startswith("unrepresentable numpy.datetime64: ")  # (a)
    assert native_equals_reference(x) == b'["9999-12-31T00:00:00"]'


def test_dv11_overflow_declined():
    x = np.array([307445734561825861], dtype="M8[m]")
    assert orjson.dumps(x, option=NUMPY) == b'["1970-01-01T00:00:44"]'  # (a): wrapped
    with pytest.raises(NoAnswer):  # (b): no answer in 0000–9999
        expected_text(x[0])
    declined(x, "unrepresentable numpy.datetime64: 307445734561825861 minutes")  # (c)


def test_dv13_generic_value_declined():
    for x in both_forms(None, 5):
        assert o_err(x) == "unsupported numpy.datetime64 unit: NaT"  # (a): misnames the value
        with pytest.raises(NoAnswer):  # (b)
            expected_text(x if x.ndim == 0 else x[0])
        declined(x, "unsupported numpy.datetime64 unit: generic")  # (c)


DV14 = [
    np.array([["NaT"], ["2026-01-01"]], dtype="M8[s]"),
    np.array([["2026-01-01"], ["10000-01-01"]], dtype="M8[s]"),
    np.array([[5], [6]], dtype="i8").view("M8[ps]"),
    np.array([[NaT], [NaT]]),
    np.array([[5], [6]], dtype="i8").view("M8"),
]


@pytest.mark.parametrize("x", DV14, ids=["nat", "out-of-range", "ps", "generic-nat", "generic-value"])
def test_dv14_malformed_json(x):
    for kw in ({}, {"default": str}):
        with pytest.raises(json.JSONDecodeError):  # (a): malformed, even with default
            json.loads(orjson.dumps({"x": x, "y": 1}, option=NUMPY, **kw))
    flat = [e for row in x for e in row]
    try:
        ref = b"[" + b",".join(b"[" + expected_text(e) + b"]" for e in flat) + b"]"  # (b)
    except NoAnswer:
        # (c) declined per FR-7 (e)/(f): default gets the whole array
        assert isojson.dumps({"x": x, "y": 1}, option=NUMPY, default=lambda a: "D") == b'{"x":"D","y":1}'
        return
    out = isojson.dumps({"x": x, "y": 1}, option=NUMPY)  # (c)
    assert out == b'{"x":' + ref + b',"y":1}'
    json.loads(out)


@pytest.mark.parametrize("unit", ["ps", "fs", "as"])
def test_dv16_sub_microsecond_units(unit):
    x = np.array([5, -1, np.iinfo("i8").max, np.iinfo("i8").min], dtype="i8").view(f"M8[{unit}]")
    names = {"ps": "picoseconds", "fs": "femtoseconds", "as": "attoseconds"}
    assert o_err(x) == f"unsupported numpy.datetime64 unit: {names[unit]}"  # (a)
    assert o_err(x, default=str) == f"unsupported numpy.datetime64 unit: {names[unit]}"  # even with default
    native_equals_reference(x)  # (b), (c)
    native_equals_reference(x[0])
