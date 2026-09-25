"""E2E-4: one regression per design §1b row where isojson intentionally
differs from orjson 3.12.0. Each proves both halves:

(a) orjson 3.12.0 fails: its wrong bytes or message, or a child killed by a
    signal (POSIX);
(b) the reference: `expected_text`, or for raising rows the Python API's own
    error;
(c) isojson == (b).

Datetime rows (DV-1, 3, 4a, 4b, 12, 15, 17) here; the numpy rows (DV-5…11, 13, 14, 16) arrive with numpy.
"""

import datetime as dt
import subprocess
import sys
import textwrap
import zoneinfo

import orjson
import pytest
import pytz

import isojson
from oracle.python_api import expected_text

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
    # (a): an invented zero offset (spelled `Z` under UTC_Z on 3.14, `+00:00`
    # on 3.12/3.13 — orjson's own version-dependent path)
    assert o.endswith(b'+00:00"') or (opts & UTC_Z and o.endswith(b'Z"')), o
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
