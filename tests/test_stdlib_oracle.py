"""E2E-8: isojson against the Python API itself (design §1a).

`output == expected_text(x, opts)` for every input, and the output parses
back with `fromisoformat` to the option-adjusted source with its
`utcoffset()`. Covers every combination of NAIVE_UTC × UTC_Z ×
OMIT_MICROSECONDS (PASSTHROUGH_DATETIME is E2E-2's).

Then datetime64: every unit × multipliers 1/2/3/7/10, sampled over the
unit's representable i64 range plus both i64 extremes, MIN+1 and the
0000/10000 crossings, where the output equals `expected_text` or, where that
raises `NoAnswer`, the FR-7 (f) decline, byte for byte. Floats by value.
"""

import datetime as dt
import itertools
import random
import zoneinfo

import numpy as np
import pytest
import pytz

import isojson
import test_divergence as e2e4
import test_parity_types as e2e2
from oracle.python_api import NoAnswer, adjusted, expected_text, float_matches

OPTS = [
    sum(c)
    for n in range(4)
    for c in itertools.combinations(
        [isojson.OPT_NAIVE_UTC, isojson.OPT_UTC_Z, isojson.OPT_OMIT_MICROSECONDS], n
    )
]

ZONES = [zoneinfo.ZoneInfo(k) for k in ("America/New_York", "Europe/London", "Asia/Kolkata",
                                        "Australia/Lord_Howe", "Pacific/Chatham", "Africa/Monrovia")]
PYTZ = [pytz.timezone(k) for k in ("America/New_York", "Asia/Shanghai", "Europe/Amsterdam",
                                   "Asia/Kathmandu")]


class TD(dt.timedelta):
    pass


def check(x, opts):
    out = isojson.dumps(x, option=opts)
    assert out == expected_text(x, opts), (x, opts)
    text = out[1:-1].decode()
    a = adjusted(x, opts)
    back = type(x).fromisoformat(text)
    if type(x) is dt.date:
        assert back == a
        return
    assert back.replace(tzinfo=None) == a.replace(tzinfo=None)
    assert back.utcoffset() == a.utcoffset()


def fixed_inputs():
    xs = [p.values[0] for p in e2e2.DATETIMES]
    xs += e2e2.DATES + e2e2.TIMES
    xs += [p[0] for p in e2e4.DV1]
    xs += [
        pytz.timezone("America/New_York").localize(dt.datetime(2026, 3, 7, 12)) + dt.timedelta(days=2),
        dt.datetime(2026, 1, 1, 1, 2, 3, 4, tzinfo=e2e4.Fixed(None)),  # DV-4a
        dt.time(1, 2, 3, 4, tzinfo=e2e4.Fixed(None)),
        dt.datetime(2026, 1, 1, tzinfo=e2e4.Fixed(TD(hours=3, microseconds=7))),
        dt.time(1, tzinfo=dt.timezone.utc),
        dt.time(0, 0, 1, 75_652),  # DV-17
    ]
    return xs


@pytest.mark.parametrize("opts", OPTS)
def test_fixed_inputs(opts):
    for x in fixed_inputs():
        check(x, opts)


def random_offset(r):
    us = r.randrange(-(24 * 3600 * 10**6) + 1, 24 * 3600 * 10**6)
    return r.choice([
        dt.timedelta(microseconds=us),
        dt.timedelta(seconds=us // 10**6),
        dt.timedelta(minutes=us // (60 * 10**6)),
        dt.timedelta(0),
    ])


def random_input(r):
    y = r.randint(1, 9999)
    wall = (y, r.randint(1, 12), r.randint(1, 28), r.randrange(24), r.randrange(60),
            r.randrange(60), r.choice([0, r.randrange(1_000_000), r.randrange(10_000, 100_000)]))
    kind = r.randrange(7)
    if kind == 0:
        return dt.datetime(*wall)
    if kind == 1:
        return dt.datetime(*wall, tzinfo=dt.timezone(random_offset(r)))
    if kind == 2:
        # zoneinfo across DST and before standard time (second offsets)
        return dt.datetime(*wall, tzinfo=r.choice(ZONES), fold=r.randrange(2))
    if kind == 3:
        yp = r.randint(1800, 2100)
        return r.choice(PYTZ).localize(dt.datetime(yp, *wall[1:]))
    if kind == 4:
        return dt.datetime(*wall, tzinfo=e2e4.Fixed(r.choice([None, random_offset(r), TD(seconds=r.randrange(-86399, 86400))])))
    if kind == 5:
        return dt.time(*wall[3:], tzinfo=r.choice([None, dt.timezone(random_offset(r)), e2e4.Fixed(None)]))
    return dt.date(*wall[:3])


@pytest.mark.parametrize("seed", range(20))
def test_random_inputs(seed):
    r = random.Random(seed)
    for _ in range(500):
        check(random_input(r), r.choice(OPTS))


# ---- datetime64 --------------------------------------------------------------

NUMPY = isojson.OPT_SERIALIZE_NUMPY
UNITS = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"]
WORDS = {"Y": "years", "M": "months", "W": "weeks", "D": "days", "h": "hours", "m": "minutes",
         "s": "seconds", "ms": "milliseconds", "us": "microseconds", "ns": "nanoseconds",
         "ps": "picoseconds", "fs": "femtoseconds", "as": "attoseconds"}
# units per µs as (num, den) for range sampling; Y/M by months
_US = {"W": (7 * 86_400 * 10**6, 1), "D": (86_400 * 10**6, 1), "h": (3_600 * 10**6, 1),
       "m": (60 * 10**6, 1), "s": (10**6, 1), "ms": (10**3, 1), "us": (1, 1),
       "ns": (1, 10**3), "ps": (1, 10**6), "fs": (1, 10**9), "as": (1, 10**12)}
I64 = np.iinfo("i8")


def representable(unit, mult):
    """The v range whose meaning lies in 0000-01-01 … 9999-12-31 (approx.)."""
    if unit == "Y":
        return -1970 // mult, 8029 // mult
    if unit == "M":
        return -1970 * 12 // mult, (8030 * 12 - 1) // mult
    num, den = _US[unit]
    lo = int(np.datetime64("0000-01-01", "D").view("i8")) * 86_400 * 10**6
    hi = int(np.datetime64("10000-01-01", "D").view("i8")) * 86_400 * 10**6
    return max(I64.min + 1, lo * den // (num * mult)), min(I64.max, hi * den // (num * mult))


def dt64_values(unit, mult, r):
    lo, hi = representable(unit, mult)
    vs = {I64.min, I64.min + 1, I64.max, I64.max - 1, 0, 1, -1}
    for edge in (lo, hi):
        vs.update(v for v in range(edge - 2, edge + 3) if I64.min <= v <= I64.max)
    vs.update(r.randint(lo, hi) for _ in range(40))
    vs.update(r.randint(I64.min + 1, I64.max) for _ in range(5))
    return sorted(vs)


def check_dt64(v, unit, mult, opts):
    x = np.array([v], dtype="i8").view(f"M8[{mult}{unit}]")[0]
    try:
        want = expected_text(x, opts)
    except NoAnswer:
        msg = f"unrepresentable numpy.datetime64: {v} {WORDS[unit]}" + (f" × {mult}" if mult != 1 else "")
        with pytest.raises(TypeError) as e:
            isojson.dumps(x, option=NUMPY | opts)
        assert str(e.value) == msg
        return False
    assert isojson.dumps(x, option=NUMPY | opts) == want, (v, unit, mult, opts)
    assert isojson.dumps(np.array([x]), option=NUMPY | opts) == b"[" + want + b"]"
    return True


@pytest.mark.parametrize("mult", [1, 2, 3, 7, 10])
@pytest.mark.parametrize("unit", UNITS)
def test_datetime64_against_the_exact_meaning(unit, mult):
    r = random.Random(f"{unit}{mult}")
    written = declined = 0
    for v in dt64_values(unit, mult, r):
        for opts in OPTS:
            if check_dt64(v, unit, mult, opts):
                written += 1
            else:
                declined += 1
    lo, hi = representable(unit, mult)
    assert written
    # where the representable range is narrower than i64, declines were hit;
    # otherwise (ns ×1–3, ps, fs, as) every i64 value has an answer
    assert bool(declined) == (lo > I64.min + 1 or hi < I64.max), (declined, lo, hi)


def test_datetime64_year_and_month_at_i64_max():
    for unit in ("Y", "M"):
        assert not check_dt64(I64.max, unit, 1, 0)  # numpy's astype wraps these (§1a)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_floats_by_value(dtype):
    r = np.random.default_rng(7)
    raw = r.integers(0, 2**63, 5000, dtype=np.uint64)
    bits = {"f2": raw.astype(np.uint16), "f4": raw.astype(np.uint32), "f8": raw}[dtype]
    a = bits.view(dtype)
    texts = isojson.dumps(a, option=NUMPY)[1:-1].split(b",")
    for t, x in zip(texts, a, strict=True):
        assert float_matches(t, x), (t, x)
    for x in a[:200]:
        assert float_matches(isojson.dumps(x, option=NUMPY), x)
