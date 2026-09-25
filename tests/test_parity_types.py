"""E2E-2: byte parity with orjson 3.12.0 on the natively serialized types.

Bytes, and exception type and message, equal orjson's, excluding inputs a
design §1b row names (by id) and the §1b "not bugs" list. "Without default"
means the argument is omitted, not `default=None` (§1b).

Datetime part, then numpy (with and without `default`, compact and
`OPT_INDENT_2`).
"""

import datetime as dt
import os
import itertools
import random
import zoneinfo

import numpy as np
import orjson
import pytest
import pytz

import isojson

# the latest-orjson CI job (non-blocking) sets ISOJSON_ORJSON_ANY=1 to see what changed
if os.environ.get("ISOJSON_ORJSON_ANY") != "1":
    assert orjson.__version__ == "3.12.0"

DT_OPTS = [
    isojson.OPT_NAIVE_UTC,
    isojson.OPT_UTC_Z,
    isojson.OPT_OMIT_MICROSECONDS,
    isojson.OPT_PASSTHROUGH_DATETIME,
]
# every combination of the four datetime options
ALL_DT_OPTS = [sum(c) for n in range(5) for c in itertools.combinations(DT_OPTS, n)]

NY = zoneinfo.ZoneInfo("America/New_York")
SH = pytz.timezone("Asia/Shanghai")

# whole-minute offsets only: others are DV-1
TZS = {
    "naive": None,
    "utc": dt.timezone.utc,
    "+8": dt.timezone(dt.timedelta(hours=8)),
    "-5": dt.timezone(dt.timedelta(hours=-5)),
    "zoneinfo": NY,
    "pytz": SH,  # applied with localize(), so normalized (unnormalized is DV-3)
}


def aware(tz, *args):
    naive = dt.datetime(*args)
    if tz is None:
        return naive
    if isinstance(tz, dt.tzinfo) and hasattr(tz, "localize"):
        return tz.localize(naive)
    return naive.replace(tzinfo=tz)


WALLS = [
    (2026, 9, 24, 1, 2, 3, 4),
    (2026, 9, 24, 23, 59, 59, 999_999),
    (2026, 3, 8, 3, 0, 0, 0),  # NY spring-forward day
    (2026, 11, 1, 1, 30, 0, 0),  # NY fall-back hour
    (1, 1, 1, 0, 0, 0, 0),
    (999, 12, 31, 12, 0, 0, 500_000),
    (9999, 12, 30, 0, 0, 0, 0),
    (1970, 1, 1, 0, 0, 0, 0),
]

DATETIMES = [
    pytest.param(aware(tz, *w), id=f"{name}-{w[0]}-{w[1]}-{w[3]}h")
    for name, tz in TZS.items()
    for w in WALLS
    # zoneinfo/pytz before standard time have second offsets: DV-1
    if not (name in ("zoneinfo", "pytz") and w[0] < 1901)
]

DATES = [dt.date(1, 1, 1), dt.date(999, 1, 2), dt.date(2026, 9, 24), dt.date(9999, 12, 31)]
# naive only: a time with tzinfo is DV-12; no five-digit microsecond (DV-17)
TIMES = [dt.time(0), dt.time(1, 2, 3), dt.time(1, 2, 3, 4), dt.time(23, 59, 59, 999_999)]


def both(x, **kw):
    """(isojson result, orjson result), where a result is bytes or
    (exception type, message)."""
    out = []
    for lib in (isojson, orjson):
        try:
            out.append(lib.dumps(x, **kw))
        except Exception as e:  # noqa: BLE001
            out.append((type(e), str(e)))
    return out


@pytest.mark.parametrize("opts", ALL_DT_OPTS)
@pytest.mark.parametrize("x", DATETIMES)
def test_datetime(x, opts):
    a, b = both(x, option=opts)
    assert a == b


@pytest.mark.parametrize("opts", ALL_DT_OPTS)
@pytest.mark.parametrize("x", DATES + TIMES, ids=str)
def test_date_and_time(x, opts):
    a, b = both(x, option=opts)
    assert a == b


@pytest.mark.parametrize("opts", [0, isojson.OPT_PASSTHROUGH_DATETIME])
@pytest.mark.parametrize("x", [dt.datetime(2026, 1, 1), dt.date(2026, 1, 1), dt.time(1)], ids=str)
def test_default_receives_passthrough(x, opts):
    seen = []

    def default(o):
        seen.append(o)
        return "D"

    a, b = both([x], default=default, option=opts)
    assert a == b
    assert seen == ([x, x] if opts else [])


def test_datetime_subclass_goes_to_default():
    class D(dt.datetime):
        pass

    x = D(2026, 1, 1, 12)
    assert both(x) == [(TypeError, "Type is not JSON serializable: D")] * 2
    a, b = both({"k": x}, default=lambda o: o.isoformat())
    assert a == b == b'{"k":"2026-01-01T12:00:00"}'


# ---- random documents --------------------------------------------------------

def rand_moment(r):
    tz = r.choice(list(TZS.values()))
    y = r.randint(1901, 9998) if tz in (NY, SH) else r.randint(1, 9999)
    x = aware(tz, y, r.randint(1, 12), r.randint(1, 28), r.randrange(24), r.randrange(60),
              r.randrange(60), r.choice([0, r.randrange(1_000_000)]))
    t = x.time().replace(tzinfo=None)
    if dv17(t):
        t = t.replace(microsecond=0)
    return r.choice([x, x.date(), t])


def dv17(x):
    """Design §1b DV-17's predicate: orjson drops the leading zero of a
    five-digit microsecond in a `time`."""
    return type(x) is dt.time and 10_000 <= x.microsecond <= 99_999


def rand_doc(r, depth=0):
    """A random document mixing the JSON builtins with datetime objects.
    New for this test; `rand_value` in test_parity.py is left as it is."""
    k = r.randrange(9 if depth < 4 else 5)
    if k == 0:
        return r.randint(-(2**63), 2**63 - 1)
    if k == 1:
        return r.random() * 10 ** r.randint(-5, 20)
    if k == 2:
        return "".join(r.choice("ab\"\\é\n𝄞") for _ in range(r.randrange(6)))
    if k == 3:
        return r.choice([None, True, False])
    if k == 4:
        return rand_moment(r)
    if k in (5, 6):
        return [rand_doc(r, depth + 1) for _ in range(r.randrange(5))]
    if k == 7:
        return tuple(rand_doc(r, depth + 1) for _ in range(r.randrange(4)))
    return {f"k{r.randrange(50)}": rand_doc(r, depth + 1) for _ in range(r.randrange(5))}


@pytest.mark.parametrize("seed", range(300))
def test_random_documents(seed):
    r = random.Random(seed)
    doc = rand_doc(r)
    opts = r.choice(ALL_DT_OPTS) | r.choice([0, isojson.OPT_INDENT_2]) | r.choice([0, isojson.OPT_SORT_KEYS])
    kw = {"option": opts}
    if r.random() < 0.5:
        kw["default"] = repr
    a, b = both(doc, **kw)
    assert a == b


# ---- numpy -------------------------------------------------------------------

NUMPY = isojson.OPT_SERIALIZE_NUMPY

# values orjson and isojson both write; no NaT (DV-5/6/7), no multiplied or
# sub-µs unit (DV-8, DV-16), nothing before 1970 in M8[M] (DV-9), nothing at
# or after 9999-12-30T22:00, where orjson's range ends (DV-10), nothing out of range (DV-11, DV-14)
DT64_VALUES = {
    "M8[ns]": ["1970-01-01", "2026-09-24T01:02:03.123456789", "1677-09-22", "2262-04-10"],
    "M8[us]": ["0001-01-01T00:00:00.000001", "2026-09-24T01:02:03.000004", "9999-12-30T21:59:59"],
    "M8[s]": ["1000-01-01T00:00:01", "2026-09-24T12:00", "1969-12-31T23:59:59"],
    "M8[D]": ["0001-01-01", "1969-01-01", "9999-12-30"],
    "M8[M]": ["1970-01", "2026-09", "9999-12"],
    "M8[Y]": ["1970", "2026", "9999"],
}

WRITTEN = ["f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "?"]


def array_of(dtype, shape):
    n = int(np.prod(shape))
    if dtype.startswith("M8"):
        vals = DT64_VALUES[dtype]
        return np.array([vals[i % len(vals)] for i in range(n)], dtype=dtype).reshape(shape)
    if dtype == "?":
        return (np.arange(n) % 3 == 0).reshape(shape)
    base = np.array([0, 1, -2, 3.5, 100, -7.25, 0.1, 1e3], dtype="f8")
    v = np.resize(base, n)
    if dtype[0] == "u":
        v = np.abs(v)
    if dtype[0] in "iu":
        v = v.astype("i8")
    return v.astype(dtype).reshape(shape)


SHAPES = [(5,), (2, 3), (2, 1, 3), (0,), (2, 0), (0, 3)]


def _numpy_cases():
    for d in WRITTEN + list(DT64_VALUES):
        for shape in SHAPES:
            yield pytest.param(array_of(d, shape), id=f"{d}-{shape}")
        a = array_of(d, (4, 3))
        yield pytest.param(a.T, id=f"{d}-transposed")
        yield pytest.param(a[:, ::2], id=f"{d}-strided")
        yield pytest.param(a[0, 0].reshape(()), id=f"{d}-0d")
        yield pytest.param(a.reshape(-1)[0], id=f"{d}-scalar")
    for bad in (np.array(["ab"]), np.array([1j]), np.array([None, 1]), np.array([b"x"])):
        yield pytest.param(bad, id=f"unsupported-{bad.dtype.str}")
    for x in (np.complex128(1), np.longdouble(1.5), np.str_("s"), np.bytes_(b"b")):
        yield pytest.param(x, id=f"scalar-{type(x).__name__}")
    yield pytest.param(np.array([1, 2], dtype=">i4"), id="not-native")  # without default only


def excluded(x, with_default):
    """The §1b "not bugs" list: for 1-D arrays and scalars orjson raises even
    with a `default` on non-native-endian arrays (isojson calls `default`)."""
    return with_default and isinstance(x, np.ndarray) and not x.dtype.isnative


@pytest.mark.parametrize("opts", [NUMPY, NUMPY | isojson.OPT_INDENT_2], ids=["compact", "indent"])
@pytest.mark.parametrize("with_default", [False, True], ids=["no-default", "default"])
@pytest.mark.parametrize("x", list(_numpy_cases()))
def test_numpy(x, with_default, opts):
    if excluded(x, with_default):
        pytest.skip("§1b not-bugs: orjson raises with default for non-native arrays")
    kw = {"default": lambda o: {"d": type(o).__name__}} if with_default else {}
    a, b = both({"k": [x, 1]}, option=opts, **kw)
    assert a == b


@pytest.mark.parametrize("opts", [0, isojson.OPT_NAIVE_UTC, isojson.OPT_NAIVE_UTC | isojson.OPT_UTC_Z,
                                  isojson.OPT_OMIT_MICROSECONDS, isojson.OPT_PASSTHROUGH_DATETIME])
@pytest.mark.parametrize("dtype", list(DT64_VALUES))
def test_datetime64_options(dtype, opts):
    """FR-9: the datetime options apply to datetime64; PASSTHROUGH_DATETIME
    does not (as orjson)."""
    x = array_of(dtype, (3,))
    a, b = both([x, x[0]], option=NUMPY | opts)
    assert a == b


def test_without_the_option_numpy_goes_to_default():
    x = np.arange(3)
    assert both(x, default=lambda o: o.tolist()) == [b"[0,1,2]"] * 2
    a, b = both(np.float64(1.5))
    assert a == b and a[0] is TypeError


def rand_numpy(r):
    d = r.choice(WRITTEN + list(DT64_VALUES))
    shape = r.choice([(r.randrange(4),), (r.randrange(3), r.randrange(3))])
    a = array_of(d, shape)
    return a if r.random() < 0.7 or a.size == 0 else a.reshape(-1)[0]


@pytest.mark.parametrize("seed", range(300))
def test_random_documents_with_numpy(seed):
    r = random.Random(seed)
    doc = rand_doc(r)
    doc = [doc, rand_numpy(r), {"n": [rand_numpy(r), rand_doc(r)]}]
    opts = NUMPY | r.choice(ALL_DT_OPTS) | r.choice([0, isojson.OPT_INDENT_2]) | r.choice([0, isojson.OPT_SORT_KEYS])
    kw = {"default": repr} if r.random() < 0.5 else {}
    a, b = both(doc, option=opts, **kw)
    assert a == b
