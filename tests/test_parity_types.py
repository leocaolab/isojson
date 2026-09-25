"""E2E-2: byte parity with orjson 3.12.0 on the natively serialized types.

Bytes, and exception type and message, equal orjson's, excluding inputs a
design §1b row names (by id) and the §1b "not bugs" list. "Without default"
means the argument is omitted, not `default=None` (§1b).

Datetime part here; the numpy part arrives with numpy.
"""

import datetime as dt
import itertools
import random
import zoneinfo

import orjson
import pytest
import pytz

import isojson

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
