"""E2E-8: isojson against the Python API itself (design §1a).

`output == expected_text(x, opts)` for every input, and the output parses
back with `fromisoformat` to the option-adjusted source with its
`utcoffset()`. Covers every combination of NAIVE_UTC × UTC_Z ×
OMIT_MICROSECONDS (PASSTHROUGH_DATETIME is E2E-2's).

Datetime part here; the datetime64 part arrives with numpy.
"""

import datetime as dt
import itertools
import random
import zoneinfo

import pytest
import pytz

import isojson
import test_divergence as e2e4
import test_parity_types as e2e2
from oracle.python_api import adjusted, expected_text

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
