"""The correctness oracle: what Python's own API says isojson must write.

Design §1a is the rule; this module is the only place tests get "correct"
bytes from. orjson is compared against only in the explicit parity tests.

`expected_text(x, opts)` returns the JSON bytes for `x`, or raises
`NoAnswer(reason)` where the Python API gives no answer in isojson's format
(the test then asserts the decline instead). It raises `TypeError` for inputs
it has no rule for, so a test can never pass by accident on an unhandled type.
"""

import datetime as _dt
import math

OPT_NAIVE_UTC = 2
OPT_OMIT_MICROSECONDS = 8
OPT_UTC_Z = 128

_ZERO = _dt.timedelta(0)


class NoAnswer(Exception):
    """The Python API gives no answer in isojson's format for this input."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _quoted(s):
    return b'"' + s.encode() + b'"'


def adjusted(x, opts=0):
    """`x` after the options are applied to the object (design §1a), before
    it is rendered. Tests also use it to round-trip `fromisoformat`."""
    t = type(x)
    if t is _dt.datetime:
        # naive <=> utcoffset() is None, including a tzinfo returning None (DV-4a)
        if opts & OPT_NAIVE_UTC and x.utcoffset() is None:
            x = x.replace(tzinfo=_dt.timezone.utc)
        if opts & OPT_OMIT_MICROSECONDS:
            x = x.replace(microsecond=0)
        return x
    if t is _dt.time:
        if opts & OPT_OMIT_MICROSECONDS:
            x = x.replace(microsecond=0)
        return x
    if t is _dt.date:
        return x
    raise TypeError(f"expected_text has no rule for {t.__qualname__}")


def _with_utc_z(text, offset, opts):
    if opts & OPT_UTC_Z and offset == _ZERO:
        assert text.endswith("+00:00"), text
        return text[: -len("+00:00")] + "Z"
    return text


# ---- numpy (imported lazily: E2E-5 asserts isojson itself never imports it)

# µs per unit as (numerator, denominator), from numpy's unit meaning
_US = {
    "W": (7 * 86_400_000_000, 1), "D": (86_400_000_000, 1), "h": (3_600_000_000, 1),
    "m": (60_000_000, 1), "s": (1_000_000, 1), "ms": (1_000, 1), "us": (1, 1),
    "ns": (1, 10**3), "ps": (1, 10**6), "fs": (1, 10**9), "as": (1, 10**12),
}


def _np():
    import numpy

    return numpy


def _us_bounds():
    np = _np()
    day = 86_400_000_000
    lo = int(np.datetime64("0000-01-01", "D").view("i8")) * day
    hi = int(np.datetime64("10000-01-01", "D").view("i8")) * day
    return lo, hi


def _datetime64_text(x, opts):
    """Design §1a: the meaning numpy's API defines, computed exactly —
    `np.datetime_data` gives (unit, mult), the value is `v × mult` units since
    1970-01-01T00:00, sub-µs floored to µs, Y/M by floor divmod. Rendered by
    numpy's own formatter at µs, then by the `datetime` rules."""
    np = _np()
    if np.isnat(x):
        return b"null"
    unit, mult = np.datetime_data(x.dtype)
    if unit == "generic":
        raise NoAnswer("a generic-unit value has no unit ratio, so no meaning")
    n = int(x.view("i8")) * mult
    if unit in ("Y", "M"):
        y, mo = (1970 + n, 1) if unit == "Y" else (1970 + n // 12, n % 12 + 1)
        if not 0 <= y <= 9999:
            raise NoAnswer(f"year {y} is outside 0000–9999")
        us = int(np.datetime64(f"{y:04d}-{mo:02d}", "M").astype("M8[us]").view("i8"))
    else:
        num, den = _US[unit]
        us = (n * num) // den
        lo, hi = _us_bounds()
        if not lo <= us < hi:
            raise NoAnswer(f"{us} µs is outside 0000-01-01 … 9999-12-31")
    text = np.datetime_as_string(np.datetime64(us, "us"))  # YYYY-MM-DDTHH:MM:SS.ffffff
    base, frac = text.split(".")
    if frac != "000000" and not opts & OPT_OMIT_MICROSECONDS:
        base += "." + frac
    if opts & OPT_NAIVE_UTC:
        base += "Z" if opts & OPT_UTC_Z else "+00:00"
    return _quoted(base)


def float_matches(text, x):
    """§1a for floats: only the value is the reference. float64:
    `float(text) == x`; float32/float16: `np.float32(text)` equals
    `np.float32(x)` bitwise. Non-finite values are `null`."""
    np = _np()
    if not math.isfinite(float(x)):
        return text == b"null"
    if type(x) in (float, np.float64):
        return float(text) == float(x) and math.copysign(1, float(text)) == math.copysign(1, float(x))
    a = np.float32(text.decode()).view(np.uint32)
    b = np.float32(x).view(np.uint32)
    return a == b


def expected_text(x, opts=0):
    t = type(x)
    if t.__module__ == "numpy":
        np = _np()
        if t is np.datetime64:
            return _datetime64_text(x, opts)
        if t is np.bool_ or np.issubdtype(t, np.integer):
            v = x.item()
            return (b"true" if v else b"false") if t is np.bool_ else str(v).encode()
        raise TypeError(f"expected_text has no rule for {t.__qualname__} (floats: float_matches)")
    if t is _dt.datetime:
        a = adjusted(x, opts)
        return _quoted(_with_utc_z(a.isoformat(), a.utcoffset(), opts))
    if t is _dt.date:
        return _quoted(x.isoformat())
    if t is _dt.time:
        # NAIVE_UTC and UTC_Z don't apply to time (as orjson)
        return _quoted(adjusted(x, opts & OPT_OMIT_MICROSECONDS).isoformat())
    raise TypeError(f"expected_text has no rule for {t.__qualname__}")
