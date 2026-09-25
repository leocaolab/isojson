"""The correctness oracle: what Python's own API says isojson must write.

Design §1a is the rule; this module is the only place tests get "correct"
bytes from. orjson is compared against only in the explicit parity tests.

`expected_text(x, opts)` returns the JSON bytes for `x`, or raises
`NoAnswer(reason)` where the Python API gives no answer in isojson's format
(the test then asserts the decline instead). It raises `TypeError` for inputs
it has no rule for, so a test can never pass by accident on an unhandled type.
"""

import datetime as _dt

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


def expected_text(x, opts=0):
    t = type(x)
    if t is _dt.datetime:
        a = adjusted(x, opts)
        return _quoted(_with_utc_z(a.isoformat(), a.utcoffset(), opts))
    if t is _dt.date:
        return _quoted(x.isoformat())
    if t is _dt.time:
        # NAIVE_UTC and UTC_Z don't apply to time (as orjson)
        return _quoted(adjusted(x, opts & OPT_OMIT_MICROSECONDS).isoformat())
    raise TypeError(f"expected_text has no rule for {t.__qualname__}")
