"""E2E-5: isojson imports nothing, in a fresh strict own-GIL
sub-interpreter (`tests/interp/strict.py`)."""

import textwrap

from interp import strict


def run_fresh(code):
    i = strict.make()
    try:
        strict.run(i, textwrap.dedent(code))
    finally:
        strict.destroy(i)


def test_import_and_dumps_import_nothing():
    run_fresh(
        """
        import sys
        watched = ("numpy", "datetime", "_datetime", "_pydatetime")
        before = {m for m in watched if m in sys.modules}
        import isojson
        isojson.dumps([1, "x", {"k": (None, 2.5)}])
        isojson.dumps(object(), default=str)
        isojson.dumps([1], option=isojson.OPT_NAIVE_UTC | isojson.OPT_UTC_Z)
        isojson.dumps([1, "x"], option=isojson.OPT_SERIALIZE_NUMPY)
        isojson.dumps(object(), default=str, option=isojson.OPT_SERIALIZE_NUMPY)
        added = {m for m in watched if m in sys.modules} - before
        assert not added, added
        """
    )


def test_non_json_goes_to_default_with_datetime_and_no_numpy():
    run_fresh(
        """
        import sys, datetime, isojson
        assert "numpy" not in sys.modules
        class X: pass
        assert isojson.dumps([X()], default=lambda o: "x") == b'["x"]'
        try:
            isojson.dumps(X())
        except TypeError as e:
            assert str(e) == "Type is not JSON serializable: X", e
        else:
            raise AssertionError("no error")
        """
    )


def test_datetime_stays_native_without_the_datetime_module():
    """FR-2: the lookup is `sys.modules["_datetime"]`, not `datetime`."""
    run_fresh(
        """
        import sys, datetime, isojson
        assert "_datetime" in sys.modules
        x = datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.timezone.utc)
        del sys.modules["datetime"]
        assert isojson.dumps(x) == b'"2026-01-01T12:00:00+00:00"'
        """
    )
