"""E2E-5 (isojson imports nothing) and E2E-7 (3.12's pure-Python datetime),
each in a fresh strict own-GIL sub-interpreter (`tests/interp/strict.py`)."""

import datetime  # noqa: F401  (E2E-7 needs main to have imported it)
import sys
import textwrap

import pytest

from interp import strict

pytestmark = pytest.mark.skipif(not strict.AVAILABLE, reason="no sub-interpreter module")


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


@pytest.mark.skipif(sys.version_info < (3, 13), reason="3.12 falls back to _pydatetime here (E2E-7)")
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


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="3.12 only")
def test_312_pure_python_datetime_goes_to_default():
    """E2E-7: with main having imported `datetime`, a strict 3.12
    sub-interpreter can't load `_datetime` and falls back to `_pydatetime`,
    whose objects have no C layout, so they go to `default`."""
    assert "_datetime" in sys.modules
    run_fresh(
        """
        import sys, datetime, isojson
        assert "_pydatetime" in sys.modules and "_datetime" not in sys.modules
        x = datetime.datetime(2026, 1, 1)
        seen = []
        assert isojson.dumps(x, default=lambda o: seen.append(o) or "D") == b'"D"'
        assert seen == [x]
        try:
            isojson.dumps(x)
        except TypeError as e:
            assert str(e).startswith("Type is not JSON serializable"), e
        else:
            raise AssertionError("no error")
        """
    )
