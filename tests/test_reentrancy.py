"""E2E-10 / 10b: reentrancy (design FR-13).

Each case runs in a fresh child process under `PYTHONMALLOC=debug`, where a
freed object's memory is poisoned, so a use-after-free crashes instead of
passing by luck. E2E-10 makes this the first `dumps` after `import datetime`,
so the guard must come from the datetime group resolved at call start.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

PRELUDE = """
import datetime as dt, gc, json, isojson

class Mutating(dt.tzinfo):
    '''utcoffset() empties the container that holds the value being written.'''
    def __init__(self, action):
        self.action = action
    def utcoffset(self, _):
        self.action()
        gc.collect()
        return dt.timedelta(hours=1)
"""


def run_child(body):
    env = dict(os.environ, PYTHONMALLOC="debug")
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PRELUDE) + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert r.returncode == 0, (r.returncode, r.stderr[-3000:])
    return r.stdout


CASES = {
    # a list whose only item is a tuple holding the datetime; the list is cleared
    "list-clear": """
        outer = []
        x = dt.datetime(2026, 1, 1, tzinfo=Mutating(outer.clear))
        outer.append((x,))
        del x
        out = isojson.dumps(outer, **KW)
    """,
    # a dict whose value is a list holding the datetime; the key is popped
    "dict-pop": """
        d = {}
        x = dt.datetime(2026, 1, 1, tzinfo=Mutating(lambda: d.pop("a")))
        d["a"] = [x]
        del x
        out = isojson.dumps(d, **KW)
    """,
    # the same with an aware time
    "time-list-clear": """
        outer = []
        x = dt.time(1, tzinfo=Mutating(outer.clear))
        outer.append([x])
        del x
        out = isojson.dumps(outer, **KW)
    """,
}

EXPECTED = {
    "list-clear": [["2026-01-01T00:00:00+01:00"]],
    "dict-pop": {"a": ["2026-01-01T00:00:00+01:00"]},
    "time-list-clear": [["01:00:00+01:00"]],
}

VARIANTS = {
    "no-default": "KW = {}",
    "default": "KW = {'default': repr}",
    "numpy": "KW = {'option': isojson.OPT_SERIALIZE_NUMPY}",
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES)
def test_container_mutated_by_utcoffset(case, variant):
    out = run_child(
        VARIANTS[variant]
        + "\n"
        + textwrap.dedent(CASES[case])
        + "\nprint(out.decode())\n"
    )
    assert json.loads(out) == EXPECTED[case]


def test_first_dumps_after_import_is_guarded():
    """The guard comes from the datetime group resolved at call start: this
    is the first `dumps` of the process, with neither `default` nor numpy."""
    out = run_child(
        "KW = {}\n"
        + textwrap.dedent(CASES["dict-pop"])
        + "\nprint(out.decode())\n"
    )
    assert json.loads(out) == EXPECTED["dict-pop"]


def test_e2e10b_default_imports_datetime():
    """E2E-10b: `datetime` is not imported when `dumps` starts; `default`
    imports it and returns a datetime, which is written natively."""
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import sys, isojson
            assert "_datetime" not in sys.modules and "datetime" not in sys.modules
            def default(o):
                import datetime
                return datetime.datetime(2026, 1, 1, 12, 30)
            print(isojson.dumps([object(), 1], default=default).decode())
            """
        )],
        capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONMALLOC="debug"),
    )
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip() == '["2026-01-01T12:30:00",1]'
