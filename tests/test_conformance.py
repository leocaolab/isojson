"""JSONTestSuite (https://github.com/nst/JSONTestSuite) conformance.

y_*: must parse.  n_*: must raise JSONDecodeError.  i_*: implementation-
defined — either is fine, but it must not crash. The whole suite runs in a
child process so a crash is reported as a failure instead of killing pytest.
"""

import pathlib
import subprocess
import sys
import textwrap

DATA = pathlib.Path(__file__).parent / "data" / "JSONTestSuite"

RUNNER = textwrap.dedent(
    """
    import json, pathlib, sys, isojson
    out = {}
    for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
        try:
            isojson.loads(p.read_bytes())
            out[p.name] = "accept"
        except isojson.JSONDecodeError:
            out[p.name] = "reject"
        except Exception as e:  # anything but JSONDecodeError is a bug
            out[p.name] = f"wrong exception {type(e).__name__}: {e}"
    print(json.dumps(out))
    """
)


def test_json_test_suite():
    r = subprocess.run([sys.executable, "-c", RUNNER, str(DATA)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"runner crashed (exit {r.returncode}): {r.stderr[-2000:]}"
    import json

    results = json.loads(r.stdout)
    assert len(results) == 318
    bad = []
    for name, got in sorted(results.items()):
        if got.startswith("wrong exception"):
            bad.append(f"{name}: {got}")
        elif name.startswith("y_") and got != "accept":
            bad.append(f"{name}: must accept, got {got}")
        elif name.startswith("n_") and got != "reject":
            bad.append(f"{name}: must reject, got {got}")
    assert not bad, "\n".join(bad)
