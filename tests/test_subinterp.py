"""Multi-interpreter safety.

The criterion is not "it didn't crash": every worker gets its own seeded data
with a different expected answer and checks it exactly, so an object leaking
across interpreters shows up as a wrong result, not just (maybe) a crash.
"""

import subprocess
import sys
import textwrap
import threading

import pytest

interpreters = pytest.importorskip("concurrent.interpreters")


def test_strict_import_no_override():
    """Loads in own-GIL sub-interpreters without any compatibility override."""
    interps = [interpreters.create() for _ in range(6)]
    try:
        for it in interps:
            it.exec("import isojson; assert isojson.loads(isojson.dumps([1])) == [1]")
    finally:
        for it in interps:
            it.close()


def test_per_interpreter_module_state():
    """Each interpreter has its own module object and its own JSONDecodeError
    type, subclassing *that interpreter's* json.JSONDecodeError."""
    it = interpreters.create()
    try:
        it.exec(
            textwrap.dedent(
                """
                import json, isojson
                assert issubclass(isojson.JSONDecodeError, json.JSONDecodeError)
                try:
                    isojson.loads("[1,")
                except json.JSONDecodeError as e:
                    assert type(e) is isojson.JSONDecodeError
                else:
                    raise AssertionError("no error")
                """
            )
        )
    finally:
        it.close()


WORKER = textwrap.dedent(
    """
    import random, isojson

    def build(seed):
        r = random.Random(seed)
        return {
            "seed": seed,
            "rows": [{"id": i, "v": r.random(), "s": str(r.getrandbits(64)), "t": (i, None, True)}
                     for i in range(r.randrange(50, 150))],
        }

    class Box:
        def __init__(self, v):
            self.v = v

    def default(o):          # a Python callback from inside the serializer
        if isinstance(o, Box):
            return {"box": o.v}
        raise TypeError

    for n in range(ITERATIONS):
        seed = SEED * 1_000_003 + n
        doc = build(seed)
        doc["box"] = Box(seed)
        raw = isojson.dumps(doc, default=default, option=isojson.OPT_SORT_KEYS)
        back = isojson.loads(raw)
        assert back["seed"] == seed, (back["seed"], seed)
        assert back["box"] == {"box": seed}
        expect = build(seed)
        assert len(back["rows"]) == len(expect["rows"])
        for a, b in zip(back["rows"], expect["rows"]):
            assert a["id"] == b["id"] and a["v"] == b["v"] and a["s"] == b["s"]
            assert a["t"] == list(b["t"])
    """
)


@pytest.mark.parametrize("n_interps", [4, 8])
def test_concurrent_interpreters_exact_results(n_interps):
    interps = [interpreters.create() for _ in range(n_interps)]
    errors = []

    def run(i, it):
        try:
            it.exec(WORKER.replace("ITERATIONS", "300").replace("SEED", str(i + 1)))
        except Exception as e:  # noqa: BLE001
            errors.append((i, e))

    threads = [threading.Thread(target=run, args=(i, it)) for i, it in enumerate(interps)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for it in interps:
        it.close()
    assert not errors, errors


def test_create_destroy_cycles():
    """Import, use, and destroy interpreters repeatedly — module state must be
    torn down with each interpreter (run in a child so a crash is reported,
    not swallowed)."""
    code = textwrap.dedent(
        """
        from concurrent import interpreters
        for _ in range(200):
            it = interpreters.create()
            it.exec("import isojson; isojson.loads(isojson.dumps({'a': [1, 2.5, 'x']}))")
            it.close()
        print("ok")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    assert r.stdout.strip() == "ok"
