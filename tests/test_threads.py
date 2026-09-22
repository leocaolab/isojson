"""Threads: many threads calling isojson at once, in one interpreter and
mixed with own-GIL sub-interpreters.

What is shared across threads is only plain data — per-thread scratch
buffers, simd-json's CPU-feature detection, the str fast-path flag — never a
Python object. Each worker checks its own seeded data against its own
expected answer, so any cross-talk shows up as a wrong result.
"""

import random
import threading

import pytest

import isojson


def build(seed):
    r = random.Random(seed)
    return {
        "seed": seed,
        "rows": [
            {"id": i, "v": r.random(), "s": "é" * (i % 5) + str(r.getrandbits(64)), "q": 'a"b\\c\n' * (i % 3)}
            for i in range(r.randrange(20, 120))
        ],
    }


class Box:
    def __init__(self, v):
        self.v = v


def default(o):
    if isinstance(o, Box):
        return {"box": o.v}
    raise TypeError


def work(tid, iterations, errors):
    try:
        for n in range(iterations):
            seed = tid * 1_000_003 + n
            doc = build(seed)
            doc["box"] = Box(seed)
            raw = isojson.dumps(doc, default=default, option=isojson.OPT_SORT_KEYS)
            back = isojson.loads(raw)
            expect = build(seed)
            expect["box"] = {"box": seed}
            assert back == expect, (tid, n)
    except Exception as e:  # noqa: BLE001
        errors.append((tid, repr(e)))


def test_threads_one_interpreter():
    errors = []
    threads = [threading.Thread(target=work, args=(t, 200, errors)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_threads_and_subinterpreters_together():
    interpreters = pytest.importorskip("concurrent.interpreters")
    import inspect
    import textwrap

    src = "\n".join(
        [
            "import random, isojson",
            textwrap.dedent(inspect.getsource(build)),
            textwrap.dedent(inspect.getsource(Box)),
            textwrap.dedent(inspect.getsource(default)),
            textwrap.dedent(inspect.getsource(work)),
        ]
    )
    interps = [interpreters.create() for _ in range(4)]
    errors = []

    def in_interp(i, it):
        try:
            it.exec(src + f"\nerrs = []\nwork({100 + i}, 200, errs)\nassert not errs, errs\n")
        except Exception as e:  # noqa: BLE001
            errors.append((f"interp {i}", repr(e)))

    threads = [threading.Thread(target=in_interp, args=(i, it)) for i, it in enumerate(interps)]
    threads += [threading.Thread(target=work, args=(t, 200, errors)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for it in interps:
        it.close()
    assert not errors, errors
