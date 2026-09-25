"""E2E-6: datetimes under concurrent own-GIL sub-interpreters (3.13, 3.14).

4 and 8 interpreters driven by 8 threads, then 200 create/destroy cycles,
each with its own seeded aware/naive datetime payload checked byte for byte
against `isoformat()`, all in a child under `PYTHONMALLOC=debug`. Datetime
types are process-static on 3.13+, so type separation is E2E-3's job; this
proves the shared types are used safely.

Tripwire: isojson reference-counts `_datetime`'s shared types from several
interpreters, which is race-free only because they are immortal (C1).

3.14 only. CPython 3.13.15's own `_datetime` can't run this, with isojson
neither imported nor called (measured in M1, macOS arm64): strict
interpreters importing `datetime` concurrently die with SIGSEGV (5/5 runs);
under `PYTHONMALLOC=debug`, sequentially imported but concurrently used
`datetime` objects corrupt memory (10/10); and this test's workload with
creation, import and destruction serialized still aborts (SIGABRT, 10/10).
3.14.7 passes all three. isojson can't make 3.13's datetime safe here.
"""

import datetime
import os
import subprocess
import sys
import textwrap

import pytest


WORKER = textwrap.dedent(
    """
    import datetime as dt, random, isojson

    def build(seed):
        r = random.Random(seed)
        tzs = [None, dt.timezone.utc, dt.timezone(dt.timedelta(hours=r.randrange(-23, 24),
                                                               minutes=r.randrange(60)))]
        return [dt.datetime(r.randint(1, 9999), r.randint(1, 12), r.randint(1, 28),
                            r.randrange(24), r.randrange(60), r.randrange(60),
                            r.randrange(1_000_000), tzinfo=r.choice(tzs))
                for _ in range(r.randrange(20, 80))]

    for n in range(ITERATIONS):
        seed = SEED * 1_000_003 + n
        xs = build(seed)
        opts = isojson.OPT_NAIVE_UTC if n % 2 else 0
        want = "[" + ",".join(
            '"' + (x.replace(tzinfo=dt.timezone.utc) if opts and x.tzinfo is None else x).isoformat() + '"'
            for x in xs) + "]"
        got = isojson.dumps(xs, option=opts).decode()
        assert got == want, (seed, got[:200], want[:200])
    """
)

DRIVER = textwrap.dedent(
    """
    import sys, threading
    sys.path.insert(0, TESTS)
    from interp import strict

    WORKER = WORKER_SRC
    errors = []
    make, destroy = strict.make, strict.destroy

    def code(seed, iterations):
        return WORKER.replace("ITERATIONS", str(iterations)).replace("SEED", str(seed))

    # n long-lived interpreters, 8 threads. An interpreter runs one exec at a
    # time, so threads sharing one take turns through its lock, in short runs
    # that interleave.
    for n in (4, 8):
        ids = [make() for _ in range(n)]
        locks = [threading.Lock() for _ in range(n)]
        def drive(t):
            try:
                for k in range(10):
                    with locks[t % n]:
                        strict.run(ids[t % n], code(n * 1000 + t * 10 + k, 4))
            except Exception as e:
                errors.append(("drive", n, t, str(e)[:500]))
        threads = [threading.Thread(target=drive, args=(t,)) for t in range(8)]
        for th in threads: th.start()
        for th in threads: th.join()
        for i in ids:
            destroy(i)

    # 200 create/destroy cycles, spread over 8 threads
    def cycles(t):
        try:
            for k in range(25):
                i = make()
                try:
                    strict.run(i, code(10_000 + t * 25 + k, 3))
                finally:
                    destroy(i)
        except Exception as e:
            errors.append(("cycle", t, str(e)[:500]))
    threads = [threading.Thread(target=cycles, args=(t,)) for t in range(8)]
    for th in threads: th.start()
    for th in threads: th.join()

    assert not errors, errors
    print("ok")
    """
)


@pytest.mark.skipif(sys.version_info < (3, 13), reason="3.12 is E2E-7")
def test_immortal_shared_types_tripwire():
    for t in (datetime.datetime, datetime.date, datetime.time):
        if hasattr(sys, "_is_immortal"):
            assert sys._is_immortal(t), t
        else:
            assert sys.getrefcount(t) == 2**32 - 1, (t, sys.getrefcount(t))


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="3.14+: CPython 3.13's own _datetime crashes under concurrent strict "
    "interpreters with no isojson involved (see module doc); 3.12 is E2E-7",
)
def test_concurrent_interpreters_and_cycles():
    tests = os.path.dirname(os.path.abspath(__file__))
    src = DRIVER.replace("TESTS", repr(tests)).replace("WORKER_SRC", repr(WORKER))
    env = dict(os.environ, PYTHONMALLOC="debug")
    r = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, errors="replace", timeout=600, env=env,
    )
    assert r.returncode == 0, (r.returncode, r.stderr[-3000:])
    assert r.stdout.strip() == "ok", r.stdout[-2000:]
