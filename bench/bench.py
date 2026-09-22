"""isojson benchmarks.

    python bench/bench.py            # everything, markdown tables on stdout
    python bench/bench.py single     # single-interpreter dumps/loads only
    python bench/bench.py parallel   # multi-interpreter scaling only

Method: every cell is repeated and reported as the median, because one shot
is not a result. Parallel cells time only the work loop — interpreter /
process creation and imports happen before the clock starts.
"""

import json
import random
import statistics
import string
import sys
import textwrap
import threading
import time
import timeit

import orjson

import isojson

REPEAT = 7

# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------


def payloads():
    r = random.Random(42)

    def word(n=8):
        return "".join(r.choice(string.ascii_letters) for _ in range(n))

    small = {"message": "Hello, World!"}
    records = [
        {
            "id": i,
            "name": word(12),
            "email": f"{word(6)}@{word(5)}.com",
            "active": r.random() < 0.5,
            "score": r.random() * 100,
            "tags": [word(5) for _ in range(3)],
            "address": {"city": word(9), "zip": str(r.randrange(10000, 99999)), "geo": [r.uniform(-90, 90), r.uniform(-180, 180)]},
            "note": None,
        }
        for i in range(100)
    ]
    floats = [r.uniform(-1e6, 1e6) for _ in range(10_000)]
    unicode = [{"zh": "多解释器安全的 JSON 序列化" * 3, "ja": "サブインタプリタ" * 3, "emoji": "🔥🐍" * 5, "esc": 'a"b\\c\nd\te'} for _ in range(200)]
    big = {"rows": [dict(rec, id=i) for i in range(20) for rec in records]}  # 2000 records
    return {
        "small (27 B)": small,
        "records ×100": records,
        "records ×2000": big,
        "floats ×10k": floats,
        "unicode/escapes ×200": unicode,
    }


def bench(fn, number):
    t = timeit.repeat(fn, number=number, repeat=REPEAT)
    per = [x / number for x in t]
    return statistics.median(per), min(per), max(per)


def fmt_time(s):
    if s < 1e-6:
        return f"{s * 1e9:.0f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} µs"
    return f"{s * 1e3:.2f} ms"


def calibrate(fn):
    n = 1
    while True:
        t = timeit.timeit(fn, number=n)
        if t > 0.2:
            return max(1, int(n * 0.2 / t))
        n *= 4


def single():
    libs = {
        "isojson": (isojson.dumps, isojson.loads),
        "orjson": (orjson.dumps, orjson.loads),
        "json (stdlib)": (lambda o: json.dumps(o).encode(), json.loads),
    }
    for op in ("dumps", "loads"):
        print(f"\n### {op} (single interpreter, median of {REPEAT}; lower is better)\n")
        print("| payload | isojson | orjson | json (stdlib) | isojson vs orjson |")
        print("|---|---:|---:|---:|---:|")
        for name, obj in payloads().items():
            raw = orjson.dumps(obj)
            cells = {}
            for lib, (d, l) in libs.items():
                fn = (lambda d=d: d(obj)) if op == "dumps" else (lambda l=l: l(raw))
                cells[lib] = bench(fn, calibrate(fn))
            ratio = cells["isojson"][0] / cells["orjson"][0]
            row = " | ".join(fmt_time(cells[k][0]) for k in libs)
            print(f"| {name} | {row} | {ratio:.2f}× |")


# --------------------------------------------------------------------------
# parallel
# --------------------------------------------------------------------------

WORK = textwrap.dedent(
    """
    import json, random, string, time
    LIB
    r = random.Random(7)
    def word(n=8): return "".join(r.choice(string.ascii_letters) for _ in range(n))
    doc = [{"id": i, "name": word(12), "score": r.random() * 100, "tags": [word(5) for _ in range(3)],
            "geo": [r.uniform(-90, 90), r.uniform(-180, 180)], "active": True, "note": None} for i in range(100)]
    def step():
        return loads(dumps(doc))
    """
)

LIBS_SRC = {
    "isojson": "import isojson; dumps, loads = isojson.dumps, isojson.loads",
    "orjson": "import orjson; dumps, loads = orjson.dumps, orjson.loads",
    "json (stdlib)": "dumps = lambda o: json.dumps(o).encode(); loads = json.loads",
}

ITERS = 3000  # round trips per worker


def run_subinterps(lib, n):
    """n own-GIL sub-interpreters, one thread each, each does ITERS round trips.
    Returns wall seconds for the work loop only, or an error string."""
    from concurrent import interpreters

    interps = []
    try:
        for _ in range(n):
            it = interpreters.create()
            it.exec(WORK.replace("LIB", LIBS_SRC[lib]))
            interps.append(it)
    except Exception as e:  # noqa: BLE001
        for it in interps:
            it.close()
        return str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__

    barrier = threading.Barrier(n + 1)
    errors = []

    def worker(it):
        barrier.wait()
        try:
            it.exec(f"for _ in range({ITERS}): step()")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(it,)) for it in interps]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    for it in interps:
        it.close()
    if errors:
        return repr(errors[0])
    return wall


def run_threads_main(lib, n):
    """Control: n plain threads in the main interpreter (shared GIL)."""
    ns = {}
    exec(WORK.replace("LIB", LIBS_SRC[lib]), ns)
    step = ns["step"]
    barrier = threading.Barrier(n + 1)

    def worker():
        barrier.wait()
        for _ in range(ITERS):
            step()

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def _proc_worker(lib, barrier, done):
    ns = {}
    exec(WORK.replace("LIB", LIBS_SRC[lib]), ns)
    step = ns["step"]
    barrier.wait()
    for _ in range(ITERS):
        step()
    done.put(time.perf_counter())


def run_processes(lib, n):
    """Control: n separate processes (the usual way to get parallelism).
    Timed from the common start barrier to the last worker finishing its loop
    — process startup and imports are excluded, like the other rows."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n + 1)
    done = ctx.Queue()
    procs = [ctx.Process(target=_proc_worker, args=(lib, barrier, done)) for _ in range(n)]
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.perf_counter()
    ends = [done.get() for _ in range(n)]
    for p in procs:
        p.join()
    return max(ends) - t0


def median_run(f, *a):
    rs = [f(*a) for _ in range(5)]
    if any(isinstance(x, str) for x in rs):
        return next(x for x in rs if isinstance(x, str))
    return statistics.median(rs)


def parallel():
    import os

    cores = os.cpu_count()
    print(
        f"\n### Parallel round trips (dumps+loads of a 100-record document), {ITERS} per worker, "
        f"{cores} logical cores; throughput in round trips/s, higher is better\n"
    )
    ns = [1, 2, 4, 8]
    print("| setup | " + " | ".join(f"N={n}" for n in ns) + " | scaling N=1→8 |")
    print("|---|" + "---:|" * len(ns) + "---:|")
    rows = [
        ("isojson — N own-GIL sub-interpreters", run_subinterps, "isojson"),
        ("json (stdlib) — N own-GIL sub-interpreters", run_subinterps, "json (stdlib)"),
        ("orjson — N own-GIL sub-interpreters", run_subinterps, "orjson"),
        ("orjson — N threads, one interpreter (shared GIL)", run_threads_main, "orjson"),
        ("isojson — N threads, one interpreter (shared GIL)", run_threads_main, "isojson"),
        ("orjson — N processes (multiprocessing)", run_processes, "orjson"),
        ("isojson — N processes (multiprocessing)", run_processes, "isojson"),
    ]
    notes = []
    for label, f, lib in rows:
        cells, tput = [], []
        for n in ns:
            w = median_run(f, lib, n)
            if isinstance(w, str):
                cells.append("✗")
                tput.append(None)
                err = w
            else:
                tp = n * ITERS / w
                tput.append(tp)
                cells.append(f"{tp:,.0f}")
        scale = f"{tput[-1] / tput[0]:.2f}×" if tput[0] and tput[-1] else "—"
        print(f"| {label} | " + " | ".join(cells) + f" | {scale} |")
        if any(t is None for t in tput):
            notes.append(f"✗ {label.split(' — ')[0]}: `{err}`")
    for note in notes:
        print(f"\n{note}")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} · isojson {isojson.__version__} · orjson {orjson.__version__}")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "single"):
        single()
    if what in ("all", "parallel"):
        parallel()
