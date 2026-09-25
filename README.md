# isojson

**Fast JSON for CPython that works in per-interpreter-GIL sub-interpreters.**

`isojson` is a Rust extension with an orjson-compatible API (`dumps`, `loads`,
the same `OPT_*` flags, and byte-identical output for the types it supports).
The difference is that it loads and runs in **own-GIL sub-interpreters**
([PEP 684](https://peps.python.org/pep-0684/)) with no compatibility override.
orjson does not:

```text
>>> from concurrent import interpreters
>>> interpreters.create().exec("import orjson")
ImportError: module orjson.orjson does not support loading in subinterpreters

>>> interpreters.create().exec("import isojson")   # works, strict mode, no override
```

```bash
pip install isojson
```

Wheels are published for CPython 3.14 on Linux (x86_64,
aarch64), macOS (arm64, x86_64) and Windows (x86_64).

```python
import isojson

isojson.dumps({"a": [1, 2.5, None]})          # b'{"a":[1,2.5,null]}'
isojson.loads(b'{"a":[1,2.5,null]}')           # {'a': [1, 2.5, None]}
isojson.dumps(obj, default=str, option=isojson.OPT_SORT_KEYS | isojson.OPT_INDENT_2)
```

Switching from orjson is usually a one-line change: `import isojson as orjson`.
Check the [differences](#differences-from-orjson) first.

---

## Pure Rust, and no worrying about threads or sub-interpreters

**isojson is pure Rust.** There is no C or C++ source anywhere in the
dependency tree, and the build never runs a C compiler. The only native
library it links is libpython itself. The runtime dependencies are
`pyo3-ffi` (declarations of the CPython C API, no code), `simd-json` (with
`value-trait`, `halfbrown` and `simdutf8`), `zmij` and `itoa`. The SIMD code
(SSE2 on x86_64, NEON on aarch64) uses Rust's `core::arch` intrinsics.

**Things you do *not* have to do with isojson:**

- set `_override_multi_interp_extensions_check` or any other escape hatch;
- ship or load a physical copy of the extension per worker;
- import it in the main interpreter first, or in any particular order;
- keep calls on one thread, or add locks around them.

Import it in as many own-GIL sub-interpreters as you like, in strict mode,
and call it from any thread.

**Why that is safe:** the only state shared across interpreters or threads is
plain data, never a Python object:

| shared state | what it is |
|---|---|
| per-thread scratch buffers and size hints | bytes and sizes only (`thread_local!`) |
| simd-json's CPU-feature detection | an atomic set once per process |
| the `str` fast-path switch | an atomic, set by an import-time self-check that gives the same result in every interpreter |

Every Python object isojson keeps longer than one call lives in
per-interpreter module state. That is the `JSONDecodeError` type, the
dict-key cache, and the type cache (the `datetime` types, looked up in that
interpreter's own `sys.modules`, never imported). CPython creates it for each
interpreter and frees it with that interpreter.

**How this is tested:** strict import in 6 own-GIL sub-interpreters, with no
override; 4 and 8 sub-interpreters running concurrently; 8 threads in one
interpreter; threads and sub-interpreters running together; 200
create/use/destroy cycles. The whole suite also passes under
`PYTHONMALLOC=debug`, on macOS (arm64) and Linux (x86_64). Every concurrent
worker checks its own seeded data against its own expected answer, so a
leak between threads or interpreters shows up as a wrong result, not just
as a crash that may or may not happen.

**Free-threaded CPython (3.14t):** isojson works there, but it is
not free-threading-ready yet. The module declares that it needs the GIL, so
CPython re-enables the GIL when it is imported and prints a `RuntimeWarning`.
We checked this on 3.14t. The `str` fast path is also off on those builds.

---

## Why this exists

Python 3.12 added a separate GIL per sub-interpreter, and Python 3.14 made it
usable from Python (`concurrent.interpreters`). N sub-interpreters in one
process can run Python code on N cores, which lets a single process scale
across cores without multiprocessing.

For that to work, every C extension a sub-interpreter imports has to be safe
there. In practice this means:

1. **multi-phase initialization** (PEP 489), so each interpreter gets its own
   module object;
2. **no Python objects stored in process-global state**, since an object
   belongs to exactly one interpreter and touching it from another corrupts
   reference counts and allocator arenas;
3. **declaring `Py_MOD_PER_INTERPRETER_GIL_SUPPORTED`**, which is only honest
   once 1 and 2 hold.

orjson is the fastest JSON library for CPython, but it does not meet these
requirements and refuses to import in a sub-interpreter. That left people
running sub-interpreters two options:

- **stdlib `json`**: safe, but 2.7–13× slower than orjson depending on the
  payload (see the tables below);
- **a physical copy of orjson's shared library per worker**, loaded under an
  override. This works, but you pay one copy's memory, disk, and load time per
  worker, plus a separate mechanism to maintain.

isojson is a JSON library written from the start to meet all three
requirements.

## How it works

- **Multi-phase init with per-interpreter state.** The module declares
  `Py_MOD_PER_INTERPRETER_GIL_SUPPORTED`, which is honest only because
  nothing process-global holds a Python object (see above). The only
  process-global pointers isojson touches are CPython's static builtin types,
  `_datetime`'s static types and C-API struct, and the immortal singletons
  `None`/`True`/`False`, which every interpreter shares by design.
- **The key cache is per interpreter.** Like orjson, `loads` caches recently
  seen dict keys, so repeated keys reuse one `str` with its hash already
  computed. orjson keeps that cache process-wide. isojson keeps one per
  interpreter, because a shared cache would hand one interpreter's objects to
  another.
- **`dumps` borrows; only types are kept.** It only borrows the objects it
  writes, for the length of the call, on the calling thread, under the
  caller's GIL; across calls it keeps only the per-interpreter type cache.
  When Python code could run during the walk (a `default=` callback, or a
  `datetime`'s `utcoffset()`) and mutate a container, items are also held by
  a reference for that span. Output is written
  straight into the result `bytes` object, so there is no final copy. String
  escaping scans 16 bytes at a time: SSE2 on x86_64 and NEON on aarch64, both
  part of those architectures' baseline, so no runtime detection is needed.
- **`str` contents are read from the object header, after a self-check.**
  Where CPython 3.14 already holds a string's UTF-8 (compact ASCII, or a
  cached UTF-8 copy), isojson reads it directly instead of calling
  `PyUnicode_AsUTF8AndSize`. That layout is not public API. So at import,
  isojson compares its reading with the API on probe strings and turns the
  fast path on only if every probe agrees. If a future CPython changes the
  layout, isojson gets slower but never wrong.
- **`loads` parses with simd-json and uses no recursion.** simd-json turns
  the document into a flat tape where every array and object carries its
  length. isojson builds Python objects from the tape with an explicit
  stack, so lists are created at their exact size. A document nested 1024
  deep is safe even on threads with small C stacks; deeper documents are
  rejected, as in orjson. simd-json never touches a Python object.
- **Exceptions belong to the interpreter.** `isojson.JSONDecodeError`
  subclasses the calling interpreter's own `json.JSONDecodeError`. Its
  messages are plain sentences, never the parser's internal error names.
- **No PyO3.** The module is written against the raw C API (`pyo3-ffi`).
  PyO3's high-level layer caches type objects and modules in process-global
  statics and rejects a second interpreter, which is exactly what this
  package exists to avoid.

## Why simd-json, not sonic-rs

Both are fast, pure-Rust JSON parsers. We compared them directly on
JSONTestSuite (318 cases, each run in its own child process so a crash is
recorded), on edge cases that matter for a Python library, and on raw
parse speed:

| | simd-json 0.18.1 | sonic-rs 0.5.10 |
|---|---|---|
| JSONTestSuite: must-accept rejected / must-reject accepted | 0 / 0 | 0 / 0 |
| JSONTestSuite: crashes | **0** | **2** (stack overflow: `n_structure_100000_opening_arrays`, `n_structure_open_array_object`) |
| Deeply nested input | depth limit 1024, then a clean error (same as orjson) | **no limit**: 100,000 levels abort the process, which cannot be caught |
| `-0` | integer `0` (same as orjson) | float `0.0` |
| Integers beyond 64 bits, `1e400` | float / error (same as orjson) | same |
| Lone surrogate `"\ud800"` | decoded as `"\x00"`: **bug, fixed by us** (see below) | error |
| API for building Python objects | a tape where every container carries its length | serde visitor (recursive) or its own DOM |

Parse only, no Python objects, µs, Apple M5 Pro:

| payload | simd-json (tape) | sonic-rs (`Value`) | serde_json (`Value`) |
|---|---:|---:|---:|
| floats ×10k | **142** | 152 | 255 |
| records ×100 | 18.6 | 18.9 | 100 |
| records ×2000 | 360 | **336** | 1906 |
| unicode/escapes ×200 | 28.0 | **19.8** | 96.5 |

Speed is a draw. The deciding factor is robustness. A JSON library inside a
server must not let one request take the process down, and sonic-rs does
exactly that on deeply nested input. simd-json's tape is also the shape
isojson needs.

simd-json had one real bug for us: a lone high surrogate (`"\ud800"`) was
decoded as U+0000 instead of being rejected, so invalid input silently
became different data. The cause was an old "0 means failure" sentinel that
kept its value but lost its meaning in a 2023 refactor. We reported it
([simd-lite/simd-json#481](https://github.com/simd-lite/simd-json/issues/481))
and sent the fix with a regression test
([#482](https://github.com/simd-lite/simd-json/pull/482)). Until a release
contains it, isojson pins our fork at that commit (see `Cargo.toml`), and
isojson's own tests cover the case.

## Feature comparison

| | isojson | orjson | json (stdlib) |
|---|:---:|:---:|:---:|
| **Loads in own-GIL sub-interpreters (strict, no override)** | ✅ | ❌ `ImportError` | ✅ |
| `dict`, `list`, `tuple`, `str`, `int`, `float`, `bool`, `None` | ✅ | ✅ | ✅ |
| Subclasses of `str` / `int` / `dict` / `list` | ✅ | ✅ | ✅ |
| `default=` callable | ✅ | ✅ | ✅ |
| `OPT_INDENT_2`, `OPT_SORT_KEYS`, `OPT_APPEND_NEWLINE` | ✅ | ✅ | ~ (`indent=`, `sort_keys=`) |
| `OPT_STRICT_INTEGER`, `OPT_PASSTHROUGH_SUBCLASS` | ✅ | ✅ | — |
| `loads` from `bytes` / `bytearray` / `memoryview` / `str` | ✅ | ✅ | `str`/`bytes` only |
| Output bytes identical to orjson for the types above | ✅ | — | ❌ |
| `datetime` / `date` / `time` natively, with `OPT_NAIVE_UTC`, `OPT_UTC_Z`, `OPT_OMIT_MICROSECONDS`, `OPT_PASSTHROUGH_DATETIME` | ✅ | ✅ | ❌ |
| numpy arrays and scalars (`OPT_SERIALIZE_NUMPY`), incl. `datetime64` | ✅ (numpy ≥ 2) | ✅ | ❌ |
| `uuid.UUID`, `enum.Enum`, dataclasses natively | ❌ → `default=` | ✅ | ❌ |
| Non-`str` dict keys (`OPT_NON_STR_KEYS`) | ❌ raises | ✅ | ~ (coerced) |
| `orjson.Fragment` | ❌ | ✅ | ❌ |
| Integers beyond 64 bits in `dumps` | ❌ (like orjson) | ❌ | ✅ |
| NaN / ±Infinity in `dumps` | `null` (like orjson) | `null` | `NaN` / `Infinity` |
| Python versions | CPython 3.14 | CPython 3.10+ | all |

## Differences from orjson

These are all the known differences. Anything not listed produces the same
result as orjson 3.12.0 (the test suite checks this, see [Testing](#testing)).

### Where orjson crashes or writes wrong data

isojson writes what Python's own API says the value is: `isoformat()` for
`datetime` / `date` / `time` (after the options are applied), and for
`datetime64` the meaning numpy's API defines (`v × mult` units since 1970,
sub-µs floored to µs). What has no answer in that format is *declined* (next
section). Each row has a regression test that proves both halves.

| # | Input | orjson 3.12.0 | isojson |
|---|---|---|---|
| DV-1 | UTC offset with seconds or microseconds (`+05:59:30`, `-00:00:01`, New York before 1883 = `-04:56:02`) | rounds to the minute without carrying: `+05:60`, `-00:00` (sign lost), `-04:56` | `isoformat()`'s offset, exactly |
| DV-3 | pytz datetime after arithmetic, not normalized | the normalized offset on the un-normalized wall time (a different instant) | `dt.utcoffset()` |
| DV-4a | a tzinfo whose `utcoffset()` returns `None` | an invented offset (`+00:00` on macOS / x86_64 Linux, garbage such as `+18:12` on aarch64 Linux) | no offset: naive to Python |
| DV-4b | `utcoffset()` raises (datetime) / is invalid (time) | datetime: crashes (SIGSEGV) | `TypeError("datetime.utcoffset() raised …")`, the exception as `__cause__` |
| DV-5 | `datetime64` NaT in `ns` | `"1677-09-21T00:12:43.145224"` | `null` |
| DV-6 | NaT in `W D h m` | `"1970-01-01T00:00:00"` | `null` |
| DV-7 | NaT in `Y M s ms us`, and generic NaT | `TypeError` | `null` |
| DV-8 | multiplied units (`M8[10ms]`, `M8[2D]`) | crashes (`unreachable!()`) | value × multiplier |
| DV-9 | `M8[M]` before 1970 | crashes / `TypeError` | floor division (`1969-12-01T00:00:00`) |
| DV-10 | `M8[D…us]` from `9999-12-30T22:00` to `9999-12-31` | `TypeError` | written |
| DV-11 | values whose seconds overflow i64 (`M8[m]` 307445734561825861) | a wrapped, wrong value | declined |
| DV-12 | `datetime.time` with `tzinfo` | `TypeError` | `t.isoformat()`: `"01:00:00+00:00"` |
| DV-13 | generic-unit `M8` holding a value | `TypeError: … unit: NaT` (misnames it) | declined |
| DV-14 | ≥2-D `M8` arrays with an element orjson's 1-D writer rejects | **malformed JSON, no error, even with `default`** (`[[,[…]]`) | per the rows above |
| DV-15 | `utcoffset()` returns a non-timedelta or ≥ 24 h | an invented or garbage offset | `TypeError`, as `dt.utcoffset()` raises |
| DV-16 | `datetime64` in `ps`, `fs`, `as` | `TypeError`, even with `default` | floored to µs |
| DV-17 | a `time` whose microsecond has five digits (`time(0, 0, 1, 75652)`) | drops the leading zero: `"00:00:01.75652"` | `"00:00:01.075652"` |

### Declined numpy objects

A numpy object isojson can't write — a non-C-contiguous or non-native-endian
array, a 0-d array, an unsupported dtype, a generic-unit value, an
unrepresentable `datetime64` — goes **whole** to `default=` when one is given.
A decline found mid-array rolls the output back first, so `default` gets the
array, not a half-written one. Without `default`, `dumps` raises orjson's
message for the reason, with a note (`add_note`) carrying the raw evidence
(`dtype.str`, flags, shape, value). orjson sends some of these to `default`
too, but raises for non-native-endian arrays and out-of-range `datetime64`
even with a `default`. Unrecognized numpy scalars (`complex128`,
`longdouble`) take the ordinary `default` path. Unaligned C-contiguous arrays
(`np.frombuffer(…, offset=1)`) are read correctly (orjson's typed-slice read
is undefined behaviour there).

### Kept from 0.1

- **`uuid.UUID`, `enum.Enum`, dataclasses and `orjson.Fragment`** go to
  `default=` (or raise `Type is not JSON serializable`).
- **`OPT_NON_STR_KEYS` raises** `TypeError: isojson does not support
  OPT_NON_STR_KEYS`, rather than being silently ignored.
  `OPT_PASSTHROUGH_DATACLASS` is accepted and does nothing (isojson never
  serializes dataclasses); `OPT_SERIALIZE_DATACLASS` and `OPT_SERIALIZE_UUID`
  are `0`, as in orjson.
- **`default=None`** means "no default". orjson calls `None` and raises
  `Type is not JSON serializable` with a `'NoneType' object is not callable`
  cause.
- **Argument errors** (`dumps()` with no object, unknown keywords) have
  different wording.
- **`loads` error messages differ.** The exception type (`JSONDecodeError`,
  a subclass of `json.JSONDecodeError` and `ValueError`) and `.pos` / `.lineno`
  / `.colno` match. The wording of `.msg` does not always match orjson's.

## Limitations

- **CPython 3.14 only.** Per-interpreter GIL arrived in 3.12, but 3.12's
  and 3.13's own `_datetime` isn't usable from concurrent strict
  sub-interpreters, so isojson targets 3.14.
- **Free-threaded builds (3.14t):** the module declares that it needs the
  GIL, so CPython re-enables it on import (see above).
- **numpy:** `OPT_SERIALIZE_NUMPY` is tested with numpy ≥ 2. isojson reads
  arrays through `__array_struct__` and never imports numpy; it uses the
  numpy the calling interpreter already has in `sys.modules`.

## Performance

Reproduce with `python bench/bench.py`. Each cell is the median of repeated
runs. Parallel cells time only the work loop: interpreter or process
creation and imports happen before timing starts.

### macOS arm64: Apple M5 Pro, 18 cores, CPython 3.14.7, orjson 3.12.0

**Parallel, the case isojson is built for.** Each worker does 3,000 round
trips (`loads(dumps(doc))`) of a 100-record document. Throughput is in round
trips per second; higher is better.

| setup | N=1 | N=2 | N=4 | N=8 | scaling 1→8 |
|---|---:|---:|---:|---:|---:|
| **isojson, N own-GIL sub-interpreters, one process** | 15,162 | 31,731 | 60,705 | **114,405** | **7.55×** |
| json (stdlib), N own-GIL sub-interpreters | 4,019 | 8,277 | 14,230 | 26,679 | 6.64× |
| orjson, N own-GIL sub-interpreters | ✗ | ✗ | ✗ | ✗ | — |
| orjson, N threads, one interpreter (shared GIL) | 16,277 | 20,206 | 18,185 | 21,703 | 1.33× |
| isojson, N threads, one interpreter (shared GIL) | 18,160 | 17,932 | 18,270 | 19,076 | 1.05× |
| orjson, N processes (multiprocessing) | 22,773 | 42,917 | 80,230 | 149,056 | 6.55× |
| isojson, N processes (multiprocessing) | 19,286 | 36,307 | 64,255 | 123,112 | 6.38× |

✗ `ImportError: module orjson.orjson does not support loading in subinterpreters`

How to read this:

- **In one process, isojson on 8 sub-interpreters does about 5× the best
  orjson can do** (114k vs 22k round trips/s here; 4.9–5.3× across our runs).
  Adding threads to orjson gains almost nothing, because every thread shares
  one GIL.
- **isojson on sub-interpreters does 4.3× stdlib `json` on
  sub-interpreters**, and stdlib `json` was the only other option that works
  there.
- **Sub-interpreters get close to process-level scaling in one process**:
  isojson reaches 114k round trips/s on 8 sub-interpreters vs 123k on 8
  processes.
- **orjson on 8 processes is still faster** (149k) because orjson is faster
  per call. If you already run multiprocessing and never use
  sub-interpreters, orjson remains the faster choice.

**Single interpreter, per-call cost** (lower is better):

| `dumps` | isojson | orjson | json (stdlib) | isojson / orjson |
|---|---:|---:|---:|---:|
| small (27 B) | 47 ns | 46 ns | 594 ns | 1.03× |
| records ×100 | 18.33 µs | 17.04 µs | 153.80 µs | 1.08× |
| records ×2000 | 349.52 µs | 298.00 µs | 2.86 ms | 1.17× |
| floats ×10k | 111.83 µs | 209.36 µs | 2.39 ms | **0.53×** |
| unicode/escapes ×200 | 12.88 µs | 11.05 µs | 124.12 µs | 1.17× |

| `loads` | isojson | orjson | json (stdlib) | isojson / orjson |
|---|---:|---:|---:|---:|
| small (27 B) | 107 ns | 101 ns | 1.09 µs | 1.06× |
| records ×100 | 62.46 µs | 45.94 µs | 111.03 µs | 1.36× |
| records ×2000 | 1.39 ms | 851.92 µs | 2.16 ms | 1.63× |
| floats ×10k | 224.24 µs | 199.21 µs | 1.14 ms | 1.13× |
| unicode/escapes ×200 | 94.63 µs | 68.32 µs | 171.29 µs | 1.39× |

In summary: `dumps` is within 1.2× of orjson and about 2× faster on
float-heavy documents. `loads` is 1.1–1.6× slower than orjson. Both are
1.5–21× faster than stdlib `json`.

The remaining `loads` gap is mostly fixed per-call cost around simd-json:
the input is copied (simd-json unescapes in place, and `bytes` are
immutable), and a fresh tape is allocated on every call. Both are next on
the list.

### Linux x86_64: AMD Ryzen 7 7840HS, 8 cores / 16 threads, CPython 3.14.7, orjson 3.12.0

Run on a shared box with light background load (1-minute load average 6.0 at
the start, 3.9 at the end).

| setup | N=1 | N=2 | N=4 | N=8 | scaling 1→8 |
|---|---:|---:|---:|---:|---:|
| **isojson, N own-GIL sub-interpreters, one process** | 14,129 | 27,887 | 54,893 | **78,242** | **5.54×** |
| json (stdlib), N own-GIL sub-interpreters | 3,862 | 7,677 | 15,154 | 19,973 | 5.17× |
| orjson, N own-GIL sub-interpreters | ✗ | ✗ | ✗ | ✗ | — |
| orjson, N threads, one interpreter (shared GIL) | 19,934 | 18,492 | 17,668 | 17,537 | 0.88× |
| isojson, N threads, one interpreter (shared GIL) | 13,890 | 12,388 | 13,034 | 12,301 | 0.89× |
| orjson, N processes (multiprocessing) | 19,651 | 38,657 | 74,130 | 98,140 | 4.99× |
| isojson, N processes (multiprocessing) | 14,122 | 27,388 | 54,055 | 75,194 | 5.32× |

Here isojson on 8 sub-interpreters does **3.9×** the best orjson manages in
one process, and **3.9×** stdlib `json` on the same sub-interpreters. It
matches isojson on 8 processes (78k vs 75k). orjson on 8 processes is again
the fastest row (98k).

| `dumps` | isojson | orjson | json (stdlib) | isojson / orjson |
|---|---:|---:|---:|---:|
| small (27 B) | 78 ns | 79 ns | 933 ns | 0.99× |
| records ×100 | 27.25 µs | 19.83 µs | 171.77 µs | 1.37× |
| records ×2000 | 736.39 µs | 382.51 µs | 3.99 ms | 1.93× |
| floats ×10k | 180.59 µs | 183.01 µs | 3.23 ms | 0.99× |
| unicode/escapes ×200 | 16.93 µs | 13.93 µs | 107.23 µs | 1.22× |

| `loads` | isojson | orjson | json (stdlib) | isojson / orjson |
|---|---:|---:|---:|---:|
| small (27 B) | 146 ns | 128 ns | 948 ns | 1.13× |
| records ×100 | 78.60 µs | 54.66 µs | 151.23 µs | 1.44× |
| records ×2000 | 1.67 ms | 1.17 ms | 3.10 ms | 1.42× |
| floats ×10k | 300.53 µs | 188.56 µs | 1.58 ms | 1.59× |
| unicode/escapes ×200 | 87.24 µs | 54.87 µs | 164.43 µs | 1.59× |

Per call, x86_64 is harder on isojson than arm64. `dumps` of large record
documents is 1.9× slower than orjson here, against 1.2× on the Mac, and `loads`
is 1.1–1.6× slower. Both are still 1.9–18× faster than stdlib `json`.

### Across both machines

On 8 own-GIL sub-interpreters in one process, isojson does **4–5×** the best
orjson can do in a single process (3.9× on Linux x86_64, 5.3× on macOS arm64),
and about 4× stdlib `json`. orjson on multiprocessing stays faster per core on
both.

## Testing

```bash
pip install maturin
pip install -e ".[test]"      # pytest, orjson==3.12.0, numpy>=2, pytz, tzdata
maturin develop --release
pytest tests
cargo test --lib              # the pure Rust core
```

- **Parity with orjson** (`tests/test_parity.py`):
  - randomized documents (control characters, non-BMP text, 64-bit edges,
    random float bit patterns) compared byte-for-byte against orjson under
    every supported option combination;
  - 250,000 floats compared bit-exactly;
  - error types and messages for every `dumps` failure mode;
  - the recursion and `default` depth limits;
  - `loads` accept/reject behaviour on edge-case documents.
- **JSON conformance** (`tests/test_conformance.py`): all 318 cases of
  [JSONTestSuite](https://github.com/nst/JSONTestSuite). Every must-accept
  document parses, every must-reject document raises `JSONDecodeError`, and
  nothing crashes. The suite runs in a child process, so a crash is reported
  as a failure.
- **Multi-interpreter safety** (`tests/test_subinterp.py`):
  - strict import in 6 own-GIL sub-interpreters with no override;
  - per-interpreter module state and exception types;
  - 4 and 8 interpreters running concurrently, each on its own seeded data
    with its own expected answer, with a Python `default=` callback running
    inside the serializer on every call. A leak between interpreters would
    show up as a wrong answer, not just a possible crash;
  - 200 create/use/destroy cycles in a child process, so a crash is reported
    instead of swallowed.
- **Threads** (`tests/test_threads.py`): 8 threads in one interpreter, and
  4 threads plus 4 sub-interpreters at the same time, each worker on its own
  seeded data with its own expected answer.
- **datetime and numpy** (0.2):
  - byte parity with orjson for every datetime option combination, numpy
    dtypes × shapes × `default` × `OPT_INDENT_2`, and random documents
    (`tests/test_parity_types.py`);
  - one regression per [difference](#differences-from-orjson), proving
    orjson's failure, the Python-API reference, and isojson's match; the
    crash rows run in child processes (`tests/test_divergence.py`);
  - the Python API as the oracle: `isoformat()` / `fromisoformat()` round
    trips, and `datetime64` in every unit × multiplier across the i64 range
    against the exact integer meaning (`tests/test_stdlib_oracle.py`);
  - every decline rule, rollback, notes, and unaligned arrays
    (`tests/test_declines.py`); all 65,536 float16 values bit-exactly
    (`tests/test_f16.py`); a replaced `sys.modules["numpy"]`
    (`tests/test_numpy_swap.py`);
  - reentrancy: `utcoffset()` mutating the container being written, under
    `PYTHONMALLOC=debug` (`tests/test_reentrancy.py`); concurrent
    sub-interpreters writing datetimes (`tests/test_concurrency_dt.py`);
  - no process-global Python state: every `static` reviewed, banned symbols,
    cached types traversed by the GC (`tests/test_no_global_pyobject.py`).
- **Per-worker numpy in Pyronova**: Pyronova's
  `tests/test_isojson_numpy_workers.py` runs 4 workers, each with its own
  numpy copy, against orjson's bytes.

The suite passes on macOS arm64 and Linux x86_64, both normally and under
`PYTHONMALLOC=debug`. The `-X dev` / `PYTHONDEVMODE=1` run is on Linux.

## Status

Version 0.2.0: native `datetime` / `date` / `time` and numpy. The API is
stable (it is orjson's). `UUID`, `Enum` and dataclass support is next.

## License

Apache-2.0.

Third-party: simd-json (Apache-2.0 OR MIT); the float16 → float32
conversion is half-rs's `f16_to_f32_fallback` as shipped in orjson
(Apache-2.0 OR MIT, attribution in `src/float.rs`); `tests/data/JSONTestSuite` is from
[nst/JSONTestSuite](https://github.com/nst/JSONTestSuite) (MIT, license
included in that directory).
