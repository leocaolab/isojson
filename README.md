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

Wheels are published for CPython 3.12, 3.13 and 3.14 on Linux (x86_64,
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
| per-thread scratch buffers | bytes only (`thread_local!`) |
| simd-json's CPU-feature detection | an atomic set once per process |
| the `str` fast-path switch | an atomic, set by an import-time self-check that gives the same result in every interpreter |

Every Python object isojson keeps longer than one call lives in
per-interpreter module state. That is the `JSONDecodeError` type and the
dict-key cache. CPython creates it for each interpreter and frees it with
that interpreter.

**How this is tested:** strict import in 6 own-GIL sub-interpreters, with no
override; 4 and 8 sub-interpreters running concurrently; 8 threads in one
interpreter; threads and sub-interpreters running together; 200
create/use/destroy cycles. The whole suite also passes under
`PYTHONMALLOC=debug`, on macOS (arm64) and Linux (x86_64). Every concurrent
worker checks its own seeded data against its own expected answer, so a
leak between threads or interpreters shows up as a wrong result, not just
as a crash that may or may not happen.

**Free-threaded CPython (3.13t / 3.14t):** isojson works there, but it is
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
  process-global pointers isojson touches are CPython's static builtin types
  and the immortal singletons `None`/`True`/`False`, which every interpreter
  shares by design.
- **The key cache is per interpreter.** Like orjson, `loads` caches recently
  seen dict keys, so repeated keys reuse one `str` with its hash already
  computed. orjson keeps that cache process-wide. isojson keeps one per
  interpreter, because a shared cache would hand one interpreter's objects to
  another.
- **`dumps` borrows and doesn't keep.** It only borrows objects for the
  length of the call, on the calling thread, under the caller's GIL. When a
  `default=` callback could run arbitrary Python code and mutate a container,
  items are also held by a reference for that span. Output is written
  straight into the result `bytes` object, so there is no final copy. String
  escaping scans 16 bytes at a time: SSE2 on x86_64 and NEON on aarch64, both
  part of those architectures' baseline, so no runtime detection is needed.
- **`str` contents are read from the object header, after a self-check.**
  Where CPython 3.12–3.14 already holds a string's UTF-8 (compact ASCII, or a
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
| `datetime` / `date` / `time` serialized natively | ❌ → `default=` | ✅ | ❌ |
| `uuid.UUID`, `enum.Enum`, dataclasses natively | ❌ → `default=` | ✅ | ❌ |
| numpy arrays (`OPT_SERIALIZE_NUMPY`) | ❌ raises | ✅ | ❌ |
| Non-`str` dict keys (`OPT_NON_STR_KEYS`) | ❌ raises | ✅ | ~ (coerced) |
| `orjson.Fragment` | ❌ | ✅ | ❌ |
| Integers beyond 64 bits in `dumps` | ❌ (like orjson) | ❌ | ✅ |
| NaN / ±Infinity in `dumps` | `null` (like orjson) | `null` | `NaN` / `Infinity` |
| Python versions | CPython 3.12–3.14 | CPython 3.10+ | all |

## Differences from orjson

These are all the known differences. Anything not listed produces the same
result as orjson 3.12 (the test suite checks this, see [Testing](#testing)).

- **Native `datetime`, `UUID`, `Enum`, dataclass, and numpy support is not
  implemented yet.** Such objects go to your `default=` callable, the same as
  any other unsupported type. Without a `default`, `dumps` raises
  `TypeError: Type is not JSON serializable: <type>`, with the same message as
  orjson.
- **`OPT_NON_STR_KEYS` and `OPT_SERIALIZE_NUMPY` raise** `TypeError: isojson
  does not support ...`. They are not silently ignored, because ignoring them
  would produce output you did not ask for.
- **Options that only affect datetimes and dataclasses are accepted and do
  nothing:** `OPT_NAIVE_UTC`, `OPT_OMIT_MICROSECONDS`, `OPT_UTC_Z`,
  `OPT_PASSTHROUGH_DATETIME`, `OPT_PASSTHROUGH_DATACLASS`. isojson never
  serializes those types natively, so these options can have no effect.
  `OPT_SERIALIZE_DATACLASS` and `OPT_SERIALIZE_UUID` are `0`, as in orjson.
- **`loads` error messages differ.** The exception type (`JSONDecodeError`,
  a subclass of `json.JSONDecodeError` and `ValueError`) and `.pos` / `.lineno`
  / `.colno` match. The wording of `.msg` does not always match orjson's.
- **CPython 3.12–3.14 only.** Per-interpreter GIL arrived in 3.12. On
  free-threaded builds (3.13t/3.14t) the module declares that it needs the
  GIL, so CPython re-enables it on import (see above).

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

## Testing

```bash
pip install maturin pytest orjson
maturin develop --release
pytest tests
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

The suite passes on macOS arm64 and Linux x86_64, both normally and under
`PYTHONMALLOC=debug`. The `-X dev` / `PYTHONDEVMODE=1` run is on Linux.

## Status

Version 0.1.0. The API is stable (it is orjson's). Native `datetime`, `UUID`,
`Enum`, and dataclass support is next.

## License

Apache-2.0.

Third-party: simd-json (Apache-2.0 OR MIT); `tests/data/JSONTestSuite` is from
[nst/JSONTestSuite](https://github.com/nst/JSONTestSuite) (MIT, license
included in that directory).
