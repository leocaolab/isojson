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

```python
import isojson

isojson.dumps({"a": [1, 2.5, None]})          # b'{"a":[1,2.5,null]}'
isojson.loads(b'{"a":[1,2.5,null]}')           # {'a': [1, 2.5, None]}
isojson.dumps(obj, default=str, option=isojson.OPT_SORT_KEYS | isojson.OPT_INDENT_2)
```

Switching from orjson is usually a one-line change: `import isojson as orjson`.
Check the [differences](#differences-from-orjson) first.

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

## How it stays safe

- **Nothing process-global holds a Python object.** Per-interpreter state
  (the `JSONDecodeError` type and the dict-key cache) lives in the module's
  state, which CPython allocates per interpreter and frees with it. The only
  process-global pointers isojson touches are CPython's static builtin types
  and the immortal singletons `None`/`True`/`False`, which every interpreter
  shares by design.
- **The key cache is per interpreter.** Like orjson, `loads` caches recently
  seen dict keys, so repeated keys reuse one `str` with its hash already
  computed. orjson keeps that cache process-wide. isojson keeps one per
  interpreter, because a shared cache would hand one interpreter's objects to
  another.
- **Borrow, don't keep.** `dumps` only borrows objects for the length of the
  call, on the calling thread, under the caller's GIL. When a `default=`
  callback could run arbitrary Python code and mutate a container, items are
  also held by a reference for that span.
- **Exceptions belong to the interpreter.** `isojson.JSONDecodeError`
  subclasses the calling interpreter's own `json.JSONDecodeError`.
- **The parser uses no recursion.** `loads` keeps nesting on a heap stack, so
  a document nested 1024 deep is safe even on threads with small C stacks.
- **No PyO3.** The module is written against the raw C API (`pyo3-ffi`).
  PyO3's high-level layer caches type objects and modules in process-global
  statics and rejects a second interpreter, which is the problem this package
  exists to avoid.

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
  free-threaded builds (3.13t/3.14t) the module declares that it needs the GIL
  rather than claiming it doesn't.

## Performance

Apple M5 Pro (18 cores), macOS, CPython 3.14.7, isojson 0.1.0, orjson 3.12.0.
Each cell is the median of repeated runs. Reproduce with
`python bench/bench.py`.

### Parallel: the case isojson is built for

Each worker does 3,000 round trips (`loads(dumps(doc))`) of a 100-record
document. The clock covers only the work loop: interpreter or process creation
and imports happen before timing starts. Throughput is in round trips per
second; higher is better.

| setup | N=1 | N=2 | N=4 | N=8 | scaling 1→8 |
|---|---:|---:|---:|---:|---:|
| **isojson, N own-GIL sub-interpreters, one process** | 19,316 | 37,667 | 64,965 | **117,274** | **6.07×** |
| json (stdlib), N own-GIL sub-interpreters | 4,673 | 8,898 | 16,153 | 29,332 | 6.28× |
| orjson, N own-GIL sub-interpreters | ✗ | ✗ | ✗ | ✗ | — |
| orjson, N threads, one interpreter (shared GIL) | 22,780 | 23,389 | 22,727 | 23,872 | 1.05× |
| isojson, N threads, one interpreter (shared GIL) | 19,530 | 19,477 | 19,396 | 19,116 | 0.98× |
| orjson, N processes (multiprocessing) | 23,370 | 43,621 | 81,240 | 149,781 | 6.41× |
| isojson, N processes (multiprocessing) | 19,131 | 36,212 | 66,764 | 125,719 | 6.57× |

✗ `ImportError: module orjson.orjson does not support loading in subinterpreters`

How to read this:

- **In one process, isojson on 8 sub-interpreters is 4.9× the best orjson can
  do** (117.3k vs 23.9k round trips/s). Adding threads to orjson gains nothing
  because every thread shares one GIL.
- **isojson on sub-interpreters is 4.0× stdlib `json` on sub-interpreters**,
  and stdlib `json` was the only other option that works there.
- **isojson on sub-interpreters scales nearly as well as isojson on
  processes** (117.3k vs 125.7k at N=8). Sub-interpreters get close to
  process-level scaling with one process, one address space, and shared
  memory.
- **orjson on 8 processes is still faster** (149.8k) because orjson is faster
  per call on this document. If you already run multiprocessing and never use
  sub-interpreters, orjson remains the faster choice.

### Single interpreter: per-call cost

`dumps` (lower is better):

| payload | isojson | orjson | json (stdlib) | isojson / orjson |
|---|---:|---:|---:|---:|
| small (27 B) | 43 ns | 46 ns | 601 ns | 0.94× |
| records ×100 | 20.24 µs | 17.10 µs | 154.86 µs | 1.18× |
| records ×2000 | 390.11 µs | 298.73 µs | 2.86 ms | 1.31× |
| floats ×10k | 98.08 µs | 205.63 µs | 2.28 ms | **0.48×** |
| unicode/escapes ×200 | 17.57 µs | 11.39 µs | 116.35 µs | 1.54× |

`loads` (lower is better):

| payload | isojson | orjson | json (stdlib) | isojson / orjson |
|---|---:|---:|---:|---:|
| small (27 B) | 82 ns | 80 ns | 562 ns | 1.02× |
| records ×100 | 52.79 µs | 40.12 µs | 108.58 µs | 1.32× |
| records ×2000 | 1.10 ms | 803.72 µs | 2.15 ms | 1.37× |
| floats ×10k | 293.54 µs | 171.09 µs | 1.14 ms | 1.72× |
| unicode/escapes ×200 | 59.87 µs | 45.74 µs | 126.34 µs | 1.31× |

In summary: isojson ties orjson on tiny documents and is about 2× faster on
float-heavy `dumps`. Elsewhere it is 1.2–1.7× slower per call. It is 2–23×
faster than stdlib `json` everywhere.

Part of the gap to orjson is a deliberate choice. orjson reads CPython's
internal object layouts directly, and isojson goes through the C API. That
costs time per call but keeps the extension off version-specific struct
layouts. There is still room to speed it up, mainly in `loads` number
parsing, in dict iteration, and in the final copy into the result `bytes`.

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
- **Multi-interpreter safety** (`tests/test_subinterp.py`):
  - strict import in 6 own-GIL sub-interpreters with no override;
  - per-interpreter module state and exception types;
  - 4 and 8 interpreters running concurrently, each on its own seeded data
    with its own expected answer, with a Python `default=` callback running
    inside the serializer on every call. A leak between interpreters would
    show up as a wrong answer, not just a possible crash;
  - 200 create/use/destroy cycles in a child process, so a crash is reported
    instead of swallowed.

## Status

Version 0.1.0. The API is stable (it is orjson's). Native `datetime`, `UUID`,
`Enum`, and dataclass support is next. Not yet published on PyPI. Build from
source with `maturin`.

## License

Apache-2.0
