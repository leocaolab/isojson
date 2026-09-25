# Design — isojson 0.2: native datetime and numpy

> Status: design, 2026-09-24. Consolidated rewrite after gate round 5, amended in rounds 6–12 (gate passed at round 12: 0 blocking): every rule has one
> home, and other sections reference it. **isojson is primary** (maintainer, 2026-09-24).
> Its rules follow §1a and orjson, never a caller's convenience. Callers adapt to isojson.
> SkyTrade is out of scope (maintainer). Format oracle: orjson 3.12.0 (commit `6737895`). Correctness
> oracle: the Python API (§1a). Audit history: `isojson-0.2-native-types-impl.md`, Step 3.
> Next: gate to zero blocking, then `impl-roadmap`.

## 1. Overview & goal

isojson 0.1 matches orjson byte-for-byte on the core JSON types and loads in strict own-GIL
sub-interpreters. Every other type goes to `default=`, and `OPT_SERIALIZE_NUMPY` raises
(`src/encode.rs:26-29`). Callers therefore re-implement orjson's datetime and numpy
support in their `default`, which is easy to get subtly wrong. One production
re-implementation writes `datetime64[ns]` arrays as integer nanoseconds.

0.2 serializes `datetime` / `date` / `time` and numpy natively, and makes the four
datetime options and `OPT_SERIALIZE_NUMPY` work. For everything orjson writes correctly,
the bytes are orjson's. Where orjson crashes or writes wrong data (§1b), isojson writes what
the Python API says.

The rule 0.1 is built on stays absolute: **no Python object in process-global storage**
(`src/lib.rs:4-10`). orjson breaks it for every type added here (§14). All type lookups go
into per-interpreter module state (C1).

**Non-goals**
- **UUID, Enum, dataclass:** Stage 2 (§15.1). They go to `default`, as in 0.1.
- **`OPT_NON_STR_KEYS`:** it still raises `isojson does not support OPT_NON_STR_KEYS`.
- **Strict mode** (NaN on encode, duplicate keys on decode): §15.2.
- **Datetime subclasses** (`pandas.Timestamp`): `default`, as orjson (exact-type
  dispatch).
- **Pure-Python datetimes** (`_pydatetime`): `default`. They have no C layout.
  - On 3.14 `_datetime` loads in strict sub-interpreters (measured on 3.14.7). (On 3.12 a
    strict sub-interpreter falls back to `_pydatetime` once main has imported `datetime`.)
  - (On 3.13 it is not safe *concurrently*: CPython 3.13.15's `_datetime` alone crashes when strict interpreters import or use it at the same time (measured in M1, isojson not involved; E2E-6). 3.14.7 is fine. isojson can't fix this; it's CPython's.)
- **CPython 3.12 and 3.13** (maintainer, 2026-09-24, during M2: "只支持 3.14"). 0.2 requires CPython 3.14: `requires-python >=3.14`, a `compile_error!` below 3.14, CI and wheels for 3.14 only. E2E-7 and `tests/test_subinterp_312_313.py` are removed; FR-14 records the change.
- **numpy objects isojson can't write:** FR-7, the single list.
- **Making numpy itself shareable across interpreters.** numpy fails strict import
  (measured), so each Pyronova worker has its own copy, and isojson uses the current
  interpreter's. numpy's own boundary belongs in the PyO3 fork's bench
  (`~/projects/pyo3/subinterp-bench/NUMPY.md`; not written yet, owned by the PyO3 fork work, O-3).
- **Free-threaded builds** (still `Py_MOD_GIL_USED`, `src/lib.rs:215-219`). `loads` is
  unchanged.

### 1a. Ground truth is the Python API

Correctness is judged against **Python's own API**: CPython's standard library, and numpy's
API for numpy objects. orjson is the reference for format conventions only. Where the two
disagree, the Python API wins; §1b lists every such case. The test helper
`tests/oracle/python_api.py::expected_text(x, opts)` encodes the rules below, and is the
only place tests get "correct" bytes from.

- **`datetime`: `dt.isoformat()`** after applying the options to the object:
  - `NAIVE_UTC` means `replace(tzinfo=timezone.utc)` if naive. **Naive ⇔ `dt.utcoffset()` is
    `None`**: a tzinfo whose `utcoffset` returns `None` counts as naive (DV-4a), so
    `NAIVE_UTC` adds `+00:00` (`Z` with `UTC_Z`) to it too.
  - `OMIT_MICROSECONDS` means `replace(microsecond=0)`.
  - `UTC_Z` writes a zero offset as `Z`.

  The offset comes from `dt.utcoffset()`, CPython's validated method. `None` means naive to
  Python, so no offset is written.
- **`date`: `d.isoformat()`.** No options apply.
- **`time`: `t.isoformat()`**, with `OMIT_MICROSECONDS` applied. `NAIVE_UTC` and `UTC_Z`
  don't apply to `time`, as in orjson. The offset comes from `t.utcoffset()`, validated.
- **UTC offsets:** `isoformat()`'s offset text, `±HH:MM[:SS[.ffffff]]` (§15.4).
- **`datetime64`:**
  - **The truth is the meaning numpy's API defines**:
    - `np.datetime_data(x.dtype)` gives `(unit, mult)`;
    - the value is `v × mult` units since 1970-01-01T00:00 UTC-naive;
    - sub-µs units are floored to µs, and Y/M go through floor `divmod`.

    `expected_text` computes this with exact Python integers. If the result is in
    0000-01-01 … 9999-12-31T23:59:59.999999, it renders it with
    `np.datetime_as_string(np.datetime64(us, "us"))`, re-rendered by the `datetime` rules:
    always `YYYY-MM-DDTHH:MM:SS`, `.ffffff` only if non-zero, plus the `NAIVE_UTC` /
    `UTC_Z` / `OMIT_MICROSECONDS` options. Out of range there is no answer: `expected_text`
    raises a typed `NoAnswer(reason)` exception (never a sentinel value), and the test then
    asserts the FR-7 decline instead.
  - **numpy's conversion functions are not the oracle** (rounds 10–11, measured):
    - `datetime_as_string(x, unit="us")` and `x.astype("M8[us]")` overflow near i64 MIN
      (`M8[ns]` MIN+1) and for multiplied units (`M8[10ns]` 1e18, ×3/×7 multipliers).
    - For Y/M they silently wrap (`M8[Y]` i64 max → `1969-01-01T10:33:36`).

    These are numpy's own limits, and the value's meaning is still defined.
  - NaT in a supported unit is written `null`, because `x.item()` is `None` (measured;
    §15.3).
- **Integer and bool numpy scalars:** `x.item()`.
- **Floats (Python and numpy):** only the value is the reference; the text follows
  orjson.
  - float64: `float(text) == x`.
  - float32/float16: `np.float32(text) == np.float32(x)` (bitwise), written as the
    shortest f32 digits.
  - `x.item()` is not the reference for floats: `np.float32(0.1).item()` is
    `0.10000000149011612`.
- **Where the Python API gives no answer in isojson's format**, isojson writes nothing and
  declines (FR-7). `expected_text` is undefined there; tests assert the decline.
  - Generic-unit `M8` holding a value: `np.datetime_data` gives unit `generic`, which has
    no unit ratio, so the value has no meaning. Plain `datetime_as_string(x)` agrees and
    raises. (`astype("M8[us]")` would invent one, treating it as µs, so it isn't used.)
    Generic NaT (`np.datetime64('NaT')`, the usual spelling) is NaT and is written `null`.
  - A value whose exact result falls outside years 0000–9999.

### 1b. Where isojson intentionally differs from orjson

Every "bug" row is orjson 3.12.0, measured with Python 3.13.15, numpy 2.5.3 and pytz on
macOS arm64 (re-measured in gate rounds 3–9). "Dies" means killed by a signal: SIGTRAP from
a Rust panic under `panic = "abort"`, or SIGSEGV. Each row has a regression test proving
both halves (E2E-4). README "Differences from orjson" mirrors this table; E2E-9 (3) checks the DV ids (C7).

| # | Input | orjson 3.12.0 | isojson 0.2 |
|---|---|---|---|
| DV-1 | UTC offset with non-zero seconds or microseconds: `+05:59:30`, `+05:00:00.000001`, `-00:00:01`, `-23:59:59.999999`; zoneinfo before standard time (`America/New_York` 1850 = `-04:56:02`) | drops them, rounding to the minute without carrying into hours: `+05:60`, `+05:00`, `-00:00` (RFC 3339 "offset unknown"), `+00:00` (sign lost), `-04:56` | as `isoformat()`: `+05:59:30`, `+05:00:00.000001`, `-00:00:01`, `-23:59:59.999999`, `-04:56:02` (§1a) |
| DV-2 | (merged into DV-1 in round 8; id kept) | | |
| DV-3 | pytz datetime after arithmetic, not normalized (`localize(2026-03-07 12:00 EST) + 2 days`) | `…T12:00:00-04:00`: the normalized offset on the un-normalized wall time, which is a different instant | `…T12:00:00-05:00` (`dt.utcoffset()`) |
| DV-4a | aware datetime, `utcoffset()` returns `None` | appends an invented offset: it reads the `None` as a timedelta, so `+00:00` (`Z` with `UTC_Z` on 3.14) on macOS and x86_64 Linux, garbage such as `+18:12` on aarch64 Linux (measured in M1 CI) | no offset (naive to Python) |
| DV-4b | `utcoffset()` raises (datetime) or is invalid (time) | datetime: dies (SIGSEGV) | `TypeError("<datetime\|time>.utcoffset() raised <Exc>: <msg>")`, with the exception as `__cause__` |
| DV-5 | `datetime64` NaT, unit `ns` | `"1677-09-21T00:12:43.145224"`, an invented date | `null` |
| DV-6 | `datetime64` NaT, units `W D h m` | `"1970-01-01T00:00:00"` (the multiply wraps) | `null` |
| DV-7 | `datetime64` NaT, units `Y M s ms us`, and generic NaT (`np.datetime64('NaT')`) | `TypeError: unrepresentable …` / `unsupported … unit: NaT` | `null` |
| DV-8 | multiplied units `M8[10ms]`, `M8[2D]` | dies (`unreachable!()`) | value × multiplier (`"1970-01-01T00:00:00.010000"`) |
| DV-9 | `M8[M]` before 1970 (`1969-12`, `1969-11`) | dies / `TypeError` | `"1969-12-01T00:00:00"` (floor division) |
| DV-10 | `M8[D…us]` on `9999-12-31` | `TypeError` (its range ends `9999-12-30T22:00`) | `"9999-12-31T00:00:00"` |
| DV-11 | `M8[W D h m]` values whose seconds overflow i64 (`M8[m]` 307445734561825861) | wraps to a wrong value (`…T00:00:44`) | declined, FR-7 (f) |
| DV-12 | a `datetime.time` with `tzinfo` set | `TypeError: datetime.time must not have tzinfo set` | `t.isoformat()`: `"01:00:00+00:00"`; zoneinfo (offset `None`) → `"01:00:00"` |
| DV-13 | generic-unit `M8` holding a value, not NaT (`np.array([5],'i8').view('M8')`) | `TypeError: unsupported numpy.datetime64 unit: NaT`, which misnames the value | declined, FR-7 (e) |
| DV-14 | a ≥2-D `M8` array where orjson's 1-D writer would raise: an element outside orjson's range, NaT in `Y M s ms us` or generic, a `ps`/`fs`/`as`/generic unit (e.g. `[["NaT"],["2026-01-01"]]` or `[["2026-01-01"],["10000-01-01"]]` in `M8[s]`) | **writes malformed JSON, raises nothing, even with `default`**: `{"x":[[,["2026-01-01T00:00:00"]],"y":1}` (child errors dropped at `orjson/src/serialize/numpy/array.rs:136`). (≥2-D NaT in `ns`/`W D h m` and overflow give valid JSON with invented values: DV-5/6/11; `M8[M]` 1969-12 and `M8[10ms]` die: DV-8/9) | per FR-9 / FR-7: NaT → `null`, representable values written, declined per FR-7 (e) (generic value) / (f) (element) |
| DV-15 | datetime whose `utcoffset()` returns a non-timedelta or ≥ 24h | no error, invented or corrupt offsets: int → `+00:00`; 24h → `+00:00`; 25h → `+01:00`; −25h → `+23:00`; a `str` is read as a timedelta → a garbage offset that varies per run (e.g. `+4294424370:4294967261`, `+139419:32`) | `TypeError("datetime.utcoffset() raised …")` with cause, as `dt.utcoffset()` raises (DV-4b's path). (A timedelta *subclass* is valid in both; not a divergence) |
| DV-16 | `datetime64` in `ps`, `fs` or `as` (`M8[ps]` 5, −1, i64 max) | `TypeError: unsupported numpy.datetime64 unit: picoseconds` (etc.), even with `default` | floored to µs per §1a: `1970-01-01T00:00:00`, `1969-12-31T23:59:59.999999`, `1970-04-17T18:02:52.036854`; NaT → `null` |
| DV-17 | a `datetime.time` whose microsecond has five digits (10000–99999), e.g. `time(0, 0, 1, 75652)` (found in M1 by E2E-2's random documents; `datetime` is unaffected) | drops the leading zero: `"00:00:01.75652"`, a different time (0.75652 s) | `t.isoformat()`: `"00:00:01.075652"` |

**Different from orjson, but not bugs:**
- Every numpy object isojson declines goes to `default` when one is given (FR-7). For 1-D
  arrays and scalars, orjson sends some of them to `default`, but raises for:
  - non-native-endian arrays;
  - out-of-range datetime64 values.

  For ≥2-D datetime64 arrays orjson writes malformed JSON instead (DV-14).
- 0.1 behaviour kept as it is:
  - argument-parsing messages (`dumps()` with no `obj`, unknown keywords) differ in
    wording from orjson's;
  - an explicit `default=None` means "no default". orjson *calls* `None` and raises
    `Type is not JSON serializable` with a `'NoneType' object is not callable` cause.
- UUID/Enum/dataclass (Stage 2), `orjson.Fragment` and `OPT_NON_STR_KEYS` go to
  `default` or raise, as in 0.1.
- `loads` error wording, as in 0.1.

## 2. Critical User Journeys (CUJs)

- **CUJ-1 — an orjson user switches.** Actor: a service serializing datetimes and numpy.
  Steps: call `isojson.dumps` with the options it used with orjson. Success: bytes equal
  orjson's, except §1b inputs (E2E-2, E2E-8).
- **CUJ-2 — numpy in many workers, each with its own numpy.** Actor: a Pyronova app in
  sub-interpreter mode on 3.14. numpy is cloned per worker, declared
  (`app.isolate("numpy")`) or reactively. Handlers call
  `isojson.dumps(x, option=OPT_SERIALIZE_NUMPY)` concurrently. Success:
  - every worker serializes its own numpy's objects natively;
  - no worker recognizes another's types;
  - SIGINT exits cleanly.
- **CUJ-3 — an input that breaks orjson.** Trigger: a §1b input. Success: the Python
  API's answer, or a decline (FR-7). Never a crash, never an invented value.
- **CUJ-4 — isojson loads everywhere and imports nothing.** Success:
  - strict import in own-GIL sub-interpreters on 3.14 with no override;
  - neither `import isojson` nor `dumps` imports `numpy`, `datetime` or `_datetime`.

## 3. Feature list

| Feature | Serves | Notes |
|---|---|---|
| F1 — per-interpreter type cache: look up, never import | CUJ-2, CUJ-4 | C1 |
| F2 — `datetime` / `date` / `time` + `NAIVE_UTC`, `UTC_Z`, `OMIT_MICROSECONDS`, `PASSTHROUGH_DATETIME` | CUJ-1, CUJ-3 | DV-1, DV-3…4b, DV-12, DV-15, DV-17 |
| F3 — numpy arrays and scalars (`OPT_SERIALIZE_NUMPY`), the decline rule | CUJ-1, CUJ-2, CUJ-3 | DV-5…11, DV-13, DV-14, DV-16 |
| F4 — f32 shortest float formatting | CUJ-1, CUJ-2 | |
| F5 — tests and CI | all | §9 |
| F6 — README, version, changelog | CUJ-3 | C7 |

## 4. Requirements

**Functional**

| # | Requirement | Feature |
|---|---|---|
| FR-1 | No Python object in process-global or thread-local storage. Every object `TypeCache` holds lives in `ModState`, is visited by `m_traverse` and is released by `m_clear` (C1). (0.1's `key_cache` holds only `str` keys, which can't form cycles, and is cleared, not traversed; unchanged.) Enforced by E2E-9 | F1 |
| FR-2 | Types are looked up in the current interpreter's `sys.modules`, never imported: datetime from `sys.modules["_datetime"]` (C module; its `datetime_CAPI` capsule is present on 3.13/3.14, measured), numpy from `sys.modules["numpy"]`. Absent module or attribute → the group stays Absent and is retried later | F1 |
| FR-3 | A cache hit is trusted (strong refs, so no address reuse). On a numpy-step miss, if `sys.modules["numpy"]` is not the cached module, the numpy group is re-read. The check runs at the first miss of each `dumps` call, not at every miss (M4, NFR-5): a numpy replaced by a `default` in the middle of a call is seen from the next call | F1, F3 |
| FR-4 | Dispatch: exact str, int, bool, None, float, list, dict, tuple; exact `datetime`, `date`, `time`; str/int/list/dict subclass flags (unless `PASSTHROUGH_SUBCLASS`); numpy (only with `OPT_SERIALIZE_NUMPY`); else `default`. orjson's Enum/dataclass steps are absent, so those reach `default` as before | F2, F3 |
| FR-5 | `datetime`, `date`, `time` text per §1a. `PASSTHROUGH_DATETIME` sends all three to `default`. `utcoffset()` failure → DV-4b | F2 |
| FR-6 | ndarray (exact type): read via `__array_struct__`.<br>• `two != 2` → `numpy array is malformed`.<br>• Leaves are read with `ptr::read_unaligned`, so unaligned C-contiguous arrays (`np.frombuffer(…, offset=1)`) are read correctly. orjson reads them through a typed slice, which is UB.<br>• A 0-length dim → literal `[]`, without indent.<br>• Nested dims under `OPT_INDENT_2` are indented like nested lists (encoder depth + dim), as orjson (measured).<br>• Nested dims → nested lists | F3 |
| FR-7 | **The decline rule (single home).** isojson *declines* an object it can't write:<br>(a) a non-C-contiguous array;<br>(b) a non-native-endian array;<br>(c) a 0-d array;<br>(d) an array dtype outside {b1, f2/4/8, i1/2/4/8, u1/2/4/8, M8};<br>(e) a generic-unit M8 value that is not NaT (generic NaT → `null`; decided per element, with (f)'s rollback), or an M8 array/scalar whose `dtype.str` can't be parsed;<br>(f) any M8 element or scalar that is unrepresentable (overflow, or a year outside 0000–9999; DV-11);<br>Checks (a)–(d) are applied in that order, first match wins, as orjson does (`array.rs:69-100`); the pure `classify(flags, nd, kind, itemsize) -> Result<Elem, Check>` in `decline.rs` implements it (`Check` names the rule; the caller builds the `Reason` with its evidence). A declined object goes **whole** to `default` when one is given. For (f) inside an array, output written for that array so far is rolled back first (`Out` truncated to the array's start). With no `default`, isojson raises the message for the reason (orjson's text where orjson has one; the new texts are those marked below), with `reason_note` attached via `add_note`: (a) `numpy array is not C contiguous; use ndarray.tolist() in default`; (b) `numpy array is not native-endianness`; (c)/(d) `unsupported datatype in numpy array`; (e) *new:* `unsupported numpy.datetime64 unit: generic` (orjson says `NaT`), or *new:* `unsupported numpy.datetime64 dtype: <raw dtype.str>`; (f) `unrepresentable numpy.datetime64: <v> <unit>` (orjson's unit word; *new:* ` × <mult>` when mult ≠ 1). Element-less arrays (any 0-length dim) are walked by FR-6 and write `[]` / `[[],[]]` as in orjson. If `default` itself raises, the error is orjson's (`Type is not JSON serializable: <type>`, cause = `default`'s exception), and the FR-7 reason is attached with `add_note`, with the raw evidence (`dtype.str`, flags, shape, value; `reason_note`), so it isn't lost. If `add_note` itself fails, that error is cleared and the `TypeError` stands |  F3 |
| FR-8 | numpy scalars: exact `float64/32/16`, `int8…64`, `uint8…64`, `bool_`, `datetime64`. Other numpy scalar types (`np.longdouble`, complex, and `np.longlong` where it is distinct from `int64` — it *is* `int64` on Windows) are not recognized and take the ordinary `default` path (no FR-7 note). `longdouble` *arrays* are platform-dependent via FR-7 (d): itemsize 8 (macOS arm64, Windows) is `f8` and written natively; itemsize 16 (Linux x86_64) is declined. Same as orjson (measured) | F3 |
| FR-9 | `datetime64` values, element or scalar: unit and multiplier from `dtype.str` (`<M8[10ms]`); NaT → `null` (DV-5/6/7); units `Y M W D h m s ms us ns ps fs as` × multiplier (DV-8, DV-16); months by floor division (DV-9); sub-µs (`ns`, `ps`, `fs`, `as`) floored to µs (−1 ns → `…23:59:59.999999`; §1a); all arithmetic checked in i128, days range-checked before `civil_from_days` (DV-10); failures → FR-7 (f). `OMIT_MICROSECONDS` applies; `PASSTHROUGH_DATETIME` does not (as orjson, §14) | F3 |
| FR-10 | float32 and float16 (widened exactly to f32) are written as the shortest f32 round-trip string, with zmij's f32 notation switch (as orjson); NaN/inf → `null` | F4 |
| FR-11 | `OPT_SERIALIZE_NUMPY` is accepted (removed from `UNSUPPORTED_OPTS`). Without it, numpy objects go to `default` as in 0.1. `OPT_NON_STR_KEYS` still raises. Existing-test change T-1 (impl doc, approved) | F3 |
| FR-12 | Messages new in 0.2 are defined in FR-7 and the DV rows. All other errors raised on the new paths are identical to orjson's (`orjson/src/serialize/error.rs:61-114`). 0.1's existing messages and `default=None` handling are unchanged (§1b "not bugs") | all |
| FR-13 | **Reentrancy.**<br>• At the start of `dumps`, the encoder resolves the datetime group, `dt = cache.datetime()`, and sets `guard = default given \\|\\| OPT_SERIALIZE_NUMPY \\|\\| dt loaded`.<br>• While `guard` is set, every item and key taken from a list or dict holds a reference while it is serialized (`guarded`, and `dict_in_order`'s no-default branch; §14) — except an item that is an exact `str`/`int`/`float`/`bool`/`None` (and its dict key): serializing those runs no Python, so nothing can mutate the container meanwhile (M4: the refcount write on every leaf cost ~40% on a list of 100k floats on x86_64).<br>• **Invariant:** with `guard` false, no Python code can run during the walk: no `default`, no numpy, no datetime type to call `utcoffset()` on. 0.1 has no Python-running path (verified by the round-3 auditor with instrumented subclasses).<br>• While guarded and the datetime snapshot is Absent, `cache.datetime()` is retried once per object that misses the fast path (E2E-10b).<br>• Each Python-running call site carries `debug_assert!(self.guard)` (C2) | F2, F3 |
| FR-14 | Public-behaviour change is recorded: for 0.1 callers, `datetime`/`date`/`time` no longer reach `default`, and the datetime options take effect. Pyronova's response paths (§7) serialize datetimes instead of raising `TypeError`. A caller that relied on its own `default` for datetimes and passes no options will see naive datetimes written without an offset. The CHANGELOG states this and names `OPT_NAIVE_UTC`, and that 0.2 requires CPython 3.14 (3.12/3.13 dropped). Version 0.2.0 and a CHANGELOG entry (C7) | F6 |

**Non-functional** (`bench/bench.py`, median of 7, release build, macOS arm64 and Linux
x86_64; a manual release gate)

| # | Requirement | Threshold | Feature |
|---|---|---|---|
| NFR-1 | 0.1 payloads with FR-13 guarding active | each existing cell within ±3% of 0.1 | F1–F4 |
| NFR-2 | datetime records (2000 × 3 aware/naive) | ≤ 1.2× orjson | F2 |
| NFR-3 | numpy f64 ×1M, f64 1000×1000, i64 ×1M | ≤ 1.2× orjson | F3 |
| NFR-4 | 10k numpy scalars (f64, i64) | ≤ 1.3× orjson | F3 |
| NFR-5 | `dumps([object()]*1000, default=str, option=OPT_SERIALIZE_NUMPY)`, numpy loaded | within 10% of 0.1 without the option | F1 |
| NFR-6 | concurrency, measured by E2E-6 (its parameters are the single definition) | 0 failures | F1 |

## 5. Infra

| Need | Exists? | Where / new |
|---|---|---|
| Rust cdylib, pyo3-ffi 0.29.2 (`Cargo.lock`), multi-phase init | ✅ | `Cargo.toml`, `src/lib.rs:201-246` |
| f32/f64 shortest formatting | ✅ | zmij 1.0.23 `format_finite<F>` (§14) |
| f16 → f32 | ➕ port | `f16_to_f32(bits: u16) -> f32`, ported from half-rs via orjson (§14), about 27 lines, MSRV-adapted (C5). Its proof is E2E-12 |
| datetime field reads | ✅ | pyo3-ffi layout accessors (§14); none go through the C-API global |
| numpy at build time | none | `PyArrayInterface` declared in Rust. No headers, no `import_array`, no `numpy` crate (§14) |
| test deps | ➕ | `pyproject.toml` test extra, **the single list**: `pytest`, `orjson==3.12.0`, `numpy>=2`, `pytz`, `tzdata` (CI `:48` installs only `maturin` plus this extra) |
| isojson CI | ✅ → changed | `.github/workflows/ci.yml`: the install step at `:48` installs the test extra read from `pyproject.toml` (a `tomllib` one-liner), so there is one dependency list (`:50` installs the wheel with `--no-index`); `:26-27` + `cargo test --lib`; + a debug-build pytest run (FR-13 asserts, E2E-13 alignment); + a non-blocking latest-orjson job |
| pyre CI | ✅ → changed | the integration job's install list (`pyre/.github/workflows/ci.yml:88`; the unit job at `:34` doesn't run E2E-3) adds `orjson==3.12.0` and `isojson @ git+https://github.com/leocaolab/isojson@<release-commit-sha>`; after PyPI, `isojson>=0.2`. The runtime dependency stays `isojson>=0.1` (§15.5). Platform coverage: §10 |
| pyre E2E harness | ✅ | `_start`, `_get`, `_sigint` from `tests.test_isolate_shared_ext` (§14), imported, not edited; clone-dir checks follow `test_isolate.py`'s (§14); `pyre/tests/e2e/README.md` reserves `tests/e2e/` for manual drivers, so the test goes in `tests/` |

## 6. Components

### C1 — Type cache (`src/types.rs`, new; `ModState` extended)
- **Responsibility:** FR-1…3.
- **Reuses** (§14): `ModState` / `state()`, the module lifecycle, and the
  strong-ref-in-state pattern of `decode_error`.
- **Lookup APIs (the single list):**
  Citations for every API named here are in §14.
  - `sys.modules`: `PyImport_GetModuleDict()` + `PyDict_GetItemRef` (strong ref; 3.12
    compat shim in pyo3-ffi). `PyImport_GetModule` is not used: it can wait on another
    thread's module-init lock.
  - The datetime capsule:
    - `PyModule_Check(_datetime)` (non-module → Absent);
    - `PyModule_GetDict` + `PyDict_GetItemRef(d, "datetime_CAPI")`, the same mechanism
      as numpy. It runs no Python (`datetime_CAPI` is in `_datetime.__dict__` on
      3.12–3.14, measured); missing → Absent;
    - `PyCapsule_GetPointer(cap, PyDateTime_CAPSULE_NAME)` into pyo3-ffi's
      `PyDateTime_CAPI`.

    A failing `GetPointer` (wrong or invalid capsule) clears the error and leaves the
    group Absent.
  - The 14 numpy types:
    - `PyModule_Check(numpy)` (non-module → Absent);
    - `PyModule_GetDict` + `PyDict_GetItemRef`, with keys created at group load;

    The `__dict__` read runs no Python (numpy's module `__getattr__` is never triggered),
    as orjson does. `PyDict_GetItemStringRef` is 3.13+ with no 3.12 compat, so it isn't
    used.
- **Interned names** (7, created in `init`; `datetime_CAPI` is now a dict key): `"_datetime"`, `"numpy"`, `"datetime_CAPI"`,
  `"utcoffset"`, `"__array_struct__"`, `"dtype"`, `"str"`. Null names (before `init`,
  after `m_clear`) mean every group is Absent.
- **Zero-validity:** a group is Absent iff its owner pointer is null (`dt_capsule`,
  `np_module`). CPython zero-fills module state, so a zeroed cache is valid and empty.
- **Interface:**
  ```rust
  #[repr(C)] pub(crate) struct TypeCache { dt_capsule: *mut PyObject, dt: DtTypes, np_module: *mut PyObject, np: NumpyTypes, names: Names }
  #[derive(Clone, Copy)] pub(crate) struct DtTypes { datetime: *mut PyTypeObject, date: *mut PyTypeObject, time: *mut PyTypeObject, delta: *mut PyTypeObject } // delta: utcoffset() results are checked against it (M4)
  #[derive(Clone, Copy)] pub(crate) struct NumpyTypes { ndarray: *mut PyTypeObject, /* 13 scalars */ }
  impl TypeCache {
      unsafe fn init(&mut self) -> c_int;
      unsafe fn datetime(&mut self) -> Result<Option<DtTypes>, PyErrSet>;
      unsafe fn numpy(&mut self, ty: *mut PyTypeObject) -> Result<Option<NumpyTypes>, PyErrSet>; // FR-3
      unsafe fn traverse(&self, visit: visitproc, arg: *mut c_void) -> c_int;
      unsafe fn clear(&mut self);                                                         // groups and names
  }
  ```
- **Invariant** (review-enforced; the `debug_assert!` sites in C2 mark where it applies):
  a copied `NumpyTypes` is re-fetched after any call that can run Python, since a nested
  `dumps` may replace the numpy group. `DtTypes` is exempt: the types are
  static on 3.13+, and the group is never replaced once Loaded.
- **Immortality premise:** `_datetime`'s types are shared by all interpreters. INCREF/DECREF
  on them from several own-GIL interpreters is race-free only because they are immortal
  (measured: refcount 4294967295 on 3.12/3.13; `sys._is_immortal` True on 3.14). E2E-6
  asserts this as a tripwire.

### C2 — Dispatch (`src/encode.rs`)
- **Responsibility:** FR-4, FR-11, FR-13.
- **Reuses** (§14): `Encoder`, `serialize`, `guarded`, `call_default`, `dict_in_order`,
  option parsing.
- **New:**
  - `Encoder` gains `cache: *mut TypeCache` from `state(module)` in `dumps` (§14) and
    `guard: bool` (FR-13).
  - Datetime checks go after `tuple`; numpy goes before `call_default`.
  - The `OPT_SERIALIZE_NUMPY` entry is dropped from `UNSUPPORTED_OPTS`.
  - `call_default` is split into `invoke_default(obj) -> Result<*mut PyObject, DefaultErr>`,
    where `enum DefaultErr { DepthLimit, Raised }` (the `MAX_DEFAULT_DEPTH` check, the call,
    and the cause on failure; both variants leave an exception set). `DepthLimit` gets no
    FR-7 note and serializing
    the result. `call_default` and `call_default_declined(obj, reason)` both use
    `invoke_default`. Only `call_default_declined` attaches the FR-7 note, and only when
    the `default` call itself raised.
  - `debug_assert!(self.guard)` sits immediately before each call that runs Python: the
    `PyObject_CallOneArg(default, …)` in `invoke_default` (reached only when a `default`
    exists), the `utcoffset()` call, and the `__array_struct__` / `dtype` reads.
  - `pub(crate)` for `numpy.rs`, the complete list: `Encoder` and its fields `out`, `opts`,
    `depth`, `guard`, `cache`; `indent`, `newline_indent`, `write_int` (below);
    `raise_type_error`, `raise_type_error_from_current`, `call_default_declined`.
  - `write_int<I: itoa::Integer>(out, v)` is extracted from `Encoder::int`'s two
    `itoa` + `small_copy` copies (`encode.rs:227-228, 234-235`) and reused by C4
    (DUP-R12-1).
  - `PyErrSet` / `R<T>` ("a Python exception is already set") move from `decode.rs` to
    `lib.rs` next to `ModState`, as `pub(crate)`; `decode.rs` imports them. They are the
    error type of every new `Result`-returning function. The bool/`Outcome`-returning
    encoder paths follow 0.1's convention (R12-A8, A9).

### C3 — Date/time: pure `src/datetime.rs`; Python extraction in `src/encode.rs`
- **Responsibility:** FR-5, FR-9. `datetime.rs` is the single home of date/time text and
  calendar math. It is **pure**: no `pyo3_ffi` import, no `PyObject`, no `Out` (E2E-9
  checks the import). Its formatters fill caller-owned fixed-size byte arrays and return a
  length. `encode.rs` and `numpy.rs` copy the result into `Out` with `small_copy`. Each
  piece is at most 16 bytes, within `small_copy`'s 32-byte limit.
- **Reuses** (§14): pyo3-ffi datetime/timedelta accessors (in encode.rs), `Out`,
  `small_copy`, `raise_type_error_from_current`.
- **Naive fast path** (round-1 D7, restored). `utcoffset_of` first reads the tzinfo field
  (`PyDateTime_DATE_GET_TZINFO` / `PyDateTime_TIME_GET_TZINFO`, which return `None` when
  `hastzinfo` is 0).
  - If it is `None`, the value is naive and no method is called.
  - Otherwise it does CPython's own `call_tzinfo_method` inline (M4, NFR-2): `tzinfo.utcoffset(arg)` (`arg` = the datetime, or `None` for a `time`); `None` is naive (§1a, DV-4a); a timedelta (or subclass, checked against the cached `timedelta` type) strictly between −24 h and 24 h is the offset. A raising tzinfo is what `obj.utcoffset()` would propagate (DV-4b). Any other result asks `obj.utcoffset()` itself, whose answer — normally CPython's own error — is the truth (DV-15). `datetime.utcoffset()` builds its call from a format string (~40 ns per value; measured: aware datetimes 1.30× → 0.45× orjson).
- **Interface:**
  ```rust
  // src/datetime.rs — pure
  pub(crate) fn fmt_ymd(buf: &mut [u8; 10], y: u16, mo: u8, d: u8) -> usize;                   // YYYY-MM-DD
  pub(crate) fn fmt_hms(buf: &mut [u8; 15], h: u8, mi: u8, s: u8, us: u32, opts: u32) -> usize;  // HH:MM:SS[.ffffff]
  pub(crate) fn fmt_offset(buf: &mut [u8; 16], total_us: i64, opts: u32) -> usize; // §8.2; zero → +00:00 / Z
  pub(crate) fn civil_from_days(days: i64) -> (i32, u8, u8);                         // caller range-checks
  pub(crate) enum Unit { Year, Month, Week, Day, Hour, Minute, Second, Milli, Micro, Nano, Pico, Femto, Atto, Generic }
  #[derive(Clone, Copy)] pub(crate) struct Dt64Unit { pub(crate) base: Unit, pub(crate) mult: i64 }
  #[derive(Clone, Copy)] pub(crate) struct Parts { pub(crate) y: u16, pub(crate) mo: u8, pub(crate) d: u8, pub(crate) h: u8, pub(crate) mi: u8, pub(crate) s: u8, pub(crate) us: u32 }
  pub(crate) fn parse_dt64(s: &[u8]) -> Option<Dt64Unit>;    // None → FR-7 (e)
  pub(crate) enum Dt64Err { Unrepresentable, GenericValue }    // → FR-7 (f) / (e)
  pub(crate) fn dt64_to_parts(v: i64, u: Dt64Unit) -> Result<Option<Parts>, Dt64Err>; // None = NaT; Err(Unrepresentable) → FR-7 (f); Err(GenericValue) → FR-7 (e)
  // src/encode.rs — Python side
  unsafe fn utcoffset_of(&mut self, obj: *mut PyObject, tzinfo: *mut PyObject, arg: *mut PyObject, what: &str, delta: *mut PyTypeObject) -> Result<Option<i64>, PyErrSet>; // obj.utcoffset() as total µs, DV-4b
  unsafe fn datetime(&mut self, obj: *mut PyObject) -> bool;
  unsafe fn date(&mut self, obj: *mut PyObject) -> bool;
  unsafe fn time(&mut self, obj: *mut PyObject) -> bool;  // opts & !(NAIVE_UTC | UTC_Z)
  ```

### C4 — numpy (`src/numpy.rs`, new)
- **Responsibility:** FR-6…9. Applies the decline rule (FR-7).
- **Reuses** (§14): C1, C3 (formatters copied into `Out`), C5, `itoa` + `small_copy` as in `Encoder::int`,
  `indent` / `newline_indent`, `Out::set_len` (the FR-7 (f) rollback).
- **Declines** are returned as `Outcome::Declined(reason)` and handled in one place,
  `Encoder::call_default_declined(obj, reason)` in `encode.rs`. It calls `default` (FR-7),
  raises `reason_message(reason)` without one, and attaches `reason_note(reason)` when `default` raises.
- **New:** `#[repr(C)] PyArrayInterface { two, nd, typekind, itemsize, flags, shape, strides, data, descr }`,
  read with `PyCapsule_GetPointer(cap, NULL)` (the capsule is unnamed and held for the
  walk). The scalar value read is O-2.
- **Interface:**
  ```rust
  pub(crate) enum Outcome { Written, Declined(Reason), Error(PyErrSet) }
  // src/decline.rs — pure (no pyo3_ffi; E2E-9), cargo-tested
  pub(crate) enum Reason {
      NotContiguous { flags: i32, shape: Vec<isize> }, NotNative { dtype: Vec<u8> },
      ZeroDim { dtype: Vec<u8> }, Dtype { dtype: Vec<u8> },
      GenericValue { v: i64 }, UnparseableDtype { dtype: Vec<u8> },
      Unrepresentable { v: i64, unit: Dt64Unit },
  }
  pub(crate) enum Check { NotContiguous, NotNative, ZeroDim, Dtype }                 // FR-7 (a)–(d)
  pub(crate) enum Elem { Bool, F16, F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Dt64 } // the dtype table
  pub(crate) fn classify(flags: i32, nd: i32, kind: u8, itemsize: i32) -> Result<Elem, Check>;
  pub(crate) fn reason_message(r: &Reason) -> String;   // FR-7's orjson-parity message
  pub(crate) fn reason_note(r: &Reason) -> String;      // the add_note text: message + raw evidence (dtype.str, flags, shape, v)
  pub(crate) unsafe fn serialize_array(enc: &mut Encoder, obj: *mut PyObject, np: NumpyTypes) -> Outcome;   // rolls back before Declined
  pub(crate) unsafe fn serialize_scalar(enc: &mut Encoder, obj: *mut PyObject, np: NumpyTypes) -> Option<Outcome>; // None = not a numpy scalar
  ```

### C5 — Floats (`src/float.rs`)
- **Responsibility:** FR-10.
- **Reuses** (§14): `write_f64`, `small_copy`, zmij `format_finite<F>`, half-rs's
  `f16_to_f32_fallback`.
- **New:**
  - `write_f32(out, f32)`: the finite check, like `write_f64`. Both call one generic
    `write_finite<F: zmij::Float>`. zmij's trait is sealed, so a generic `is_finite()` won't
    compile.
  - `f16_to_f32(bits)`: the §5 port, written for the crate's then-MSRV 1.85 (the MSRV is 1.88 since M4: simd-json 0.18 already required it). It uses `as` casts
    instead of `cast_signed`/`cast_unsigned` (stable since 1.87; §14 row), and no `unsafe`
    around `f32::from_bits`, which clippy `-D warnings` rejects.

### C6 — Tests and CI
The test list is §9. New helper modules:
- `tests/oracle/python_api.py` (`expected_text`, §1a);
- `tests/interp/strict.py`, a strict-interpreter `make()`/`run()` shim (3.14's `concurrent.interpreters`),
  modelled on the existing shim (§14). The existing shim file is not edited;
  T-4 in the impl doc covers a later merge.

Rust `#[cfg(test)]` (pure):
- exhaustive `civil_from_days` over [−719528, 2932896] against a day-by-day walk;
- `dt64_to_parts` at the DV-8…11 boundaries;
- a zeroed `TypeCache::clear()` no-op;
- `parse_dt64` on valid, multiplied and unparseable strings;
- `classify` over flag/nd/kind/itemsize combinations in orjson's order, and `reason_message` and `reason_note` for every `Reason` (in `decline.rs`), including the
  unparseable-dtype text, which numpy can't construct end-to-end.

This list is the single home of the Rust tests. CI changes: §5.

### C7 — README, version, changelog
- **README:**
  - The feature table (`README.md:203-220`): datetime and numpy rows.
  - "Differences from orjson" (`:222-245`): mirrors §1b (E2E-9 (3)).
    - Delete 0.1's "datetime not implemented" and "`OPT_SERIALIZE_NUMPY` raises".
    - The no-op-options bullet keeps only `OPT_PASSTHROUGH_DATACLASS`.
  - `:67-70`: add the type cache.
  - `:123-126` ("the only process-global pointers we touch are CPython's static builtin
    types and the immortal singletons"): add `_datetime`'s static types and C-API struct.
  - `:133-135` (items are held only when a `default` could run): replace with FR-13's
    guard condition.
  - The "only plain data is shared" table (`:61-65`): no new shared state in 0.2. The
    per-thread row is reworded to "per-thread scratch buffers and size hints" (E2E-9 (1)),
    which E2E-9 checks against `src/`.
  - Add §1b's 0.1-behaviour items (argument-parsing wording, `default=None`). Keep the
    existing "CPython 3.12–3.14 only / free-threaded" bullets (`:243-245`), which are
    limitations rather than orjson differences, under their own heading.
  - "Status" (`:398-401`).
  - "Third-party" (`:407-409`): credit half-rs.
  - State that `OPT_SERIALIZE_NUMPY` is tested with numpy ≥ 2 (numpy 1.x isn't; §11).
  - `:132` heading "`dumps` borrows and doesn't keep": `TypeCache` now keeps type
    references across calls; reword.
- **Crate doc:** `src/lib.rs:7-10` gets the README `:123-126` process-global-pointer
  correction.
- **Code comments:** `src/encode.rs:159-162` (`guarded`) and `:384-385` state the old
  "only with `default`" condition; replace with FR-13's guard.
- **Module doc:** `src/encode.rs:1-7` is rewritten: its "nothing is cached across calls
  except a thread-local byte buffer" becomes false (the per-interpreter `TypeCache`), and it
  gains the FR-13 guard.
- **Version and changelog:** `Cargo.toml` 0.2.0 and a new isojson `CHANGELOG.md` (FR-14,
  §1b). pyre's `CHANGELOG.md` gets an entry when its CI moves to 0.2.
- **Licensing** (orjson paths are relative to its repo root):
  - Port only orjson's Apache-2.0-OR-MIT files: `src/serialize/writer/half.rs`,
    `src/typeref.rs`, `src/serialize/numpy/array.rs`, `src/ffi/numpy/datetime.rs`.
  - Its MPL-2.0 files are behavioural references only: `src/ffi/numpy/scalar.rs`,
    `src/ffi/numpy/array.rs`, `src/serialize/numpy/datetime.rs`,
    `src/serialize/numpy/scalar.rs`, `src/serialize/numpy/typeref.rs`,
    `src/serialize/error.rs`, `src/serialize/writer/num.rs`.

## 7. Interfaces with other modules

| Direction | Module | Symbol | Purpose |
|---|---|---|---|
| calls → | CPython / pyo3-ffi | C1's lookup APIs; `PyObject_CallMethodNoArgs(obj, "utcoffset")`; `PyObject_GetAttr(arr, "__array_struct__" / "dtype")`; `PyObject_CallMethod(exc, "add_note", …)` (FR-7) | C1–C4 |
| calls → | pyo3-ffi | `PyDateTime_GET_*`, `PyDateTime_DATE_GET_*` (incl. `_TZINFO`), `PyDateTime_TIME_GET_*`, `PyDateTime_DELTA_GET_*` (§14). Banned symbols: E2E-9 (2) | fields |
| calls → | numpy (runtime) | `__array_struct__` (PyArrayInterface v2), `dtype.str` | data, datetime64 unit |
| calls → | zmij | `Buffer::format_finite<F>` | floats |
| ← called by | users | `isojson.dumps(obj, /, default=None, option=None)`, signature unchanged (`src/lib.rs:186`); behaviour per FR-14 | |
| ← called by | Pyronova | `isojson.dumps(obj)` (`pyre/src/python/worker.rs:883-894`); the `pyronova_json` helper (`pyre/src/response.rs:23-39`) | FR-14; options: O-1 |

## 8. Main algorithms

### 8.1 Type lookup (C1)
Control flow only. Every check (module type, `__dict__` lookups, capsule, null
names) is C1's, and is not restated here.
```
datetime():                          numpy(ty):
  Loaded → return dt                   Loaded and ty in np → return np
  else run C1's datetime load          sys.modules["numpy"] is the cached module → miss (None)
  failure → Absent (retried later)     else run C1's numpy load (replacing the group); return np if ty in np
```

### 8.2 Offset text: `isoformat()`'s
```
total_us: i64 = (days*86400 + secs)*1_000_000 + us    # up to ±8.64e10, needs i64
if total_us == 0: "Z" if UTC_Z else "+00:00"
sign, m = ('-', -total_us) if total_us < 0 else ('+', total_us)
hh, r = divmod(m, 3_600_000_000); mm, r = divmod(r, 60_000_000); ss, us = divmod(r, 1_000_000)
sign 2d(hh) ':' 2d(mm) [':' 2d(ss) if ss or us] ['.' 6d(us) if us]
```
Mirrors CPython's `format_utcoffset`. Identical to orjson for whole-minute offsets.

### 8.3 datetime64 (FR-9, FR-7)
```
unit = parse_dt64(dtype.str)          # None → FR-7 (e)
Generic: per element, NaT → null, any other value → Dt64Err::GenericValue → FR-7 (e) with (f)'s rollback
per element:
  v == i64::MIN → null
  n = v × mult (checked i128)
  Y: y = 1970+n    M: y = 1970+n.div_euclid(12), mo = n.rem_euclid(12)+1
  Y/M: require 0 <= y <= 9999, else decline (FR-7 f)
  else: us = n × us_per_unit for W D h m s ms us, or n.div_euclid(10^3 / 10^6 / 10^9 / 10^12) for ns / ps / fs / as;
        days = us.div_euclid(86_400e6)
        days ∈ [-719528, 2932896] → civil_from_days
  any failure → decline (FR-7 f): roll back to the array's start
fmt_ymd 'T' fmt_hms(opts), copied into Out; NAIVE_UTC → fmt_offset(0, opts)
```

### 8.4 Array walk (FR-6, FR-7)
```
cap = arr.__array_struct__; iface = PyCapsule_GetPointer(cap, NULL)
two != 2 → malformed
!C_CONTIGUOUS(0x1) | !NOTSWAPPED(0x200) | nd == 0 | dtype ∉ table → decline
start = out.len()
walk(depth, ptr): shape[depth] == 0 → "[]" ; '[' (leaf: read_unaligned element | walk(depth+1, ptr + i*strides[depth])) ']'
```
Depth ≤ `nd` (numpy caps it at 64), and doesn't count toward `MAX_DEPTH`.

## 9. Integration / E2E tests

"Expected" bytes come from `expected_text` (§1a), never from a hand-typed literal or from
orjson, except the tests that are explicitly orjson-parity tests (E2E-2, E2E-3). (There is
no E2E-1.)

| Test | CUJ | Setup → Action → Assertion |
|---|---|---|
| E2E-2 parity | CUJ-1 | datetime/date/time × {naive, UTC, +8, −5, zoneinfo, pytz} × all four datetime options, plus a datetime subclass (`class D(datetime)`) → `default`; numpy dtypes × shapes (1-D, 2-D, 0-length dims, non-contiguous, 0-d) with and without `default`; scalars; a new random-document generator (`rand_value` untouched). numpy also under `OPT_INDENT_2`. "Without default" means the argument is omitted (not `default=None`, §1b). Bytes and exception type/message equal orjson, **excluding** inputs matched by any DV row's predicate (by id) or by the §1b "not bugs" list |
| E2E-3 per-worker numpy (pyre) | CUJ-2 | `pyre/tests/test_isojson_numpy_workers.py` (harness §5), run declared and reactive. 4 workers; `/s/{seed}` returns `isojson.dumps(seeded payload, default=raise_, option=NUMPY).decode()`, the interpreter id and `np.__file__`. The payload holds only layouts isojson writes: C-contiguous native arrays (f64, f32, f16, i64, u8, bool, `M8[ns]`, 2-D) and scalars, with no NaT and no declined layouts. 256 requests × 16 threads; each equals the test process's `orjson.dumps` for that seed; ≥2 interpreters; every `np.__file__` under the copies dir; 4 clone dirs; SIGINT → rc 0; no "Fatal Python error". Fails, not skips, on a missing dependency |
| E2E-4 regression, one per DV row (`tests/test_divergence.py`) | CUJ-3 | DV-1, DV-3…16 (4a/4b both; DV-4a × every option combination without `NAIVE_UTC` (`UTC_Z`, `OMIT_MICROSECONDS`, both), where orjson writes `+00:00`/`Z` but isojson no offset — every combination with `NAIVE_UTC` agrees with orjson (measured in M1: `NAIVE_UTC\|UTC_Z` gives `Z` in both, correcting round 8) and lives in E2E-8 only; an invalid `time` offset: a tzinfo returning an int, and 25h; DV-14 with `M8[s]` NaT, out-of-range, `ps` and generic ≥2-D arrays; DV-15 with int, 24h, ±25h and `str` (for `str`, part (a) asserts no exception and output ≠ reference, since orjson's bytes vary per run); DV-16 with `ps`/`fs`/`as` 1-D; DV-17 at 10000, 75652 and 99999 µs). Three parts:<br>(a) orjson 3.12.0 fails: its wrong bytes or message; for DV-14, `json.loads` fails on orjson's bytes; for the death rows (DV-4b datetime, DV-8, DV-9 `1969-12`), a child with `returncode < 0` (POSIX; skipped on Windows with that reason);<br>(b) the reference = `expected_text`, or for declined rows the delegation to `default` and the FR-7 message;<br>(c) isojson == (b) |
| E2E-5 no imports | CUJ-4 | Fresh strict interpreter via `tests/interp/strict.py`: `import isojson; isojson.dumps([1,"x"], option=NUMPY)`, then assert `numpy`, `datetime` and `_datetime` were not added to `sys.modules`. Then with `datetime` imported and numpy absent, a non-JSON `x` goes to `default`. `del sys.modules["datetime"]` with `_datetime` loaded; a datetime is still native (FR-2) |
| E2E-6 concurrency | CUJ-4 | 3.14 via the shim (3.13 dropped in M1: CPython 3.13.15's own `_datetime` crashes under concurrent strict interpreters with isojson neither imported nor called — concurrent import SIGSEGV 5/5, debug-allocator corruption 10/10, this workload with serialized setup SIGABRT 10/10; 3.14.7 passes all): 4 and 8 interpreters (patterns in §14), 8 threads, 200 create/destroy cycles, with seeded aware/naive datetime payloads, each checking its own bytes, under `PYTHONMALLOC=debug`. Datetime types are process-static on 3.13+ (measured), so type separation is E2E-3's job. Tripwire: `sys._is_immortal(datetime.datetime)` (3.14) / refcount == 2**32−1 (3.12–3.13) |
| E2E-7 (removed: CPython 3.14 only) | CUJ-4 | 3.12 strict sub-interpreter after main imported `datetime`: `datetime` is `_pydatetime`; `dumps(dt, default=str)` calls `default`, and `dumps(dt)` raises `Type is not JSON serializable` |
| E2E-8 Python-API oracle | CUJ-1, CUJ-3 | Every datetime/date/time/datetime64 input from E2E-2/4, plus seeded random ones: for each unit × multipliers 1/2/3/7/10, values sampled over that unit's representable i64 range, including both i64 extremes and MIN+1 (ns/ps/fs/as cover only ~1677–2262 / ±106 d / ±2.5 h / ±9 s); Y/M at i64 max; offsets over ±23:59:59.999999, including µs; zoneinfo and pytz across DST and pre-standard-time dates; every combination of `NAIVE_UTC` × `UTC_Z` × `OMIT_MICROSECONDS` (`PASSTHROUGH_DATETIME` is E2E-2's). Assert `output == expected_text(x, opts)`, or the FR-7 (f) decline where `expected_text` raises `NoAnswer` byte for byte. Includes DV-4a tzinfos × every option, and a tzinfo returning a `timedelta` subclass. For years ≥ 0001 also assert `fromisoformat(output)` is the option-adjusted source with its `utcoffset()` (`fromisoformat` can't parse year 0000). Floats compared as in §1a |
| E2E-9 no new process-global state | CUJ-4 | `tests/test_no_global_pyobject.py`:<br>(1) every `static` / `thread_local!` item in `src/*.rs` (comments skipped) is in one of two reviewed allow-lists. **Constants** (`HEX`, `ESCAPE`, `METHODS`, `SLOTS`, `MODULE_DEF`) are exempt from README. **Shared runtime state**: each entry records the README table row that covers it (`:61-65`), and the test asserts that row's text is present. Entries: strfast's switch → the str fast-path row; decode scratch and `out::HINT` → the per-thread row (HINT is a size, not a buffer; the README row is reworded to "per-thread scratch buffers and size hints"). The simd-json row refers to a dependency and is exempt.<br>(2) **the banned-symbol list (single home)**, as regexes over `src/` with comments skipped:
• `\bPyDateTime_IMPORT\b`, `\bPyDateTimeAPI\b`, `\bPyDateTime_TimeZone_UTC\b`;
• `\bPy(Date|DateTime|Time|Delta|TZInfo|TimeZone)_(Check\w*|From\w*)\b` (all read pyo3-ffi's process-global `PyDateTimeAPI()`);
• `\bPyCapsule_Import\b`;
• `\bPyImport_(Import\w*|GetModule|AddModule\w*)\b`, with exactly one allowed site: `PyImport_ImportModule` in `module_exec` importing `json` (`src/lib.rs:71`);
• in `src/types.rs`: `\bPyObject_Get(Optional)?Attr\w*\b` (module attributes are read from `__dict__` only, C1);
• in `src/datetime.rs` and `src/decline.rs` (purity): `\bpyo3_ffi\b`, `\bPyObject\b`, `\bOut\b`, `crate::out`, and `use crate::\*`.<br>(3) README "Differences from orjson" lists every non-tombstone DV id of the §1b table (parsed from this design doc), and no other (C7). The rest of the README is mirrored by hand.<br>(4) FR-1's traverse: after a numpy `dumps`, `np.float64 in gc.get_referents(isojson.isojson)` (CPython's module traverse calls `m_traverse`) |
| E2E-10 reentrancy | CUJ-3 | In a fresh child process where this is the first `dumps` after `import datetime`: a tzinfo whose `utcoffset` clears the enclosing list or pops the enclosing dict (`[(dt,)]`, `{"a": [dt]}`), with no `default` and no numpy → no crash under `PYTHONMALLOC=debug`. Repeat with a `default`, with `OPT_SERIALIZE_NUMPY`, and with an aware `time`. **E2E-10b:** fresh process, `datetime` not imported; a `default` imports it and returns a datetime → serialized natively |
| E2E-11 numpy swap | CUJ-2 | Each case runs in a **fresh child process**:<br>(1) one `dumps(np.float64(1.0), option=NUMPY)` loads the group; then `sys.modules["numpy"]` becomes a module whose `float64` is `F = type("float64", (np.float64,), {})` (same scalar layout) and whose other names are numpy's. After a miss (a non-numpy object bound for `default`), `F(2.5)` is written natively as `2.5` (the re-read happened).<br>(2) A stub module missing attributes → objects go to `default`.<br>(3) A non-module object in `sys.modules["numpy"]` → Absent, no exception |
| E2E-12 f16 exhaustive (the f16 proof) | CUJ-1 | `a = np.arange(65536, dtype=np.uint16).view(np.float16)`; `isojson.dumps(a, option=NUMPY)`; each finite element compared bitwise, `np.float32(text).view(np.uint32) == a.astype(np.float32).view(np.uint32)` (catches `-0.0`); inf/NaN → `null` |
| E2E-13 declines | CUJ-3 | Every FR-7 case (a)–(f), with and without `default`; generic `np.zeros((2,0), 'M8')` → `[[],[]]` (the per-element (e) rule on an element-less array, as orjson); the unparseable-dtype message is covered by C6's Rust tests; generic `[NaT, 5]` 1-D and 2-D (null then rollback to `default`); unrecognized scalars (`np.complex128`, `np.longdouble`) → plain `default`, no note. With a `default`, it receives the whole original object; without one, the exact FR-7 message; with a raising `default`, orjson's message plus the note; in both raising cases the note equals `reason_note` (contains `dtype.str` / shape / value) for one case per `Reason`. Rollback (f): `{"a":1,"x":[0, arr2d]}` with the bad element in row 2, compact and `INDENT_2`, bytes equal the same document with `arr2d` replaced by `default(arr2d)`. Unaligned C-contiguous arrays (`np.frombuffer(b'\0'+data, dtype, offset=1)`) for f2, f4, f8, i2, i8, `M8[ns]` and 2-D are written correctly (FR-6). This test also runs in the debug-build job, where a typed-slice read would fail `from_raw_parts`' alignment check |

## 10. Success criteria
- [ ] FR-1…14 met (conformance check).
- [ ] NFR-1…6 on macOS arm64 and Linux x86_64, in the release notes.
- [ ] E2E-2…13 green on the CI matrix (E2E-4 death rows on POSIX); E2E-3 green, not
      skipped, in pyre CI (Linux), and run by hand on macOS before release.
- [ ] `cargo test --lib` green.
- [ ] C7 done.

## 11. Performance considerations
- **Fast path:** gains the exact `datetime` / `date` / `time` compares after `tuple`.
- **FR-13's per-item INCREF/DECREF** (guard condition: FR-13 only). NFR-1 measures it. If
  NFR-1 misses, the only alternative is a weaker guard, which is a design change.
- One `state()` per `dumps`.
- **numpy-step miss:** one `sys.modules` dict lookup, on an object bound for `default`
  (NFR-5).
- **Arrays:** contiguous leaf rows, a monomorphic loop per dtype, one `reserve` per row.
  `read_unaligned` compiles to a plain load on x86_64/aarch64.
- **f16:** software widening (float16 is rare in JSON payloads; hardware conversion can come later if NFR-3 misses).
- **O-2** (scalar read): pick the layout read if it's ≥2× faster than `__array_struct__`
  and E2E-3 passes on numpy 2.x. numpy 1.x isn't a target (3.13+ callers; numpy 1.26 has
  no 3.13 wheels).
- `bench/bench.py` gains: datetime records, numpy f64 1M, 2-D, scalars 10k.

## 12. Reliability considerations
- **Never crash, never invent** (CUJ-3):
  - every orjson crash or UB path (§1b, unaligned reads) is a checked error or a correct
    read;
  - every getattr and call result is NULL-checked;
  - errors are `TypeError`, carry the cause, and say what happened.
- **Reentrancy:** FR-13, E2E-10.
- **numpy memory:** the capsule is held for the walk; reads stay within numpy's
  `shape × strides`; unaligned-safe; `two != 2` → malformed.
- **Stale types:** strong refs; a replaced numpy is detected on a miss (E2E-11).
- **Teardown:** `module_clear` releases groups and names; zero-valid state; E2E-6 runs 200
  cycles under `PYTHONMALLOC=debug`.
- **Fail closed:** FR-7. isojson never makes a best-effort read.

## 13. Security considerations
- Caller-owned objects: attacker-shaped data, not attacker-chosen types. Exact-type
  detection means an ndarray subclass can't feed a forged interface.
- Reads are bounded by the interface; datetime64 math is checked; offset text is bounded.
- Type lookup trusts `sys.modules`, exactly as orjson does (it imports whatever is
  registered). Whoever replaces `sys.modules["numpy"]` or `_datetime` with something that
  isn't numpy or datetime owns the consequences. isojson doesn't validate the replacement
  (maintainer, 2026-09-24).
- No new shared Python state (E2E-6, E2E-9). No secrets, I/O or network.

## 14. Abstraction & reuse

**Approach:** 0.1's architecture (one `Encoder`, one pass, `Out` writing into the result
`bytes`), plus per-interpreter type recognition (C1), a pure date/time module (C3), and a
numpy walker (C4).

**Reuse map** — the single home of `file:line` citations for reused code; component
sections name the symbols and refer here.

| Symbol | Location | Use |
|---|---|---|
| `ModState`, `state()` | `src/lib.rs:32-43` | holds `TypeCache` |
| module lifecycle | `src/lib.rs:64-176` | init; visit/release |
| option constants | `src/lib.rs:47-58` | orjson's values |
| `Encoder`, `serialize`, `enter`, `guarded`, `call_default` | `src/encode.rs:76-199` | dispatch, guard |
| `dict_in_order` (no-default branch `:379-382`) | `src/encode.rs:369-396` | FR-13 |
| `Encoder::int` (`itoa` + `small_copy`) | `src/encode.rs:214-246` | C4 |
| `dumps` (where `Encoder` is built) | `src/encode.rs:577-666` | C2 |
| orjson `PASSTHROUGH_DATETIME` uses | `orjson/src/serialize/writer/container.rs:600, 617, 631` | FR-9 |
| concurrency patterns | `tests/test_subinterp.py:52-88` (`WORKER`), `:91`, `:111-127` | E2E-6 |
| pyre clone-dir checks | `pyre/tests/test_isolate.py:106, :202` | E2E-3 |
| error helpers | `src/encode.rs:55-74` | messages, causes |
| `UNSUPPORTED_OPTS` + its comment, option parsing | `src/encode.rs:26-33, 628-647` | FR-11 |
| `PyErrSet`, `R<T>` (moved to `lib.rs`) | `src/decode.rs:25-27` | all new `Result` functions |
| orjson array check order | `orjson/src/serialize/numpy/array.rs:69-100` | FR-7 `classify` |
| `write_f64`, `small_copy` (≤ 32 bytes, `:22`) | `src/float.rs:9-45` | C3, C4, C5 |
| `Out`, `set_len` | `src/out.rs:33-156`, `:73` | output, FR-7 rollback |
| zmij `format_finite<F>`, `impl Float for f32` | `zmij-1.0.23/src/lib.rs:1764`, `:1781-1782` | f32/f64 |
| half-rs `f16_to_f32_fallback` | `orjson/src/serialize/writer/half.rs:80-107` (Apache-2.0 OR MIT); the post-1.85 calls at `:97, :104` | ported, MSRV-adapted |
| `PyDateTime_CAPI`, `PyDateTime_CAPSULE_NAME` | `pyo3-ffi-0.29.2/src/datetime.rs:523, :598` | capsule |
| `PyDict_GetItemRef` (+3.12 compat), `PyModule_Check`, `PyModule_GetDict` | `pyo3-ffi-0.29.2/src/dictobject.rs:77` (`compat/py_3_13.rs:5`), `moduleobject.rs:20` (`PyModule_Check`), `:41` (`PyModule_GetDict`) | C1 |
| numpy module `__dict__` read (behavioural reference) | `orjson/src/typeref.rs:204-240` | C1 |
| datetime/timedelta accessors | `pyo3-ffi-0.29.2/src/datetime.rs:124-150, 202-300, 321-347` | C3 |
| `_err` | `tests/test_parity.py:138-143` | E2E-2 |
| strict-interp shim pattern | `tests/test_subinterp_312_313.py:10-29` | `tests/interp/strict.py` |
| pyre `_start`, `_get`, `_sigint` | `pyre/tests/test_isolate_shared_ext.py:83-117` | E2E-3 |

**Not copied from orjson** (process-global state or crashes). The enforceable symbol list is E2E-9 (2); this list gives the reasons:
- `static mut` type pointers initialised under `INIT: OnceLock` (`orjson/src/typeref.rs:17-62, :106`);
- `PyDateTime_IMPORT` (`:80-92`);
- `PyImport_ImportModule("numpy")` with failures cached forever (`:200-245`);
- the capsule-struct poke (`src/serialize/numpy/array.rs:63-67`);
- typed-slice reads of possibly unaligned data;
- `unreachable!()` / `unwrap()` on user data (`ffi/numpy/datetime.rs:118`);
- unchecked `utcoffset`;
- the tz-library branch order (DV-3).

The `numpy` crate is not used. It is a PyO3-level binding, isojson is built on pyo3-ffi
only, and `import_array` is exactly the numpy C-API dependency that `__array_struct__`
avoids.

**Crates rejected:**
- `half`: a non-optional `zerocopy` with `derive` (a proc-macro) plus `cfg-if`
  (`half-2.7.1/Cargo.toml:83, :113-117`), for one function orjson carries portably.
- `chrono`, `time`, `jiff`, `jiff-core`: own range semantics (jiff's range is DV-10's
  cause); one `civil_from_days` suffices and is proven exhaustively.

**New abstractions:**
- `TypeCache`;
- the pure date/time module (`fmt_ymd` / `fmt_hms` / `fmt_offset` /
  `civil_from_days` / `Dt64Unit` / `dt64_to_parts`);
- `write_finite<F>`;
- the FR-7 decline rule;
- `expected_text`.

## 15. Decisions

### 15.1 Stage 2: UUID, Enum, dataclass
Not in 0.2; they go to `default`, as in 0.1. A later release.

Rules agreed if Stage 2 happens:
- Enum → `.value` (including IntEnum/StrEnum and `EnumType` metaclass subclasses); a
  failure is a `TypeError` with cause.
- dataclass detected through the MRO; a `""` attribute is written as a `""` key; an unset
  `__slots__` field → `TypeError` with cause.
- UUID: exact type, `8-4-4-4-12`.
- Each rule gets a DV row and an E2E-4 test.

### 15.2 Strict mode: not in 0.2
Encode raising on NaN/inf and decode rejecting duplicate keys are not orjson API. They'd
be isojson's first API beyond orjson's. Revisit when a caller needs them.

### 15.3 D-NaT: NaT → `null`, by the Python API
numpy 2.5 emits a `DeprecationWarning` for generic units, including `np.datetime64('NaT')`
("will raise an error in the future"). Tests filter it; if numpy removes generic units, the
generic rules in FR-7 (e) become dead code.

`x.item()` is `None` for NaT in every supported unit (measured). This is consistent with
NaN and with pandas `to_json`. orjson's invented dates and unit-dependent errors contradict
it (DV-5/6/7).

### 15.4 D-offset: offsets with seconds are written exactly
The datetime object is the ground truth. `isoformat()` writes the offset exactly and
`fromisoformat()` reads it back (measured). Rounding to the minute would silently rewrite
the data by up to 30 s, so it is rejected. The cost is that those values aren't strict RFC
3339, and JS `Date` rejects them (measured, Node 24.14). That failure is in the open.
Modern zones are whole minutes and are unchanged from orjson.

### 15.5 Judgment calls taken during the gate (current state, reversible)
Round numbers are in brackets.

- **datetime:**
  - DV-4a: `utcoffset()` → `None` is written naive [1].
  - DV-4b, DV-12 and DV-13 messages say what happened [1, 3, 4].
  - `time` → `isoformat()` via validated `utcoffset()` [2–3].
  - Naive ⇔ `utcoffset()` is `None`, for `NAIVE_UTC` too [7].
  - Invalid datetime offsets raise (DV-15) [7].
- **numpy:**
  - One decline rule (FR-7), including element errors, with rollback [4–5].
  - NaT → `null` in supported units and for generic NaT; generic values are declined [4, 7].
  - isojson trusts `sys.modules`, as orjson does. A replaced numpy is the caller's
    problem [7, 9; maintainer].
  - A raising `default` keeps orjson's error, and the decline reason is attached with
    `add_note` [6].
  - Unaligned arrays are read correctly [5].
- **Rust:**
  - f16 ported from half-rs and MSRV-adapted; `half` rejected [2–4].
  - Pure `datetime.rs` (formatters fill byte arrays), including dt64 math [3–4, 7].
  - numpy types read from the module `__dict__` [3].
  - `PyDict_GetItemRef` rather than `PyImport_GetModule` [1, 4].
  - `civil_from_days` hand-written [1].
- **Deployment:**
  - Pyronova's runtime pin stays `>=0.1`; only its CI and tests need 0.2 [3].
  - CI pins `orjson==3.12.0`, plus a non-blocking latest job [1].
  - SkyTrade out of scope (maintainer).
- **Tests:**
  - Datetime subclasses are tested with `class D(datetime)`; pandas is not a test
    dependency [6].
  - FR-3 trusts a cache hit, and E2E-11 forces the miss [6].
  - FR-12 covers only new messages; 0.1's messages and `default=None` are kept [7].
- **Rounds 10–11:**
  - Round 12: the note is attached with or without a `default`; `PyErrSet` moved to `lib.rs`.
  - The datetime64 oracle is numpy's defined meaning, computed exactly (§1a); numpy's
    conversion functions overflow or wrap outside their domain.
  - `Dt64Err::GenericValue`.
  - Declines are returned as `Outcome::Declined(Reason)`. `Reason` carries raw evidence
    for the note, in a pure `decline.rs`.
  - `PyErrSet` is reused.
- **M1 build:** DV-17 added (orjson drops a `time`'s leading microsecond zero); isojson already wrote `isoformat()`, so no behaviour changed.
- **Round 9:** `ps`/`fs`/`as` are written floored to µs, per the Python API (DV-16); only generic non-NaT values remain declined in FR-7 (e).
- **Round 8:**
  - Unrecognized numpy scalars are plain `default` (FR-7 (g) dropped).
  - `_datetime` is read from `__dict__`, like numpy.
  - DV-2 was merged into DV-1.


## 16. Open items
- **O-1 (Pyronova):** should Pyronova pass `OPT_SERIALIZE_NUMPY` or datetime options for
  handler return values? It has two per-interpreter isojson caches to set (§7's
  Pyronova row; the main-interpreter one has its own `_default`). To be decided after 0.2.
- **O-2 (decided in M2):** the layout read. Measured on macOS arm64, 10k scalars: `__array_struct__` per scalar 4.8× (f64) / 6.8× (i64) orjson's time; the layout read (value right after the object header, as orjson) 0.76× / 0.78×, i.e. ≥2× faster (§11's criterion). E2E-3 on numpy 2.x is still M3's.
- **O-3:** numpy's own capability boundary (`~/projects/pyo3/subinterp-bench/NUMPY.md`),
  owned by the PyO3 fork work, not by isojson.

