# Impl design — isojson 0.2: native datetime and numpy

> Implements `docs/design/isojson-0.2-native-types.md` ("the design"; consolidated in
> round 5). SkyTrade is out of scope (maintainer, 2026-09-24); the SkyTrade plan and fixture were removed from this doc set. Three parts: CUJ
> walkthroughs (step 1), mapping (step 2), audit record (step 3). Steps 1–2 were rewritten
> in round 5 to match the consolidated design. Each rule is referenced by its design id and
> is not restated. **Gate scope (maintainer, 2026-09-24): isojson only; SkyTrade is not considered.** **No code before the gate reaches zero blocking.**

## Step 1 — CUJ walkthroughs

### CUJ-1 — an orjson user switches

The caller runs `isojson.dumps(data, default=…, option=OPT_SERIALIZE_NUMPY | OPT_NAIVE_UTC)`.

1. `dumps` parses arguments as today. The option check no longer rejects
   `OPT_SERIALIZE_NUMPY` (FR-11).
2. It takes the module's `ModState` (from the `_module` argument it ignores today),
   resolves the datetime group (C1) and sets `guard` (FR-13).
3. The walk runs as in 0.1 for builtins. After `tuple`, it compares an object's type with
   the exact datetime types (FR-4).
   - A `datetime`'s fields are read with the pyo3-ffi accessors.
   - Its offset comes from `dt.utcoffset()` through `utcoffset_of`.
   - The pure `datetime.rs` formats the pieces, which `encode.rs` copies into `Out`
     (C3).
4. With the option on, numpy objects reach C4.
   - A scalar of an FR-8 type is written directly: floats through `write_f64` or
     `write_f32`, f16 widened first.
   - An `ndarray` is read through `__array_struct__` and walked (FR-6), or declined as a
     whole per FR-7.
5. Anything else goes to `default` as in 0.1. That includes datetime subclasses,
   UUID/Enum/dataclass, and declined numpy objects.


### CUJ-2 — each worker serializes numpy from its own numpy copy

Pyronova (3.13+) clones numpy into each worker, declared or reactive. isojson loads shared,
but each interpreter has its own module object and `ModState`, so its own `TypeCache`. A
handler's `dumps` in worker 3 looks up worker 3's `sys.modules["numpy"]` (the clone). It
therefore only ever matches worker 3's types.

If `sys.modules["numpy"]` is replaced, the next numpy-step miss re-reads the group (FR-3).
On SIGINT, `m_clear` releases the capsule, module, types and names.

### CUJ-3 — an input on which orjson is wrong

Examples:
- NaT in `M8[ns]` → `null` (DV-5).
- `M8[M]` `1969-12` → floor division (DV-9).
- New York in 1850 → `-04:56:02` (DV-1).
- An overflowing `M8[m]` value → declined (FR-7 f).

Each has an E2E-4 test in three parts:
- (a) orjson 3.12.0 fails: its bytes or message, or a child killed by a signal;
- (b) the reference, from `expected_text`, or for declines the delegation and the message;
- (c) isojson == (b).

### CUJ-4 — isojson loads everywhere and imports nothing

- `module_exec` does what it did in 0.1, plus `TypeCache::init` (design C1: names,
  zero-validity).
- `dumps` never imports anything; it only looks things up in `sys.modules`.
- On 3.12, after main has imported `datetime`, a strict sub-interpreter's datetimes are
  `_pydatetime` and go to `default` (E2E-7).

## Step 2 — Mapping

**Placement contract:** `grep -rln -E 'Belongs here|Does NOT belong|Placement'` finds no
module README contract in isojson or pyre. The only placement-like note is
`pyre/tests/e2e/README.md` (manual drivers only); M22 goes in `pyre/tests/`, which conforms. Rounds 1–5 re-ran it and confirmed.
**Every row is 无契约.** isojson's placement rule is the crate doc `src/lib.rs:1-13`,
enforced by E2E-9.

### Searches (proof of absence)

```
$ grep -rn -i -E "datetime|PyDateTime|numpy|ndarray|array_struct|capsule|f32|f16|half|TypeCache|sys.modules|utcoffset|isoformat|civil|\bNaT\b|Dt64|PyArrayInterface|write_finite|fmt_(ymd|hms|offset)|read_unaligned|align|basicsize|add_note" src/ python/ tests/ bench/ build.rs
→ (re-run in round 11, pattern + `Outcome|Reason|UnitErr|Declined|PyErrSet|Result<`) src/encode.rs:28,30-31 (UNSUPPORTED_OPTS + comment), :512 (write_unaligned, unrelated); src/decode.rs:25-27 `PyErrSet` / `R<T>` → reused (DUP-R11-3); src/lib.rs:51,56,101,104 and python/isojson/__init__.py(i) (option names); tests/test_parity.py:227 (T-1), :303 (unrelated). No datetime/numpy/type-cache/decline code exists.
$ ls ~/.cargo/registry/src/*/ | grep -E "^(half|chrono|jiff|time|numpy)-"   → present; all rejected (design §14)
$ orjson source (writer/half.rs, typeref.rs, serialize/numpy/*, ffi/numpy/*) → f16 fallback ported; the rest not copied (design §14, licensing C7)
$ pyo3-ffi 0.29.2: PyDict_GetItemRef, PyModule_Check, PyModule_GetDict, PyDateTime_CAPI, PyDateTime_CAPSULE_NAME → reused (C1; attributes are read from __dict__ only since R8-A1)
$ pyre tests/: no isojson test; tests/test_isolate_shared_ext.py has _start/_get/_sigint → reused
```

### Design-corpus sweep

```
$ ls isojson/docs/design/ pyre/docs/design/
isojson-0.2-native-types.md  isojson-0.2-native-types-impl.md
polars-in-subinterpreters.md  real-engine-in-workers{,-impl,-roadmap}.md
$ grep -n -i -E 'isojson|orjson' pyre/docs/design/*.md → polars :231, :434; real-engine :325
```

Dispositions:
- polars `:231`/`:434` (per-worker seeded-check pattern) and `:46/:161/:173/:217/:295`
  (numpy per worker; rust-numpy) are consistent with CUJ-2 and design §14.
- The polars C2 harness is not reused (round 1, A20).
- real-engine `:325` is design §7's Pyronova row; FR-14 and O-1.

### Mapping rows (Kind: E existing, N new; every row 无契约)

**Citation rule for this table (round 9, closes the recurring DUP-R5-A3/R9-1):** the
impl-design method requires `file:line` evidence on every EXISTING row, so rows keep their
citations. This table is the one allowed exception to "§14 is the single citation home",
and each **reused-code** row's range must lie within §14's range for the same symbol (a narrower sub-range is fine). **Edit-target** rows (tests, README, CI, Cargo, pyproject) have no §14 symbol; they must match §5 / C7. Checked in each gate round.

| # | Step | Kind | File / function | Design ref |
|---|---|---|---|---|
| M1 | parse args | E | `src/encode.rs:577-647` `dumps` | C2 |
| M2 | numpy option accepted | E→chg | `src/encode.rs:26-33` `UNSUPPORTED_OPTS` + comment | FR-11 |
| M3 | cache + guard into encoder | E→chg | `src/encode.rs:76-82` `Encoder` (+`cache`, `guard`; `pub(crate)` with `out`), `:649-658` | FR-13, C2 |
| M4 | cache in state | E→chg | `src/lib.rs` `ModState` / `state()` (§14) + `types: TypeCache` | C1 |
| M5 | init | E→chg | `src/lib.rs:64-135` `module_exec` → `types.init()` | C1 |
| M6 | teardown | E→chg | `src/lib.rs:137-166` traverse/clear → `types.traverse/clear` | C1 |
| M7 | lookups | N | `src/types.rs` `TypeCache`, per C1 (APIs, names, zero-validity, interface) | C1, §8.1 |
| M8 | dispatch | E→chg | `src/encode.rs:107-157` `serialize` | FR-4, C2 |
| M9 | reentrancy | E→chg | `src/encode.rs` `guarded`, `dict_in_order` (§14); `debug_assert!` sites (C2) | FR-13 |
| M10 | pure date/time + dt64 math | N | `src/datetime.rs` per C3 | C3, §8.2, §8.3 |
| M11 | Python extraction | N | `src/encode.rs` `utcoffset_of`, `datetime`, `date`, `time` per C3 | C3 |
| M12 | errors, default path, `add_note` | E→chg + N | `src/encode.rs` error helpers and `call_default` (§14), split into `invoke_default` (→ `DefaultErr`) + serialize; new `call_default_declined(obj, reason)`; `write_int` extracted; `PyErrSet`/`R<T>` moved from `src/decode.rs:25-27` to `src/lib.rs` | FR-7, FR-12, DV-4b, C2 |
| M13 | numpy walker + declines | N | `src/numpy.rs` (`Outcome`, walker) and pure `src/decline.rs` (`Reason`, `reason_message`, `reason_note`) per C4 | FR-6…9, §8.4 |
| M14 | ints | E | `src/encode.rs:214-246` (`itoa`, `small_copy`) | C4 |
| M15 | floats | E→chg + N | `src/float.rs` (§14): `write_f64` kept, `write_f32`, `write_finite<F>`, `f16_to_f32` (port) | C5, §5 |
| M16 | rollback | E | `src/out.rs` `set_len` (§14) | FR-7 (f) |
| M17 | output / indent | E→chg | `src/out.rs` `Out`; `src/encode.rs` `indent`/`newline_indent` → `pub(crate)` | C4, FR-6 |
| M18 | Python-API oracle | N | `tests/oracle/python_api.py` | §1a |
| M19 | strict-interp shim | N | `tests/interp/strict.py` | C6 |
| M20 | tests E2E-2, 4, 5–13 | N | `test_parity_types.py` (E2E-2, incl. `class D(datetime)`), `test_divergence.py`, `test_isolation.py` (5, 7), `test_concurrency_dt.py` (6), `test_stdlib_oracle.py` (8), `test_no_global_pyobject.py` (9), `test_reentrancy.py` (10), `test_numpy_swap.py` (11), `test_f16.py` (12), `test_declines.py` (13) | §9 |
| M21 | Rust unit tests | N | per C6 (the single list) | C6 |
| M22 | pyre E2E-3 | N | `pyre/tests/test_isojson_numpy_workers.py` | §5, §9 |
| M23 | test change T-1 | E→chg | `tests/test_parity.py:224-229` | FR-11 |
| M24 | README / crate doc / code comments | E→chg | per C7 | C7 |
| M24b | test deps | E→chg | `pyproject.toml:21` `test = ["pytest", "orjson"]` → the §5 single list | §5 |
| M25 | version / changelog / docs | E→chg + N | `Cargo.toml:3`; new `CHANGELOG.md`; commit `docs/design/` (untracked today) with the implementation, since E2E-9 (3) parses it | C7, FR-14 |
| M26 | isojson CI | E→chg | `.github/workflows/ci.yml:26-27, :48` + jobs | §5 |
| M27 | pyre CI | E→chg | `pyre/.github/workflows/ci.yml:88` (integration job only); pyre `CHANGELOG.md` entry | §5, C7 |
| M28 | bench | E→ext | `bench/bench.py` | §11 |

Rows M1–M35 from rounds 1–4 are superseded by this table. Their dispositions are kept in
the audit record below.

**T-items (existing-test edits needing the maintainer):**
- **T-1** (approved 2026-09-24): `tests/test_parity.py:224-229` keeps only its
  `OPT_NON_STR_KEYS` half.
- **T-3:** withdrawn. SkyTrade is out of scope.
- **T-4** (optional, later): point `tests/test_subinterp_312_313.py` at
  `tests/interp/strict.py`, so there is one strict-interpreter shim.
- **T-2:** withdrawn in round 3.

## Step 3 — Audit record

### Pre-audit amendments (found while mapping)
- **A-1** f16 via the `half` crate (revised in round 1: with `std`, see A13). **Superseded in round 2 (R2-A5): `half` rejected.**
- **A-2** reuse the polars C2 harness for the shadow: **withdrawn in round 1** (A20).
- **A-3** cite rust-numpy's global `PY_ARRAY_API` in design §14.
- **A-4** citation fixes (pyre `:83-116`; `test_invalid_opts` `:224-228`).
- **T-1** `tests/test_parity.py:224-229` froze 0.1's "numpy option raises": approved
  2026-09-24. Applied together with the change that enables `OPT_SERIALIZE_NUMPY`.

### Round 1 (fresh adversarial auditor, 2026-09-24): 12 blocking, 28 advisory, 8 duplicates

Rubrics applied: arc `onion-`, `data-truth-`, `economy-portable-rules.md`. Every
blocking item was verified by the writer before amending. Findings accumulate; later rounds
add to this list and never replace it.

| Id | Finding (class) | Disposition |
|---|---|---|
| B1 | 3.12 datetime premise wrong (grounding-gap) | **Confirmed, root cause found.** On 3.12 the outcome depends on import order. If the main interpreter has not imported `datetime`, a strict sub-interpreter loads `_datetime` (capsule present), which is what the writer measured twice. If main imported it first, as Pyronova does, `_datetime` fails the multi-interpreter check and `datetime` falls back to `_pydatetime` (no capsule). The writer's "per-interpreter types on 3.12" was measuring the pure-Python class. Pyronova and SkyTrade require 3.13+, so this is out of scope: design non-goal + E2E-7 now asserts that pure-Python datetimes go to `default`. Lookup moved to `sys.modules["_datetime"]` (FR-2). `PyDateTime_IMPORT` is rejected because it imports and is process-global; the "wrong on 3.12" rationale is withdrawn |
| B2a | `utcoffset()` → `None` contradicts §1a (judgment-call) | Taken per §1a: written naive (DV-4a). §15.5 |
| B2b | DV-4 unmeasured (grounding-gap) | Measured (auditor): `None` → orjson appends `+00:00`; raise → SIGSEGV. DV-4 split into 4a/4b; 4b is a death row |
| B3 | i128 can overflow (method-gap) | `checked_mul` everywhere; days range-checked before `civil_from_days` (FR-9, §8.3) |
| B4 | reentrancy guard incomplete, use-after-free (method-gap) | FR-13: a per-call `guard`; INCREF items/keys in `guarded` and in `dict_in_order`'s no-default branch; E2E-10; the NFR-1 cost is measured |
| B5 | §1a `x.item()` contradicts FR-9/10 (method-gap) | §1a rewritten: `x.item()` only for NaT and int/bool scalars; floats and datetime64 have their own rules |
| B6 | E2E-4(c) vs `datetime_as_string` (method-gap) | Shared `expected_text` re-renders the numpy value by the `datetime` rules with options applied (§1a, E2E-4, E2E-8) |
| B7 | NFR-6 numpy in strict interpreters impossible (method-gap) | NFR-6 is datetime-only on 3.13+; concurrent numpy lives in pyre E2E-3 |
| B8 | numpy 1.26 criterion impossible (grounding-gap) | Dropped; numpy 2.x only (§11) |
| B9 | no CI/deps rows (method-gap) | M29–M31; E2E-3 must fail, not skip; handler returns `.decode()` |
| B10 | commit `4b44d804` not found (grounding-gap) | The auditor used a stale local SkyTrade checkout (844 commits behind), so "not found" was an artifact of the checkout. The writer's premise was stale too: it assumed the commit was unpushed. **Corrected in round 2 (R2-B2):** `4b44d804` was merged to SkyTrade `main` via `8ce46db9` (2026-09-24 09:22) and is live. Cases preserved in `fixtures/` (M32). The local checkout is synced to `79c61085` |
| B11 | abort rc assertion vs platforms (method-gap) | `returncode < 0` on POSIX; Windows is skipped with a reason |
| A1–A3 | citations (zmij `:1764/:1781`, `Out` `:33`, timedelta accessors, capsule name) | fixed in the design |
| A4 | §11 vs §8.1 slow-path cost | §11 rewritten; keys interned |
| A5 | FR-3 path untested; stub numpy raised | missing attr → `Absent`; E2E-11 |
| A6 | FR-2 premise false (judgment-call) | lookup via `_datetime`; pure-Python datetimes → `default` |
| A7 | zero-filled state | zero-validity requirement; `clear()` releases names |
| A8 | borrow through raw pointer | groups return `Copy` pointer structs |
| A9 | unenforced claims | E2E-5 (no `datetime` / `_datetime` import), E2E-3 `default=raise_`; NFRs are a manual release gate |
| A10 | no tripwire for FR-1 | E2E-9 / M27 |
| A11 | contract changes unrecorded | FR-14, §7 rows, M31 |
| A12 | existing-test changes | T-1 in the design (FR-11); `rand_value` untouched, own generator (M20) |
| A13 | `half` F16C claim false without `std` (judgment-call) | `std` enabled (§15.5). **Superseded in round 2 (R2-A5): `half` rejected.** |
| A14 | E2E-8 details | options applied via `expected_text`; µs offsets sampled |
| A15 | DV precision | DV-10 units corrected (ns can't reach 9999); DV-11 message format; the generic `M8` wording fixed |
| A16 | "truncated" vs floor | FR-9 says floored, as numpy |
| A17 | `OMIT_MICROSECONDS` for datetime64 | FR-9 |
| A18 | "everything else identical" false | §1b gains a "not bugs" list; C7 deletes the 0.1 bullets |
| A19 | `app.isolate` path, `test_isolate.py` | E2E-3 runs both paths (declared and reactive). **Harness superseded (R3-A5)** |
| A20 | A-2 misfit (judgment-call) | A-2 withdrawn; in-process shadow (§15.1) |
| A21 | orjson unpinned (judgment-call) | pinned 3.12.0 + non-blocking latest job (§15.5) |
| A22 | `codec.py` not in main (grounding-gap) | **False positive**, same stale checkout. It is on `origin/main`, and the local checkout now has it. The design cites "on SkyTrade `origin/main`" |
| A23 | thin proof-of-absence | pasted searches extended (tests dirs); `jiff-core` dispositioned |
| A24 | FR-4 vs C2 order | aligned (datetime after tuple) |
| A25 | calendar in numpy.rs, no unit tests | calendar in `src/datetime.rs` with exhaustive `cargo test` |
| A27 | (not issued: the round-1 auditor's numbering skips A27) | — |
| A26 | DV-4 message (judgment-call) | isojson message names `utcoffset()` and the real exception (§15.5) |
| A28 | NaT common in SkyTrade | CUJ-1 states it; the shadow measures it |
| A29 | 3.12/3.13 coverage | E2E-5/7 in `test_subinterp_312_313.py`. **Superseded (R2-A12, R3-A14): `tests/interp/strict.py`** |
| D1 | `write_f32` duplicates `write_f64` | one `write_float<F>`. **Superseded (R3-A7): typed writers + `write_finite<F>`** |
| D2 | three date/time formatters | `write_ymd` + `write_hms` compose everything **Superseded (R7-B1): pure `fmt_ymd` / `fmt_hms` / `fmt_offset`** |
| D3 | zero-offset text in two places | datetime64 calls `write_offset(0,0,0,opts)` **Superseded (R7-B1): pure `fmt_ymd` / `fmt_hms` / `fmt_offset`** |
| D4 | two reference derivations | one `expected_text` helper |
| D5 | new pyre harness | built on `test_isolate.py` fixtures. **Superseded (R2-B4, R3-A5): imports from `tests.test_isolate_shared_ext`** |
| D6 | hand-written lookup vs `PyImport_GetModule` (judgment-call) | kept hand-written; reason recorded (no module-lock wait in `dumps`) |
| D7 | re-checking `hastzinfo` | uses `PyDateTime_DATE_GET_TZINFO` |
| D8 | `civil_from_days` vs `jiff-core` (judgment-call) | rejected with reason (§14) |

Judgment-calls taken by the writer under the maintainer's standing principles ("the Python
API is ground truth"; "say what actually happened") are listed in design §15.5, for the
maintainer to reverse.

### Round 2 (fresh adversarial auditor, 2026-09-24): 4 blocking, 21 advisory, 3 duplicates

Local SkyTrade was synced to `origin/main` first. Every blocking item was verified by the
writer (R2-B2 with `git branch -r --contains`, `api/helpers.py:298-345`; R2-A5 with
`half-2.7.1/Cargo.toml:113-117`; DV-12 measured against orjson 3.12.0).

| Id | Finding (class) | Disposition |
|---|---|---|
| R2-B1 | FR-13 guard misses the first datetime load; condition stated three ways (method-gap) | `cache.datetime()` resolved at call start, then `guard`; invariant stated (FR-13, C2, impl CUJ-1, §11 aligned); E2E-10 in a fresh child process |
| R2-B2 | SkyTrade premise stale: `4b44d804` is on main and live; 0.2 would silently drop `+00:00` (grounding-gap) | **Confirmed.** Design §1/CUJ-1/FR-14/§10 rewritten; hazard; the interim pin (M33) was dropped by the maintainer in favour of the release order (§10); B10 disposition corrected. The earlier plan to revert SkyTrade to orjson is obsolete |
| R2-B3 | §10 criterion contradicts the non-goals (method-gap) | Criterion restated: keep UUID/Enum/dataclass branches until Stage 2; SkyTrade's test file unchanged |
| R2-B4 | E2E-3 fixtures skip and are fixed-script (grounding-gap) | Import `_start_server`/`_wait_ready` + `_sigint` without editing; hard `import numpy` (M23). **Superseded (R3-A5): `tests.test_isolate_shared_ext` only** |
| R2-A1 | CUJ-3 misstated orjson's failure | fixed |
| R2-A2 | NaT in ps/fs/as (judgment-call) | `null` per §1a; added to DV-7. **Reversed (round 4): unsupported units are delegated whole (FR-7)** |
| R2-A3 | aware `time` vs §1a (judgment-call) | `time.isoformat()` per §1a; new DV-12 (measured) |
| R2-A4 | datetime types are process-static on 3.13+ | recorded in E2E-6; type separation is proven by numpy in pyre E2E-3 |
| R2-A5 | `half` pulls `zerocopy` + a proc-macro (judgment-call) | `half` rejected; in-crate `f16_to_f32` + exhaustive E2E-12 (reverses round-1 A-1/A13) |
| R2-A6 | §14 mislabels rust-numpy's global | reason corrected: pyo3-ffi only; `import_array` avoided |
| R2-A7 | tripwire too weak | E2E-9 is an explicit allow-list of every `static`/`thread_local!` |
| R2-A8 | pointer-lifetime claim false | invariant: copies are re-fetched after any call that can run Python |
| R2-A9 | interned-name count | 7 interned; the 14 numpy names via `GetAttrString`. **Superseded (R3-A18): module `__dict__`** |
| R2-A10 | `dumps` after `m_clear` | null names → Absent |
| R2-A11 | borrowed module across `__getattr__` | INCREF'd immediately |
| R2-A12 | 3.13 E2E-6 coverage | shared shim covers 3.12–3.14 (M22) |
| R2-A13 | pandas / tzdata test deps | added (§5) |
| R2-A14 | README/doc incomplete | C7/M25: `PASSTHROUGH_DATACLASS` kept in no-op bullet; `:66-68`; `encode.rs:1-7` |
| R2-A15 | M31 kind | N: new CHANGELOG |
| R2-A16 | E2E-3 route syntax, DV inputs | `/s/{seed}`; seeded data excludes DV inputs |
| R2-A17 | `np.longlong` macOS too | FR-8 |
| R2-A18 | shadow has no owner | §15.1: SkyTrade follow-up, deferred, not enforced; Stage 2 waits for it |
| R2-A19 | A27 missing | recorded as not issued |
| R2-A20 | SkyTrade `docs/event-system-design.md:115` stale reason | noted in §15.2 as a SkyTrade doc fix, outside this design |
| R2-A21 | citation nits | fixed (`:111-127`, `src/serialize/numpy/array.rs`) |
| DUP-1 | capsule name/struct redeclared | reuse pyo3-ffi `PyDateTime_CAPI` / `PyDateTime_CAPSULE_NAME` |
| DUP-2 | two pyre helper sets | both imported. **Superseded (R3-A5): its premise was false; one module has everything; T-2 withdrawn** |
| DUP-3 | two sub-interpreter harness files | one shim, 3.12–3.14. **Refined (R3-A14): a new helper module; existing files untouched** |

Round-1 dispositions that round 2 found not landed (B10, B4, D5, A10, A8, A29, A11, A27)
are all covered by the rows above.

**T-2: withdrawn in round 3** (R3-A5). It was: move `_start_server`/`_wait_ready`
(`pyre/tests/test_isolate.py`) and `_start`/`_get`/`_sigint`
(`pyre/tests/test_isolate_shared_ext.py`) into one shared pyre test helper module. It is an
existing-test edit, so it is not done without approval. E2E-3 does not need it.

### Round 3 (fresh adversarial auditor, 2026-09-24): 6 blocking, 21 advisory, 3 duplicates

The auditor also verified the FR-13 invariant as **sound**. It instrumented 0.1.0 with
subclasses overriding `__index__`, `__int__`, `__hash__`, `__eq__`, `__iter__`, `items`,
`__len__` and `__getitem__`: zero Python calls ran during `dumps`.

| Id | Finding (class) | Disposition |
|---|---|---|
| R3-B1 | SkyTrade's test can't stay unchanged (grounding-gap) | M34: shared `_dumps` helper; T-3 (SkyTrade test edit, needs the maintainer) |
| R3-B2 | branches to keep unspecified; ndarray fallback (judgment-call) | exact list in design §1 and M34. **Narrowing corrected in round 4 (R4-B1)** |
| R3-B3 | f16 test wrong (grounding-gap) | `np.arange(65536, uint16).view(float16)`, bitwise compare, Rust bit-exact test |
| R3-B4 | DV-12 had no test (executor-violation) | DV-1…13 in E2E-4, M19, F2/F3 |
| R3-B5 | `UTC_Z` on `time` contradiction | §1a scoped; `time` gets `opts & !(NAIVE_UTC\|UTC_Z)` |
| R3-B6 | `time` offset unvalidated (method-gap) | `obj.utcoffset()` (validated) via one `utcoffset_of` helper for datetime and time; `TypeError` with cause; tests for invalid offsets |
| R3-A1 | stale "inline `_default`" | §7 and CUJ-1 now name `_json_default` / `_dumps` |
| R3-A2 | C6 still cites the fixtures | fixed; D5/A19 marked superseded |
| R3-A3 | `:111-129` | `:111-127` |
| R3-A4 | E2E-12 before E2E-11 | reordered |
| R3-A5 | DUP-2 premise false; `.format()` breaks `{seed}` | one import: `tests.test_isolate_shared_ext`; T-2 withdrawn |
| R3-A6 | f16 re-derived; orjson ships half-rs's fallback | ported with attribution (Apache-2.0 OR MIT) |
| R3-A7 | `write_float<F>` won't compile (E0599) | typed `write_f64`/`write_f32` + generic `write_finite<F>`, as orjson |
| R3-A8 | generic `M8` non-NaT (judgment-call) | `null` per §1a (`x.item()` is `None`); DV-13. **Reversed (R4-A15): delegated/raised, since `datetime_as_string` raises** |
| R3-A9 | DV-12 predicate | "any `time` with `tzinfo` set"; zoneinfo time → no offset |
| R3-A10 | CI doesn't run `cargo test` | M29 |
| R3-A11 | lazy retry undefined | FR-13: retried once per object that misses the fast path, while guarded; E2E-10b |
| R3-A12 | FR-13 prose-only | `debug_assert!(self.guard)` at every Python-running call site; debug CI run |
| R3-A13 | pyre/SkyTrade pins conflict (judgment-call) | pyre runtime stays `>=0.1`, only CI/tests `>=0.2` (built from a git tag until PyPI). The SkyTrade pin is moot: the maintainer dropped it |
| R3-A14 | M22 edits an existing test file | new `tests/interp/strict.py`; existing files untouched |
| R3-A15 | CI extra omitted pandas/tzdata | M29, §5 |
| R3-A16 | SkyTrade docs name "orjson 3.11" | recorded as a SkyTrade doc fix (§15.2) |
| R3-A17 | SkyTrade locks orjson 3.11.7 | recorded in CUJ-1: bytes identical to 3.12.0 on these types (measured by the auditor) |
| R3-A18 | numpy attribute reads (judgment-call) | module `__dict__`, as orjson; no Python runs |
| R3-A19 | citation nits | README `:67-70`; `half` `:83`; `helpers.py:304-338` |
| R3-A20 | C3 mixes pure and Python (judgment-call) | split: pure `datetime.rs`; Python extraction in `encode.rs` |
| R3-A21 | untested claims (zero-valid cache, FR-2 removed `datetime`, re-fetch rule) | Rust test for a zeroed `clear()`; E2E-5 `del sys.modules["datetime"]`; re-fetch stays review-enforced, with `debug_assert!` sites marking where it matters |

**T-3 (needs the maintainer):** SkyTrade `python/tests/api/test_json_response.py:87`
changes from `isojson.dumps(value, default=_json_default)` to `_dumps(value)`. Verdict:
- the test encodes the *call*, and that call changes shape on purpose (options move into
  `_dumps`);
- the cases and the orjson reference stay the same;
- the edit lands in the migration commit; approval is needed **before** the pre-publish test run (release order step 1, R4-A16).

### Round 4 (fresh adversarial auditor, 2026-09-24): 3 blocking, 18 advisory, 8 duplicates

The auditor audited the post-"no pin" version (both docs changed mid-audit) and re-measured
all DV rows on orjson 3.12.0.

| Id | Finding (class) | Disposition |
|---|---|---|
| R4-B1 | the narrowed SkyTrade default changes data or raises on 0-d, `U`/`StringDType`, `np.ma`, `np.longlong`, big-endian, `M8[ps]` (grounding-gap) | **Confirmed by measurement.** Two changes:<br>(1) isojson: every array or scalar it can't write goes to `default` when given, including non-native-endian and unsupported datetime units (FR-7; a "not a bug" divergence for endianness).<br>(2) SkyTrade: a native contiguous copy for non-contiguous/non-native supported arrays, else `.tolist()`; `np.generic → .item()` kept. This also avoids a `default` ping-pong on `M8[ps]`. The inputs become T-3 cases **Part (1)'s "unsupported datetime units → default" superseded by R9-B1 (ps/fs/as are written).** |
| R4-B2 | E2E-5's `del datetime` can't pass on 3.12 | limited to 3.13+ |
| R4-B3 | §8.3 lacks the generic branch | unit check once per array/scalar before elements; ps/fs/as/generic → `default` or raise **Superseded (R9-B1): only generic non-NaT values are declined.** |
| R4-A1 | stale `_default` / Stage-2 mechanism | §15.1: the real difference is `_`-prefixed dataclass fields; CUJ text fixed |
| R4-A2 | stale `write_float` | fixed; D1 marked superseded |
| R4-A3 | `helpers.py:302-339` | `:304-338` |
| R4-A4 | unmarked superseded rows; §15.5 contradictions | R2-A9, R2-B4, A29, D1, R2-A2, R3-A8, R3-B2 marked; §15.5 rewritten as the current state |
| R4-A5 | f16 port breaks MSRV 1.85 and clippy | `as` casts, no `unsafe` (C5, M16) |
| R4-A6 | licensing | only orjson's Apache/MIT files are ported; MPL files are references only; half-rs credited in README (C7) |
| R4-A7 | Rust f16 test has no oracle (judgment-call) | dropped; E2E-12 (numpy oracle) is the proof |
| R4-A8 | E2E-1 can't import the fixture | AST extraction of `CASES` (one source) |
| R4-A9 | zero-validity via an enum discriminant is UB-prone | Absent ⇔ owner pointer null; no discriminant |
| R4-A10 | pure dt64 math in the FFI module | moved to `datetime.rs` with cargo tests |
| R4-A11 | `tests/helpers/` is a banned name | `tests/oracle/`, `tests/interp/` |
| R4-A12 | E2E-9 misses pyo3-ffi globals / imports | banned-symbol list added |
| R4-A13 | §1a lacks 0000–9999 | added; `expected_text` defined only where isojson writes |
| R4-A14 | `PASSTHROUGH_DATETIME` on dt64 | FR-9: no effect |
| R4-A15 | generic `M8` `null` hides a real value (judgment-call) | reversed R3-A8: delegated/raised with an accurate message (DV-13) |
| R4-A16 | release-order mechanics | §10 steps (uv git source, re-lock; T-3 first; pyre CI from the tag); M35 pyre changelog |
| R4-A17 | placement sweep skipped SkyTrade | auditor searched: no contracts; M34 is 无契约 |
| R4-A18 | citation nits | pyo3-ffi 0.29.2 (files identical); `:224-229`; `:83-117`; `typeref.rs:204-240` |
| DUP-R4-1/2 | hand-rolled `GetItemRef` / `GetOptionalAttr` | use pyo3-ffi's |
| DUP-R4-3 | two strict-interp shims (judgment-call) | **T-4** (optional, needs the maintainer): later, point `test_subinterp_312_313.py` at `tests/interp/strict.py` |
| DUP-R4-4 | allow-list vs README table | E2E-9 asserts they match |
| DUP-R4-5 | f16 proven twice | Rust test dropped (R4-A7) |
| DUP-R4-6 | duplicated citations drifted | `helpers.py` range fixed in both docs |
| DUP-R4-7 | `civil_from_days` test listed twice | one entry |
| DUP-R4-8 | pyre's two dumps caches (upstream-inherited) | noted under O-1 |

### Round 5 (fresh adversarial auditor, 2026-09-24): 8 blocking, 13 advisory, 7 duplicates

**Root cause (writer's analysis).** Blocking counts went 12 → 4 → 6 → 3 → 8, so the gate
was not converging. Most round-5 findings (the 7 duplicates, and the partial landings of
R4-A1/A7/A15/A18, DUP-R4-1/2/4/5) had one cause. Each rule was restated on several
surfaces, and each round's patch reached only some of them. Separately, SkyTrade's
`_json_default` details (a different oracle: live 0.1 bytes) were mixed into the isojson
design.

**Response:** a consolidated rewrite.
- Every rule now has one home (e.g. the FR-7 decline rule, §1a oracle, E2E-12 f16 proof,
  C1 lookup APIs, the T-items in step 2), and other sections reference it.
- SkyTrade moved to `skytrade-migration.md`.
- Steps 1–2 of this doc were rewritten to reference design ids instead of restating them.

| Id | Finding (class) | Disposition |
|---|---|---|
| R5-B1 | element errors raise; contradicts FR-7/§1a; live responses would raise (judgment-call) | option (a): FR-7 (f), where the whole array/scalar is declined with `Out` rolled back (`src/out.rs:73`); DV-11 updated; E2E-13 |
| R5-B2 | narrowed default changes f32/f16 text and 0-d | `skytrade-migration.md` §3: `obj[()]` for 0-d, live's f32 trick kept before `.tolist()` **Withdrawn (scope: SkyTrade out).** |
| R5-B3 | declined datetime64 scalars change SkyTrade output (judgment-call) | option (b): the live `np.datetime64` branch kept unchanged **Withdrawn (scope: SkyTrade out).** |
| R5-B4 | release order pushes a `v*` tag early | `skytrade-migration.md` §6: uv source and pyre CI use `rev = <sha>`; `v0.2.0` only at step 3 **Withdrawn (scope: SkyTrade out).** |
| R5-B5 | T-3 scope contradictory; orjson can't be the oracle for new inputs | T-3 rewritten: existing cases keep orjson; new golden-bytes test with live 0.1 as oracle (`skytrade-migration.md` §4) **Withdrawn (scope: SkyTrade out).** |
| R5-B6 | §1a still said generic → `null` | removed in the rewrite; §1a lists generic as "no answer" → FR-7 (e) |
| R5-B7 | E2E-9 fails on day one | `PyImport_ImportModule` allowed by name at `src/lib.rs:71`; allow-list split into constants and shared runtime state; README table `:61-65` |
| R5-B8 | unaligned C-contiguous arrays (upstream-inherited UB) | FR-6: `ptr::read_unaligned`; E2E-13; §12, §14 |
| R5-A1 | f16 "Rust test" text | removed; E2E-12 is the single f16 proof |
| R5-A2 | lookup APIs described three ways | single list in C1; §7 references it |
| R5-A3 | M7 stale citations | the mapping table is rewritten; citations live in the design only |
| R5-A4 | "SkyTrade's `_default`" | removed |
| R5-A5 | dangling fragment in §1 | removed |
| R5-A6 | `M8[us]` already ISO under live | `skytrade-migration.md` §5: only `ns` changes **Withdrawn (scope: SkyTrade out).** |
| R5-A7 | false "measured" claim for the §1 function | `skytrade-migration.md` §5: "not yet measured; the golden test at step 2 is the measurement" **Withdrawn (scope: SkyTrade out).** |
| R5-A8 | longdouble (judgment-call) | recorded (live's behaviour unchanged); design FR-7 (d) covers wide longdouble; itemsize-8 longdouble is written natively as f8, as orjson |
| R5-A9 | E2E-2 exclusions incomplete | E2E-2 excludes the §1b "not bugs" list; FR-7 (e) checks the unit only for non-empty arrays (orjson writes `[]` for empty) |
| R5-A10 | pyre CI not concrete | design §5: exact `git+…@<sha>` install; test extra only after PyPI; M27 E→chg |
| R5-A11 | E2E-1 extraction scope | E2E-1: `CASES`, `_orjson_default`, the class definitions, and the imports **Withdrawn (scope: SkyTrade out).** |
| R5-A12 | T-3 also `:92` | included **Withdrawn (scope: SkyTrade out).** |
| R5-A13 | E2E-3 seeded layouts | E2E-3 payload limited to layouts isojson writes |
| DUP-R5-1…7 | one fact, many homes | resolved by the consolidated rewrite (single homes listed above) |

**Judgment-calls taken:** R5-B1 (a), R5-B3 (b), R5-A8 (record). They are listed in design §15.5.

**Maintainer, 2026-09-24:** "以isojson为主", then "skytrade不要管". The gate covers isojson only. `skytrade-migration.md` and the fixture were parked outside the repo. E2E-1, CUJ-1's SkyTrade case, FR-14's SkyTrade hazard, the §15.1 SkyTrade measurement and the §15.2 SkyTrade codec/docs were removed from the design. Round-1…5 rows about SkyTrade are historical.

### Round 6 (fresh adversarial auditor, 2026-09-24): 1 blocking, 18 advisory, 9 duplicates

Scope: isojson only. The auditor started before SkyTrade was dropped, and its SkyTrade
simulation (appendix) is out of scope and not recorded here.

**Blocking finding**
- **R6-B1: orjson writes malformed JSON for ≥2-D `M8` arrays containing NaT or an
  element it can't write.** It raises nothing, even with `default`; child errors are dropped
  at `orjson/src/serialize/numpy/array.rs:136`. Classes: upstream-inherited +
  grounding-gap.
  - Measured by the writer: `{"x":[[,["2026-01-01T00:00:00"]],"y":1}` for NaT-first, and
    `{"x":[["…"],[],"y":1}` for NaT-second or out-of-range. Invalid JSON with and without
    `default`.
  - Disposition: new DV-14; an E2E-4 row whose part (a) is that `json.loads` fails on
    orjson's bytes; E2E-2 excludes it.

**Judgment-calls**
- **R6-A2:** option (b). E2E-2 tests `class D(datetime)`, and pandas is dropped from the
  test deps.
- **R6-A6:** keep FR-3 (a hit is trusted). E2E-11 runs in fresh child processes with a
  forced miss.
- **R6-A18:** option (b), without changing orjson parity. The message and type stay
  orjson's, and the FR-7 reason is attached with `add_note`, in FR-7.

**Advisories**
- **R6-A1:** E2E-1 references removed (§9 intro, E2E-8, §10, M20).
- **R6-A3:** the `debug_assert!` moves to just before `PyObject_CallOneArg` (C2).
- **R6-A4:** `small_copy` is used only for pieces of 32 bytes or less; the date/time
  writers write fixed-size fields (C3).
- **R6-A5:** `PyModule_Check` added (C1, §8.1, E2E-11 case 3).
- **R6-A7, R6-A8:** E2E-13 now specifies the rollback input, covers more dtypes and
  shapes, and runs in the debug job.
- **R6-A9:** D7 restored: tzinfo NULL/None skips `utcoffset()` (C3).
- **R6-A10:** longdouble recorded in FR-8.
- **R6-A11:** the E2E-9 banned list is the single home and uses identifier-exact
  matching. `PyImport_GetModule` and numpy getattr in `types.rs` were added.
- **R6-A12:** E2E-8 round-trips only years ≥ 0001.
- **R6-A13:** C7 rewrites the `encode.rs:1-7` "nothing cached" sentence.
- **R6-A14:** the CI list is explicit (`:48`; `:50` is `--no-index`).
- **R6-A15:** C6's sentence is scoped to the shim.
- **R6-A16:** E2E-9 (3) checks the README DV list.
- **R6-A17:** citations fixed (README `:203-220`, `response.rs:23-37`).

**Duplicates**
- **DUP-R6-1…9:**
  - §14 is the single citation home; components refer to it (C1–C5, §5, §7).
  - DV-11 and DV-13 reference FR-7.
  - §11 references FR-13.
  - NFR-6 references E2E-6.
  - The empty-array rule lives only in FR-7.
  - The forbidden-symbol list lives only in E2E-9.
  - The impl doc no longer restates C1; M4 is aligned.
- **Carried reinventions:**
  - T-4, the second shim;
  - E2E-2's generator, which is new by design: `rand_value` stays untouched.

**Round-5 not landed:** R5-A2 and R5-A3 fixed by the §14 single citation home; R5-A8 is
now FR-8.

**Round 1–4 decision lost:** D7, restored.

### Round 7 (fresh adversarial auditor, 2026-09-24): 1 blocking, 19 advisory, 10 duplicates

The auditor re-verified every citation (all correct) and measured DV-14's full extent.

**Blocking finding**
- **R7-B1** (the pure `datetime.rs` took a Python-backed `Out`): its formatters now fill
  caller-owned byte arrays, and `encode.rs`/`numpy.rs` copy them into `Out`. E2E-9 bans
  `pyo3_ffi` in `datetime.rs`.

**Judgment calls taken**
- **R7-A3:** generic NaT → `null` (the Python API answers it); generic values are
  declined.
- **R7-A8 — Reversed (maintainer, after round 8): no validation.** Was: the numpy group is validated at load (types, `tp_basicsize`); otherwise
  isojson trusts `sys.modules`, as orjson does (§13).
- **R7-A9:** FR-12 is scoped to new messages; 0.1's messages and `default=None` handling
  are kept and listed in §1b.

**Advisories**
- **R7-A1:** DV-14's predicate is now "where orjson's 1-D writer would raise". The
  "not bugs" list is scoped to 1-D arrays and scalars. E2E-2 refers to DV rows by id.
- **R7-A2:** new DV-15 (invalid datetime offset); DV-1 now covers µs-only offsets.
- **R7-A4:** E2E-9 (2) is now regex bans.
- **R7-A5:** allow-list entries carry their README row; comments are skipped; the README
  wording is "mirrors".
- **R7-A6:** each piece is ≤ 16 bytes.
- **R7-A7:** E2E-11 (1) now calls `dumps` before the swap, and uses a same-layout subclass.
- **R7-A10:** numpy `INDENT_2` is added to FR-6 and E2E-2.
- **R7-A11:** E2E-3 is listed as a parity test.
- **R7-A12:** FR-1 is scoped to `TypeCache`, with the `key_cache` note.
- **R7-A13:** `_datetime` gets `PyModule_Check`; a `GetPointer` failure leaves the group
  Absent; `add_note` is in §7, and its failure is cleared.
- **R7-A14:** §15.5 is completed.
- **R7-A15:** the E2E-1 tombstone is gone; the SkyTrade placement line is fixed; the
  header says rounds 3–7; `pyre/tests/e2e/README.md` is cited.
- **R7-A16:** licensing uses full paths.
- **R7-A17:** the README states numpy ≥ 2, and the test extra has `numpy>=2`.
- **R7-A18:** a noisy grep pattern, noted. The conclusion is unchanged.
- **R7-A19:** naive ⇔ `utcoffset()` is None (§1a); E2E-4/8 cover DV-4a × options.

**Duplicates**
- **DUP-R7-1…6:** all reused-code citations moved to §14. C1, C3, C5, C6, §5 and O-1
  refer to it. Pyronova's call sites are cited only in §7. Impl rows M4/M9/M12/M15–M17
  now point at §14.
- **DUP-R7-7:** E2E-2 refers to DV rows by id.
- **DUP-R7-8:** the pyproject test extra is the single dependency list, which CI reads.
- **DUP-R7-9:** §1a's "no answer" list gives FR-7 (e)/(f)'s *justification*; the rule
  itself lives in FR-7.
- **DUP-R7-10:** impl CUJ-1 step 4 refers to FR-7.

**Round-6 dispositions not landed:** DUP-R6-1, R5-A2/A3 and the missing §15.5 entries are
fixed above.

### Round 8 (fresh adversarial auditor, 2026-09-24): 4 blocking, 22 advisory, 6 duplicates

All citations were re-verified. The auditor re-measured DV-5…14, the fmt buffer maxima,
the E2E-12 oracle (0 mismatches over 65,536 inputs), `tp_basicsize` across numpy 2.0–2.5
and CPython 3.12–3.14, and the CI `tomllib` step (works on all runners).

**Blocking findings and judgment calls**

| Id | Finding | Disposition |
|---|---|---|
| R8-B1 | FR-7 (g) can't be implemented: there is no `np.generic` lookup (judgment-call) | Option (i): (g) dropped. Unrecognized scalars take plain `default` with no note (FR-7, FR-8, E2E-13) |
| R8-B2 | E2E-4 (a) can't hold for DV-4a × `NAIVE_UTC`, where orjson agrees | Part (a) keeps only `NAIVE_UTC\|UTC_Z`; plain `NAIVE_UTC` is in E2E-8 |
| R8-B3 | C7 contradicted the README rewording | C7 lists the rewording |
| R8-B4 | §8.1 diverged from C1 | §8.1 reduced to control flow; checks live only in C1 |
| R8-A1 | `_datetime` used a second mechanism (judgment-call) | Option (a): `__dict__` + `PyDict_GetItemRef`. It runs no Python, and the E2E-9 exception is gone |
| R8-A2 | `tp_basicsize` check needed a formula (judgment-call) | `>= size_of::<PyObject>() + width`; applies only if O-2 picks the layout read; the §13 claim is bounds-only **Reversed (maintainer): no validation.** |
| R8-A16 | a silently Absent group hides the reason (judgment-call) | Kept orjson-like for now; **open for the maintainer as O-4** **Withdrawn (maintainer): O-4 removed.** |
| DUP-R8-3 | the orjson pin is in the isojson extra and pyre CI | Accepted (cross-repo); pyre only at `:88` |
| DUP-R8-4 | DV-2's input also matches DV-1 | Merged into DV-1 (the id is kept as a tombstone) |

**Other advisories**
- **R8-A3:** §13 scoped to scalars; the stub-`ndarray` pointer risk is stated. **Moot (maintainer reversal after round 8): §13 no longer discusses stub arrays.**
- **R8-A4/A5:** E2E-9 bans `PyObject` / `Out` / `crate::out` / `use crate::*` in
  `datetime.rs`, and `TimeZone` was added to the regex.
- **R8-A6:** moot after A1.
- **R8-A7:** §3 DV lists fixed.
- **R8-A8:** remaining citations moved to §14.
- **R8-A9:** `fmt_offset` takes an i64 `total_us`.
- **R8-A10:** an unparseable `dtype.str` → FR-7 (e) with the raw string; `Unit` and
  `Parts` defined.
- **R8-A11:** the unit is checked at the first element; element-less arrays go through
  FR-6.
- **R8-A12:** generic `[NaT, value]` cases added; §1a says `datetime_as_string` decides;
  the numpy deprecation is recorded in §15.3.
- **R8-A13:** DV-15/DV-1 evidence added; a timedelta-subclass case is in E2E-8.
- **R8-A14:** the E2E-9 (4) gc tripwire.
- **R8-A15:** the C1 invariant is marked review-enforced.
- **R8-A17:** pyre CI is Linux-only, and macOS is a manual check; the pin goes only at
  `:88`.
- **R8-A18:** C7 covers the README additions and the limitations heading.
- **R8-A19:** D2/D3 marked superseded; the search regex updated; §9 wording.
- **R8-A20:** NUMPY.md given an owner (O-3).
- **R8-A21:** "tested with numpy ≥ 2".
- **R8-A22:** DV-14 cites (e)/(f).

**Round-7 dispositions not landed:** all covered above (R7-A2 §3, R7-A5 C7, R7-A8/A13
§8.1, R7-A15 §9, DUP-R7-1…6 §14).

**Maintainer, 2026-09-24 (on O-4):** "为什么我要管这个。它换了不是活该吗？另外orjson做不来哦，为什么我门要管？" isojson trusts `sys.modules` exactly as orjson does. **R7-A8's validation (the `tp_basicsize` check) and R8-A2's formula are removed; O-4 is withdrawn.** E2E-11 keeps only the natural cases (re-read, a missing attribute, a non-module). This reverses R7-A8 and R8-A2/A16.

### Round 9 (fresh adversarial auditor, 2026-09-24): 2 blocking, 10 advisory, 2 duplicates

None of the auditor's findings depended on the removed numpy validation; it re-read the
changed sections.

**Blocking findings**

| Id | Finding | Disposition |
|---|---|---|
| R9-B1 | §1a claimed `ps`/`fs`/`as` have no µs form; `datetime_as_string(unit="us")` answers every value (measured by the writer: 5 → `…00:00:00`, −1 → `…23:59:59.999999`, i64 max → `1970-04-17T18:02:52.036854`) (grounding-gap + judgment-call) | Option (a), per §1a: written floored to µs, NaT → `null`. New DV-16: orjson raises even with `default`. FR-7 (e) now covers only generic non-NaT values and unparseable dtypes. This also removes R9-A2's rollback gap |
| R9-B2 | `np.longlong` is `np.int64` on Windows | E2E-13 uses `np.complex128` / `np.longdouble`; FR-8 qualifies the claim |

**Advisories**
- **R9-A1:** C7 adds README `:120-125` and `:131-135`, and the crate doc `src/lib.rs:7-10`.
- **R9-A2:** moot after R9-B1.
- **R9-A3:** DV-15's `str` output varies per run. The row says so, and E2E-4 (a) asserts
  no exception and output ≠ reference.
- **R9-A4:** E2E-9 (3) checks non-tombstone ids, parsed from the design doc.
- **R9-A5:** the test extra includes `pytest`; CI installs `maturin` plus the extra.
- **R9-A6:** `Dt64Err` is defined; `Unit` variants are CamelCase.
- **R9-A7:** the pasted search result was refreshed.
- **R9-A8:** the R7-A8, R8-A2 and R8-A16 rows are marked reversed or withdrawn.
- **R9-A9:** the unparseable-dtype message now lives in FR-7 (e).
- **R9-A10:** `moduleobject.rs:41`; `response.rs:23-39`; header round counts.

**Duplicates**
- **DUP-R9-1:** the impl-table citation rule is written above the mapping table: rows keep
  `file:line` by method mandate and must equal §14. `encode.rs:26-29` in §1 is context
  (what 0.1 does), not a reuse citation.
- **DUP-R9-2:** the same fact as R9-A9.

**Round-8 duplicates not recorded as their own rows:**
- DUP-R8-1 was resolved by R8-B4.
- DUP-R8-2 was resolved by R8-A1.
- DUP-R8-5 was resolved by R8-A7.
- DUP-R8-6 was resolved by R8-A8.

### Round 10 (fresh adversarial auditor, 2026-09-24): 1 blocking, 8 advisory, 3 duplicates

The auditor re-verified all citations and re-measured DV-16, ≥2-D `ps`/`fs`/`as` arrays in
orjson (malformed, DV-14), and the unrecognized scalars. All matched the docs.

**Blocking finding**

| Id | Finding | Disposition |
|---|---|---|
| R10-B1 | The §1a oracle (`datetime_as_string(x, unit="us")`) raises `OverflowError` for multiplied units where v × mult overflows the base unit (e.g. `M8[10ns]` 1e18 = 2286-11-20), which contradicts FR-9 and makes E2E-8 unpassable (grounding-gap + judgment-call) | Option (a): the oracle is `datetime_as_string(x.astype("M8[us]"))`, numpy's own conversion, which floors exactly like FR-9 (measured by the auditor). FR-9 is unchanged **Superseded (R11-B1): exact-integer oracle.** |

**Judgment calls**
- **R10-A1**, option (i): `Dt64Err::GenericValue`, so the pure core decides FR-7 (e).
- **R10-A7**, option (i): `Outcome::Declined(Reason)`, handled once in
  `call_default_declined`; `reason_message` is pure and cargo-tested.

**Advisories**
- **R10-A2:** §8.3 gives the ps/fs/as divisors and the Y/M year check.
- **R10-A3:** E2E-13 uses generic `(2,0)`.
- **R10-A4:** R4-B1 (1) and R4-B3 are marked superseded; T-1 is `:224-229`.
- **R10-A5:** C7's README bullets are un-nested from the crate doc; the ranges are fixed
  to `:123-126` / `:133-135`.
- **R10-A6:** the unparseable-dtype message is enforced by a `reason_message` Rust test
  (C6).
- **R10-A8:** the stale lookup line is fixed.

**Duplicates**
- **DUP-R10-1:** the citation rule is now "lies within §14's range"; `Encoder::int` was
  added to §14.
- **DUP-R10-2:** C6 is the single home for the Rust tests, and E2E-13 refers to it.
- **DUP-R10-3:** the §5 pyre row refers to §10 for platform coverage, and the no-skip rule
  lives only in E2E-3.

### Round 11 (fresh adversarial auditor, 2026-09-24): 1 blocking, 14 advisory, 3 duplicates

The auditor measured the round-10 oracle against FR-9: 13 units × 6 multipliers × 605
values, including i64 extremes and the 0000/10000 boundaries.

**Blocking finding**

| Id | Finding | Disposition |
|---|---|---|
| R11-B1 | numpy's own conversions are not total. `astype`/`datetime_as_string` overflow near i64 MIN and for non-dividing multipliers, and wrap silently for Y/M (`M8[Y]` i64 max → 1969). So §1a contradicted FR-9 (grounding-gap + judgment-call) | Option (a). §1a's truth is the *meaning* numpy's API defines (`np.datetime_data` unit/mult, since 1970), computed with exact integers. numpy's formatter renders only in-range µs values. numpy's conversion functions are recorded as not the oracle. E2E-8 samples each unit's i64 range, including MIN+1 and Y/M max, with ×3/×7 multipliers |

**Judgment calls**
- **R11-A1**, option (i): generic values stay declined. `datetime_data` gives a unit with no
  ratio, so the value has no meaning; `astype` would invent one.
- **R11-A11**, option (ii): `Reason` carries the raw evidence, and `reason_note` puts it in
  the `add_note` text. The message keeps orjson parity.
- **R11-A12**, option (ii): `Reason` and the message functions live in pure
  `src/decline.rs`, which is in E2E-9's purity ban.

**Advisories**
- **R11-A2:** §15.5 has entries for rounds 10–11.
- **R11-A3:** C7's README items are un-nested.
- **R11-A4:** `UnitErr` is replaced by flat `GenericValue` / `UnparseableDtype` reasons.
- **R11-A5:** `invoke_default` is shared; the note is attached only when `default` raised;
  one `debug_assert` site; M12.
- **R11-A6:** M24b maps `pyproject.toml:21`.
- **R11-A7:** M27 cites `:88` only.
- **R11-A8:** §14 covers `26-33`; the citation rule is scoped to reused-code rows.
- **R11-A9:** C7 adds README `:132` and `encode.rs:159-162, :384-385`.
- **R11-A10:** `pub(crate)` in M3 and M17.
- **R11-A13:** M25 commits `docs/design/` with the implementation.
- **R11-A14:** E2E-8 ranges per unit.

**Duplicates**
- **DUP-R11-1:** DV-16 and FR-9 refer to §1a.
- **DUP-R11-2:** M21 points at C6.
- **DUP-R11-3:** `PyErrSet` is reused (§14, M12); the search was re-run.

### Round 12 (fresh adversarial auditor, 2026-09-24): **0 blocking** — gate passed; 13 advisory, 2 duplicates

The auditor implemented §1a's exact-integer oracle and §8.3's arithmetic separately. It ran
13 units × multipliers {1,2,3,7,10,1000} × boundary values (i64 MIN, MIN+1, MAX, 0, ±1,
±2 around 0000-01-01 and 10000-01-01) plus random values: **62,982 cases, 0 mismatches**,
25,138 declined by both. All citations re-verified, and every round-11 disposition landed.

The advisories were applied after the gate passed (they don't change behaviour
decisions):
- **R12-A1:** `DefaultErr { DepthLimit, Raised }`; no note on `DepthLimit`.
- **R12-A2 / DUP-R12-2:** the complete `pub(crate)` list; `Dt64Unit` and `Parts` are
  `Copy` with `pub(crate)` fields.
- **R12-A3:** `expected_text` raises a typed `NoAnswer`; E2E-8 asserts the decline there.
- **R12-A4:** E2E-8 option set scoped.
- **R12-A5:** immortality premise stated in C1, with an E2E-6 tripwire.
- **R12-A6** (judgment-call, option ii): `reason_note` is attached with or without a
  `default` (message and type parity kept).
- **R12-A7:** check order (a)→(d), first match, via a pure `classify` (cargo-tested).
- **R12-A8:** `PyErrSet` claim scoped to `Result` functions.
- **R12-A9** (judgment-call, option ii): `PyErrSet` / `R<T>` moved to `lib.rs`.
- **R12-A10:** R10-B1 and R8-A3 marked.
- **R12-A11:** new messages marked "*new*" in FR-7.
- **R12-A12:** E2E-13 asserts the note content.
- **R12-A13:** `321-347`.
- **DUP-R12-1:** `write_int` extracted.

**Gate result:** zero blocking. The design is finished. Blocking findings by round:
12 → 4 → 6 → 3 → 8 → 1 → 1 → 4 → 2 → 1 → 1 → **0**. The round-5 consolidated rewrite
(single homes) is what made it converge. Next per method: `impl-roadmap`.

## Step 4 — Build plans (impl-build P0, one per milestone)

### M0 (isojson#1) — pure core

- **Ports and doubles:** none. M0 is pure functions (`datetime.rs`, `decline.rs`, the
  f16 widening); nothing in it touches IO or Python, so there is no adapter to mock and
  P3 ("real adapters") is empty. The compiler boundary is the E2E-9 (2) purity ban, run
  as `tests/test_no_global_pyobject.py` (its purity part lands in M0; M1 adds the rest).
- **Oracle (the test double of M0):** an exact-integer model of §1a written separately
  from §8.3, in `src/datetime.rs`'s test module: total µs as i128 by unit table,
  `floor` to days, date by a year/month walk (not Hinnant's formula), range by comparing
  µs against 0000-01-01 and 10000-01-01. Same model the round-12 auditor used.
- **Test data:** boundary sets per unit (i64 MIN, MIN+1, MAX, 0, ±1, ±2 around the
  0000-01-01 and 10000-01-01 crossings for each unit × mult) plus seeded splitmix64
  values; no files.
- **Case matrix:** C6's list, verbatim —
  exhaustive `civil_from_days` over [−719528, 2932896] against a day-by-day walk;
  `dt64_to_parts` at DV-8…11/16 boundaries; the property test (13 units ×
  {1,2,3,7,10,1000} × boundaries + random) against the model; `parse_dt64` valid /
  multiplied / unparseable; `classify` in orjson's order; `reason_message` and
  `reason_note` for every `Reason`. Plus the `fmt_*` formatters (they are the §8.2 text).
  `f16_to_f32` has no Rust test by design (R4-A7); E2E-12 is its proof in M2.
- **E2E:** the existing pytest suite, unchanged, proves the C2 refactors
  (`PyErrSet`/`R<T>` move, `write_int`) change nothing.
- **TDD order:** signatures (P1, `todo!`-free canned bodies) → tests red → real bodies.
- **Interface resolution (found at P1):** design C4 gives
  `classify(flags, nd, kind, itemsize) -> Option<Reason>`, but `Reason`'s (a)–(d)
  variants carry evidence `classify` does not receive (`shape`, and `dtype.str`, which
  needs a Python read). Resolved as `classify(..) -> Result<Elem, Check>`: `Check` names
  the rule (a)–(d), `Elem` is the accepted dtype (the dtype table's single home), and
  `numpy.rs` builds the `Reason` with the evidence it reads. Design C4 amended.

### M1 (isojson#2) — type cache, guard, native datetime

- **Ports and doubles:** CPython itself is the adapter; there is no in-memory
  double for an interpreter. The reusable test homes are `tests/interp/strict.py`
  (strict own-GIL interpreters, 3.12–3.14) and `tests/oracle/python_api.py`
  (`expected_text`, `adjusted`, `NoAnswer`). Crash-class cases (E2E-4 death rows,
  E2E-6, E2E-10) run in child processes under `PYTHONMALLOC=debug`.
- **Mutation check:** with `guarded()` / `dict_in_order` forced unguarded, E2E-10
  fails 7/7 (use-after-free under the debug allocator); restored.
- **Found while building:**
  - DV-17 (new §1b row): orjson drops the leading zero of a `time`'s five-digit
    microsecond (90,000 values). Found by E2E-2's random documents.
  - DV-4a: every option combination *with* `NAIVE_UTC` agrees with orjson;
    E2E-4's text corrected (round 8 had it wrong for `NAIVE_UTC|UTC_Z`).
    orjson spells the invented offset `+00:00` on 3.12/3.13 and `Z` under
    `UTC_Z` on 3.14; E2E-4 (a) accepts either.
  - E2E-6 is 3.14-only: CPython 3.13.15's own `_datetime` crashes under
    concurrent strict interpreters with isojson not involved (design §1,
    E2E-6). Maintainer: "我们都升级到3.14 not a big deal".
  - pyo3-ffi's 3.12 `PyDict_GetItemRef` shim lives at `pyo3_ffi::compat`.
- **Moved to later milestones (not dropped):** E2E-5's `option=NUMPY` call, E2E-10's
  numpy variant and E2E-9 (4)'s numpy check → M2 (numpy option still raises in M1);
  E2E-9 (3) (README DV list) → M4 with the README rewrite. E2E-9 (4) checks the
  datetime types now.
- **README/crate doc:** the statements M1 made false were corrected now (shared-state
  row, per-interpreter state paragraph, process-global pointers, "borrows and doesn't
  keep"); the full C7 rewrite stays in M4.
