# Conformance — isojson 0.2 against its design

Checks `isojson-0.2-native-types.md` (the design) §10 success criteria on the
`m4-release` branch, 2026-09-25. Each FR names the code that implements it
and the test that proves it. Symbols rather than line numbers, so this stays
true as the files move.

## Functional requirements

| FR | Implementation | Proof | |
|---|---|---|---|
| FR-1 no Python object in process-global storage; `TypeCache` in `ModState`, traversed, cleared | `src/types.rs` `TypeCache` (`traverse`, `clear`); `src/lib.rs` `ModState.types`, `module_traverse`, `module_clear` | E2E-9 (1) every `static` reviewed, (2) banned symbols, (4) cached datetime and numpy types in `gc.get_referents(module)`; `types::tests::zeroed_cache_clear_is_a_no_op` | ✅ |
| FR-2 types looked up in the current interpreter's `sys.modules`, never imported | `TypeCache::datetime` (`_datetime.__dict__["datetime_CAPI"]`), `TypeCache::numpy` (`numpy.__dict__`), both via `PyImport_GetModuleDict` + `PyDict_GetItemRef` | E2E-5 (nothing imported; `del sys.modules["datetime"]` still native); E2E-9 (2) | ✅ |
| FR-3 hit trusted; miss re-reads a replaced numpy | `TypeCache::numpy` | E2E-11 (trusted hit, re-read, stub, non-module) | ✅ |
| FR-4 dispatch order | `Encoder::serialize` (builtins, datetime after `tuple`, subclass flags, numpy before `default`) | E2E-2 | ✅ |
| FR-5 datetime/date/time text, `PASSTHROUGH_DATETIME`, DV-4b | `Encoder::datetime` / `date` / `time`, `utcoffset_of`, `write_datetime`; `src/datetime.rs` formatters | E2E-2, E2E-4 (DV-1, 3, 4a, 4b, 12, 15, 17), E2E-8 | ✅ |
| FR-6 ndarray via `__array_struct__`, unaligned reads, `[]`, `INDENT_2` | `src/numpy.rs` `serialize_array`, `walk`, `rd` (`read_unaligned`) | E2E-2 (dtypes × shapes × indent), E2E-13 unaligned (also in the debug-build job) | ✅ |
| FR-7 the decline rule | `src/decline.rs` `classify`, `Reason`, `reason_message`, `reason_note`; `numpy.rs` `Outcome::Declined` with rollback; `Encoder::call_default_declined` | `cargo test` decline tests (order, table, every message and note); E2E-13 (a)–(f), rollback compact and indented, notes, depth limit | ✅ |
| FR-8 numpy scalars; unrecognized → plain `default` | `numpy.rs` `scalar_elem`, `serialize_scalar` (layout read, O-2) | E2E-2 scalars, E2E-13 unrecognized scalars | ✅ |
| FR-9 datetime64 arithmetic | `src/datetime.rs` `parse_dt64`, `dt64_to_parts` | `cargo test` exhaustive `civil_from_days`, DV boundaries, exact-integer model (~57k cases); E2E-4 DV-5…11, 13, 14, 16; E2E-8 every unit × multiplier over the i64 range | ✅ |
| FR-10 f32 / f16 | `src/float.rs` `write_f32`, `write_finite`, `f16_to_f32` | E2E-12 all 65,536 float16; E2E-8 floats by value; E2E-2 parity | ✅ |
| FR-11 `OPT_SERIALIZE_NUMPY` accepted, `OPT_NON_STR_KEYS` raises, T-1 | `encode.rs` `UNSUPPORTED_OPTS` | `test_parity.py::test_invalid_opts` (T-1 applied, approved) | ✅ |
| FR-12 new messages only where defined; others orjson's | FR-7 texts in `decline.rs`; DV-4b in `utcoffset_raised` | `cargo test` messages; E2E-2 compares every other error's type and message with orjson; E2E-13 raising-`default` message | ✅ |
| FR-13 reentrancy guard | `Encoder.guard` (set in `dumps` after resolving the datetime group), `guarded`, `dict_in_order`, `debug_assert!(self.guard)` at `invoke_default`, `utcoffset_of`, `interface`, `dtype_str` | E2E-10 / 10b under `PYTHONMALLOC=debug`; mutation check (guard forced off → 7/7 crash); the debug-build CI job runs the asserts | ✅ |
| FR-14 public-behaviour change recorded; 0.2.0 | `CHANGELOG.md` (breaking changes, `OPT_NAIVE_UTC`, 3.14-only); `Cargo.toml` 0.2.0; Pyronova's CHANGELOG note | — | ✅ |

## Other §10 criteria

- **E2E-2…13 green on the CI matrix:** isojson CI (Linux x86_64 and arm64,
  macOS, Windows × CPython 3.14, plus the debug-build job); E2E-4's death rows
  run on POSIX. E2E-7 was removed with 3.12 support (below).
- **E2E-3 green in Pyronova CI (Linux), run by hand on macOS:**
  leocaolab/pyronova#10.
- **`cargo test --lib` green:** isojson CI lint job.
- **C7 done:** README (feature table, differences mirrored from §1b and
  checked by E2E-9 (3), limitations, testing, status, half-rs), crate doc,
  code comments, CHANGELOG.
- **NFR-1…6:** measured with `python bench/bench.py nfr --baseline <0.1>`
  on macOS arm64 and Linux x86_64; numbers go in the release notes.
  NFR-6 is E2E-6. **NFR-1 is not met and was accepted:** `dumps` is 3–5%
  slower than 0.1 on 0.1's payloads (macOS arm64, quiet box, same-toolchain
  0.1 baseline); the FR-13 guard itself measures zero. Maintainer,
  2026-09-25: accepted for 0.2.0, to optimize later — leocaolab/isojson#10.
  NFR-2…5 pass on macOS (NFR-2 0.76×, NFR-3 0.68–1.09×, NFR-4 0.84–0.87×,
  NFR-5 1.08×).

## Where the build changed the design (each recorded in the design)

- `classify` returns `Result<Elem, Check>`: `Reason`'s evidence needs a
  Python read `classify` doesn't have (M0).
- `nd < 0` is treated as 0-d (M0).
- DV-17 added: orjson drops a `time`'s leading microsecond zero (M1).
- DV-4a: every option combination with `NAIVE_UTC` agrees with orjson; orjson's
  invented offset is platform-dependent garbage (M1).
- E2E-6 runs on 3.14 only; then CPython 3.12 and 3.13 were dropped
  (maintainer), removing E2E-7 (M1, M2).
- O-2 decided: scalar layout read (M2).
- `DtTypes` gains `delta`; `utcoffset_of` does CPython's `call_tzinfo_method`
  inline (M4, NFR-2).
