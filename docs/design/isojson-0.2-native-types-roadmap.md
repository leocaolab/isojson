# Roadmap — isojson 0.2: native datetime and numpy

> Built from the **finished** design, `isojson-0.2-native-types.md`, and its impl map,
> `isojson-0.2-native-types-impl.md`, after the impl-design gate passed on 2026-09-24
> (round 12: 0 blocking). It reflects the landed design:
> - SkyTrade is out of scope.
> - The numpy validation was removed by the maintainer.
> - FR-7 (g) was dropped in round 8.
> - `ps`/`fs`/`as` are written (DV-16).
> - The oracle is the exact-integer §1a.
>
> Re-run this roadmap if the design changes.

Tracker: GitHub milestone **"isojson 0.2"** in `leocaolab/isojson`. M3 is tracked in
`leocaolab/pyronova` and linked from here. Issue numbers are in the table at the end.

## Sequencing

Risk goes first. M0 is pure Rust with no Python runtime: the calendar and datetime64
arithmetic, the decline classification and the f16/f32 formatting. These are the logic
where the audit found the most bugs (DV-8…11, DV-16, R10-B1, R11-B1), and cargo tests can
prove them exhaustively before any FFI exists.

The FFI milestones follow in dependency order: datetime (M1), then numpy (M2), which
reuses M1's cache and guard. Then the cross-repo multi-worker proof (M3), then the
release (M4).

```
M0 pure core ──► M1 cache + guard + datetime ──► M2 numpy ──► M3 pyre per-worker E2E ──► M4 release
```

## M0 — Pure core (no Python)
- **Scope:**
  - C3 pure half: `src/datetime.rs`, i.e. `fmt_ymd`, `fmt_hms`, `fmt_offset` (§8.2),
    `civil_from_days`, `Unit`, `Dt64Unit`, `Parts`, `parse_dt64`, `dt64_to_parts` with
    `Dt64Err` (FR-9 arithmetic, §8.3).
  - `src/decline.rs`: `Reason`, `classify` (FR-7 order), `reason_message`, `reason_note`.
  - C5: `write_finite<F>`, `write_f32`, the `f16_to_f32` port (MSRV-adapted).
  - C2 refactors with no behaviour change: `write_int` extracted; `PyErrSet`/`R<T>` moved
    to `lib.rs`.
  - CI: `cargo test --lib` and E2E-9 (2)'s purity bans for `datetime.rs` / `decline.rs`.
- **Dependencies:** none.
- **Verification:**
  - C6 Rust tests: exhaustive `civil_from_days` over [−719528, 2932896]; `dt64_to_parts`
    at the DV-8…11/16 boundaries; `parse_dt64`; `classify`, `reason_message` and
    `reason_note` for every `Reason`.
  - The round-12 cross-check as a Rust property test: `dt64_to_parts` against an
    exact-integer model over 13 units × multipliers {1,2,3,7,10,1000} × boundary values.
  - The existing Python suite stays green, which shows the refactors change nothing.
- **Independently shippable:** yes. It is internal only (no user-visible change), and the
  0.1 behaviour is intact.

## M1 — Type cache, reentrancy guard, native datetime
- **Scope:**
  - C1: `TypeCache` with the datetime group, 7 interned names and zero-validity; FR-1,
    FR-2.
  - C2: `Encoder.cache` / `guard`, the `invoke_default` / `DefaultErr` split, the
    `debug_assert!` sites, and the `pub(crate)` list; FR-4, FR-13.
  - C3 Python half: `utcoffset_of` (D7 fast path, naive ⇔ `utcoffset()` is `None`),
    `Encoder::datetime` / `date` / `time`; FR-5, DV-1, DV-3, DV-4a, DV-4b, DV-12, DV-15.
  - C7 partial: the `encode.rs` module doc and `guarded` comments.
  - Test helpers: `tests/oracle/python_api.py` (`expected_text`, `NoAnswer`) and
    `tests/interp/strict.py`.
  - The test deps single list (M24b) and the CI install step.
- **Dependencies:** M0.
- **Verification:**
  - E2E-2 (datetime part, including the `class D(datetime)` → `default` case).
  - E2E-4: the datetime DV rows.
  - E2E-5 (no imports), E2E-6 (concurrency with the immortality tripwire), E2E-7 (3.12
    pure-Python datetime; removed in M2 with 3.12/3.13 support), E2E-8 (datetime part), E2E-9 (1)–(4), E2E-10 / 10b
    (reentrancy in fresh processes).
  - The debug-build pytest job.
- **Independently shippable:** yes. `datetime` / `date` / `time` serialize natively with
  the four options. It could go out as 0.2.0a1, with numpy still raising as in 0.1.

## M2 — numpy
- **Scope:**
  - C1 numpy group (FR-3; `__dict__` reads; re-read on replacement).
  - C4: `src/numpy.rs`, i.e. `PyArrayInterface`, `Outcome`, `serialize_array` /
    `serialize_scalar`, the unaligned-safe walk with `INDENT_2`, and the FR-7 declines
    with rollback through `call_default_declined`; FR-6, FR-7, FR-8, FR-9.
  - FR-11: `OPT_SERIALIZE_NUMPY` accepted, plus the approved test change T-1
    (`tests/test_parity.py:224-229`).
  - DV-5…11, DV-13, DV-14, DV-16.
- **Dependencies:** M0 (arithmetic, decline core, f16/f32), M1 (cache, guard,
  default-path split).
- **Verification:**
  - E2E-2 (numpy part, compact and `INDENT_2`).
  - E2E-4: the numpy DV rows, including DV-14 malformed-JSON proofs and the death rows on
    POSIX.
  - E2E-8 (datetime64 part: per-unit i64 ranges, MIN+1, Y/M max, ×3/×7).
  - E2E-11 (numpy swap), E2E-12 (all 65,536 f16), E2E-13 (every FR-7 case, rollback
    bytes, note content, unaligned arrays in the debug job).
- **Independently shippable:** yes. `OPT_SERIALIZE_NUMPY` works in a single interpreter.

## M3 — Per-worker numpy in Pyronova (cross-repo)
- **Scope:** `pyre/tests/test_isojson_numpy_workers.py` (E2E-3, CUJ-2). It uses the
  harness from `tests.test_isolate_shared_ext` (imported, not edited) and runs both the
  declared (`app.isolate("numpy")`) and the reactive clone paths. Pyre CI's integration
  job (`ci.yml:88`) installs isojson from the M2 commit SHA and `orjson==3.12.0`. The
  pyre CHANGELOG gets an entry (FR-14).
- **Dependencies:** M2 (the isojson commit that contains numpy).
- **Verification:** E2E-3 green, not skipped, in pyre CI (Linux), and run by hand on
  macOS.
- **Independently shippable:** yes. It proves CUJ-2 (each worker's own numpy copy, no
  cross-worker type matches, clean SIGINT) without changing Pyronova's runtime
  dependency (`isojson>=0.1` stays).

## M4 — Release 0.2.0
- **Scope:**
  - C7: README feature table, "Differences from orjson" mirrored from §1b (E2E-9 (3)),
    limitations heading, `:67-70`, `:123-126`, `:132-135`, the shared-state table
    rewording, Status, Third-party (half-rs), the numpy ≥ 2 note, and the crate doc
    `lib.rs:7-10`.
  - `Cargo.toml` 0.2.0 and a new `CHANGELOG.md` (FR-14, §1b).
  - CI's non-blocking latest-orjson job.
  - `bench/bench.py` payloads, with NFR-1…5 measured on macOS arm64 and Linux x86_64 and
    recorded in the release notes.
  - Merge `docs/design/` (M25), then tag `v0.2.0`, which publishes to PyPI via
    `release.yml`.
- **Dependencies:** M2 and M3.
- **Verification:** §10 success criteria:
  - FR-1…14 conformance check against the design;
  - NFR-1…6 thresholds;
  - E2E-2…13 green on the CI matrix;
  - `cargo test --lib` green.
- **Independently shippable:** it is the release.

## Deferred / not milestones (so no dead work-items)
- **Stage 2** (UUID / Enum / dataclass, design §15.1): a later release. No issue now.
- **Strict mode** (§15.2): not planned.
- **O-1** (Pyronova passing options): a Pyronova decision after 0.2. Not tracked here.
- **O-2** (numpy scalar read strategy): decided inside M2 by benchmark.
- **O-3** (`NUMPY.md`): owned by the PyO3 fork work.
- **T-4** (merge the two strict-interpreter shims): optional, needs the maintainer. Not
  tracked.
- **Removed during the gate, never tracked:**
  - SkyTrade migration;
  - numpy type validation (O-4);
  - FR-7 (g);
  - the `half` crate;
  - the polars C2 reuse.

## Tracker

| Milestone | Issue | Status |
|---|---|---|
| (milestone) | GitHub milestone #1 "isojson 0.2" (leocaolab/isojson) | created |
| M0 | leocaolab/isojson#1 | done (#6) |
| M1 | leocaolab/isojson#2 | done (#7) |
| M2 | leocaolab/isojson#3 | done (#8) |
| M3 | leocaolab/pyronova#9 (tracked by leocaolab/isojson#5) | done (pyronova#10) |
| M4 | leocaolab/isojson#4 | open |

Commands run (2026-09-24):
- `gh api repos/leocaolab/isojson/milestones -f title="isojson 0.2" …`, which created
  milestone 1;
- `gh issue create --repo leocaolab/isojson --milestone "isojson 0.2" …` ×5 (#1–#5);
- `gh issue create --repo leocaolab/pyronova …` (#9).

Reconciliation: isojson had no issues or milestones before, so there was nothing to align
or close. The one pre-existing open pyronova issue (#8, polars UDFs) is unrelated and was
left as is. Nothing the gate removed was ever tracked.
