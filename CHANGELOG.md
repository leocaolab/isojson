# Changelog

## 0.2.0

Native `datetime` / `date` / `time` and numpy, still safe in own-GIL
sub-interpreters: every type isojson recognizes is looked up in the calling
interpreter's own `sys.modules` and cached in that interpreter's module state.
isojson never imports `datetime` or `numpy`.

### Breaking changes

- **CPython 3.14 is required** (`requires-python >=3.14`; wheels for 3.14
  only). 3.12 and 3.13 are dropped: their own `_datetime` isn't usable from
  concurrent strict sub-interpreters (CPython 3.13.15 crashes with no isojson
  code involved).
- **`datetime`, `date` and `time` no longer reach `default=`.** They are
  written natively, as `isoformat()` writes them. If you relied on your own
  `default` for datetimes and pass no options, check the output:
  - a naive `datetime` is written **without** an offset
    (`"2026-09-24T12:00:00"`). Pass `OPT_NAIVE_UTC` to write it as UTC
    (`+00:00`, or `Z` with `OPT_UTC_Z`);
  - `OPT_PASSTHROUGH_DATETIME` sends all three types to `default` as before.
- **The datetime options now take effect:** `OPT_NAIVE_UTC`, `OPT_UTC_Z`,
  `OPT_OMIT_MICROSECONDS`, `OPT_PASSTHROUGH_DATETIME` (they were accepted and
  ignored in 0.1).

### Added

- `datetime` / `date` / `time`, byte-identical to orjson except where orjson
  is wrong (below).
- `OPT_SERIALIZE_NUMPY`: C-contiguous native-endian arrays of `bool`,
  `float16/32/64`, `int8…64`, `uint8…64` and `datetime64`, any number of
  dimensions, `OPT_INDENT_2` included; and the matching numpy scalars. Tested
  with numpy ≥ 2.
- A numpy object isojson can't write goes whole to `default=`; without one,
  `dumps` raises orjson's message with a note carrying the evidence
  (`dtype.str`, flags, shape, value). See README, "Declined numpy objects".

### Where isojson differs from orjson 3.12.0

orjson crashes or writes wrong data on these inputs; isojson writes what
Python's own API says, or declines. Every row has a regression test.

- DV-1: offsets with seconds or microseconds are written exactly, not
  rounded to the minute.
- DV-3: pytz datetimes after arithmetic use `dt.utcoffset()`.
- DV-4a: a tzinfo returning `None` from `utcoffset()` is naive.
- DV-4b: a raising `utcoffset()` is a `TypeError` with the cause (orjson
  crashes).
- DV-5, DV-6, DV-7: `datetime64` NaT is `null` in every unit.
- DV-8: multiplied units (`M8[10ms]`) are written (orjson crashes).
- DV-9: `M8[M]` before 1970 uses floor division (orjson crashes).
- DV-10: `datetime64` up to 9999-12-31 is written.
- DV-11: `datetime64` values that overflow are declined, not wrapped.
- DV-12: a `time` with `tzinfo` is `t.isoformat()`.
- DV-13: a generic-unit `datetime64` value is declined with an accurate
  message.
- DV-14: ≥2-D `datetime64` arrays never produce malformed JSON.
- DV-15: an invalid `utcoffset()` result raises, as `dt.utcoffset()` does.
- DV-16: `datetime64` in `ps` / `fs` / `as` is written, floored to µs.
- DV-17: a `time` with a five-digit microsecond keeps its leading zero.

### Performance

- Integer numpy rows are written through one reserve per row.
- `utcoffset()` is called the way CPython's `datetime.utcoffset()` calls
  it, without its format-string call: aware datetimes are faster than
  orjson's.
- Floats are formatted in place in the output. 0.1 formatted each into a
  stack buffer and copied it out, which x86_64 can't store-forward: floats in
  [0, 1), normals and integral floats took up to twice orjson's time on
  Linux; they are now at parity.
- Plain-JSON `dumps` is 2–7% slower than 0.1 (a few ns per call and per
  element), and a list of objects bound for `default` with
  `OPT_SERIALIZE_NUMPY` up to 13% on Linux (leocaolab/isojson#10).

## 0.1.0

First release: orjson-compatible `dumps` / `loads` that load in own-GIL
sub-interpreters with no override.
