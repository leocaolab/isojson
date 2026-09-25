"""E2E-9: no new process-global Python state, enforced over the Rust source.

(1) every `static` / `thread_local!` item is on a reviewed allow-list, and each
    piece of shared runtime state is covered by a row of the README's
    "shared state" table;
(2) the banned-symbol list — the single home of those bans (design E2E-9);
(4) FR-1's traverse: cached types are visited by the module's `m_traverse`.
Comments are skipped. Part (3) (the README's DV list) lands with the README
rewrite (M4).
"""

import datetime
import gc
import pathlib
import re

import pytest

import isojson

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def code_only(text: str) -> str:
    """The source with `//` and `/* */` comments removed."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def sources():
    return {p.relative_to(SRC).as_posix(): code_only(p.read_text(encoding="utf-8")) for p in sorted(SRC.rglob("*.rs"))}


# ---- (1) static / thread_local! allow-lists ----------------------------------

STATIC = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?static\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:", re.M)

# Plain constants: tables and CPython module definitions, never mutated
# after load. Exempt from the README.
CONSTANTS = {
    "encode.rs": {"HEX", "ESCAPE"},
    "lib.rs": {"METHODS", "SLOTS", "MODULE_DEF"},
}

# Shared runtime state, each with the README table row that covers it.
SHARED = {
    ("strfast.rs", "ENABLED"): "the `str` fast-path switch",
    ("decode.rs", "SCRATCH"): "per-thread scratch buffers and size hints",
    ("out.rs", "HINT"): "per-thread scratch buffers and size hints",
}


def test_every_static_is_reviewed():
    found = {(f, name) for f, code in sources().items() for name in STATIC.findall(code)}
    allowed = {(f, n) for f, names in CONSTANTS.items() for n in names} | set(SHARED)
    assert found == allowed, {"unreviewed": found - allowed, "gone": allowed - found}


@pytest.mark.parametrize(("item", "row"), SHARED.items(), ids=lambda v: str(v))
def test_shared_state_has_its_readme_row(item, row):
    assert f"| {row} |" in README, f"{item}: README row {row!r} missing"


def test_static_regex_sees_both_forms():
    code = "static A: u8 = 0;\nthread_local! {\n    static B: Cell<u8> = Cell::new(0);\n}\npub(crate) static mut C: u8 = 0;"
    assert STATIC.findall(code) == ["A", "B", "C"]


# ---- (2) banned symbols ------------------------------------------------------

# pattern -> files it applies to (None = every file)
BANS = [
    (r"\bPyDateTime_IMPORT\b", None),
    (r"\bPyDateTimeAPI\b", None),
    (r"\bPyDateTime_TimeZone_UTC\b", None),
    # all read pyo3-ffi's process-global PyDateTimeAPI()
    (r"\bPy(Date|DateTime|Time|Delta|TZInfo|TimeZone)_(Check\w*|From\w*)\b", None),
    (r"\bPyCapsule_Import\b", None),
    (r"\bPyImport_(Import\w*|GetModule|AddModule\w*)\b", None),
    # module attributes are read from __dict__ only (C1)
    (r"\bPyObject_Get(Optional)?Attr\w*\b", {"types.rs"}),
    # purity: no Python, no output buffer
    (r"\bpyo3_ffi\b", {"datetime.rs", "decline.rs"}),
    (r"\bPyObject\b", {"datetime.rs", "decline.rs"}),
    (r"\bOut\b", {"datetime.rs", "decline.rs"}),
    (r"crate::out\b", {"datetime.rs", "decline.rs"}),
    (r"use crate::\*", {"datetime.rs", "decline.rs"}),
]

# The one allowed site: module_exec imports `json` for JSONDecodeError.
ALLOWED = {("lib.rs", 'PyImport_ImportModule(c"json".as_ptr())')}


def test_no_banned_symbols():
    hits = []
    for f, code in sources().items():
        for line in code.splitlines():
            for pattern, files in BANS:
                if files is not None and f not in files:
                    continue
                if re.search(pattern, line) and not any(
                    f == af and site in line for af, site in ALLOWED
                ):
                    hits.append((f, pattern, line.strip()))
    assert not hits, hits


def test_the_allowed_import_site_exists():
    code = sources()["lib.rs"]
    assert all(site in code for _, site in ALLOWED)


@pytest.mark.parametrize(
    "line",
    [
        "PyDateTime_IMPORT();",
        "let api = PyDateTimeAPI();",
        "if PyDateTime_Check(o) != 0 {",
        "PyDelta_FromDSU(0, 0, 0)",
        "PyTZInfo_CheckExact(o)",
        "PyCapsule_Import(name, 0)",
        "PyImport_ImportModule(c\"numpy\".as_ptr())",
        "PyImport_GetModule(name)",
    ],
)
def test_bans_catch_each_symbol(line):
    assert any(re.search(p, line) for p, files in BANS if files is None)


def test_bans_leave_the_lookup_apis_alone():
    for line in ["PyImport_GetModuleDict()", "PyDateTime_DATE_GET_TZINFO(o)", "PyDateTime_CAPSULE_NAME"]:
        assert not any(re.search(p, line) for p, files in BANS if files is None), line


def test_comments_are_skipped():
    assert code_only("let x = 1; // writes into Out\n/* PyObject */") == "let x = 1; \n"


# ---- (4) FR-1's traverse -----------------------------------------------------

def test_cached_datetime_types_are_traversed():
    isojson.dumps(datetime.datetime(2026, 1, 1))
    referents = gc.get_referents(isojson.isojson)
    for t in (datetime.datetime, datetime.date, datetime.time):
        assert t in referents


def test_cached_numpy_types_are_traversed():
    np = __import__("numpy")
    isojson.dumps(np.float64(1.0), option=isojson.OPT_SERIALIZE_NUMPY)
    referents = gc.get_referents(isojson.isojson)
    assert np.float64 in referents and np.ndarray in referents and np in referents
