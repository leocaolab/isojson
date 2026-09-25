"""E2E-9: no new process-global Python state, enforced over the Rust source.

Part (2), the banned-symbol list, is the single home of those bans (design
E2E-9). Only the purity bans exist so far: `src/datetime.rs` and
`src/decline.rs` must not reach Python or the output buffer, so their logic
stays cargo-testable without an interpreter. Comments are skipped.
"""

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

PURE_FILES = ["datetime.rs", "decline.rs"]

PURITY_BANS = [
    r"\bpyo3_ffi\b",
    r"\bPyObject\b",
    r"\bOut\b",
    r"crate::out\b",
    r"use crate::\*",
]


def code_only(text: str) -> str:
    """The source with `//` and `/* */` comments removed."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


@pytest.mark.parametrize("name", PURE_FILES)
def test_pure_modules_stay_pure(name):
    code = code_only((SRC / name).read_text())
    assert "fn " in code, f"{name}: nothing left to check"
    hits = [
        (pattern, line)
        for line in code.splitlines()
        for pattern in PURITY_BANS
        if re.search(pattern, line)
    ]
    assert not hits, f"{name} must stay pure: {hits}"


@pytest.mark.parametrize(
    "line",
    [
        "use pyo3_ffi::*;",
        "fn f(o: *mut PyObject) {}",
        "fn f(out: &mut Out) {}",
        "use crate::out::Out;",
        "use crate::*;",
    ],
)
def test_purity_bans_catch_each_symbol(line):
    assert any(re.search(p, code_only(line)) for p in PURITY_BANS)


def test_comments_are_skipped():
    assert code_only("let x = 1; // writes into Out\n/* PyObject */") == "let x = 1; \n"
