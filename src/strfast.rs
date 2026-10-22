//! Reads a `str`'s characters without a function call or an allocation.
//!
//! CPython does not expose this in its public API, so this module mirrors the
//! object header layout of CPython 3.14 (GIL builds): a compact string stores
//! its characters right after its header (`PyASCIIObject` for ASCII,
//! `PyCompactUnicodeObject` otherwise), and a compact non-ASCII string may also
//! keep a cached UTF-8 pointer.
//!
//! The layout is private, so it is not trusted blindly: `self_check` runs at
//! module import, compares this fast path against the public API on probe
//! strings, and enables it only if every probe agrees. If a CPython version
//! ever changes the layout, isojson falls back to the public API — slower,
//! never wrong. Free-threaded builds (different layout) never enable it.

use core::sync::atomic::{AtomicBool, Ordering};
use pyo3_ffi::*;

/// Process-wide: the header layout is a property of the running libpython,
/// the same for every interpreter. Plain data, not a Python object.
static ENABLED: AtomicBool = AtomicBool::new(false);

#[repr(C)]
struct PyASCIIObject {
    ob_base: PyObject,
    length: Py_ssize_t,
    hash: Py_hash_t,
    state: u32,
}

#[repr(C)]
struct PyCompactUnicodeObject {
    base: PyASCIIObject,
    utf8_length: Py_ssize_t,
    utf8: *const u8,
}

// state bits (GIL builds, 3.14): interned:2, kind:3, compact:1, ascii:1
const KIND_SHIFT: u32 = 2;
const KIND_MASK: u32 = 0b111;
const COMPACT: u32 = 1 << 5;
const ASCII: u32 = 1 << 6;

/// The characters of a `str`, borrowed in whichever form CPython holds them.
pub(crate) enum StrText<'a> {
    /// The text is available as UTF-8: ASCII, or a cached copy.
    Utf8(&'a [u8]),
    Latin1(&'a [u8]),
    Ucs2(&'a [u16]),
    Ucs4(&'a [u32]),
}

impl StrText<'_> {
    /// Returns whether the text has a code point UTF-8 can't encode (a surrogate).
    pub(crate) fn has_surrogate(&self) -> bool {
        let surrogate = |c: u32| (0xD800..=0xDFFF).contains(&c);
        match self {
            StrText::Utf8(_) | StrText::Latin1(_) => false,
            StrText::Ucs2(s) => s.iter().any(|&c| surrogate(c.into())),
            StrText::Ucs4(s) => s.iter().any(|&c| surrogate(c)),
        }
    }
}

unsafe fn text_of_kind<'a>(kind: u32, data: *const u8, n: usize) -> StrText<'a> {
    if kind == PyUnicode_1BYTE_KIND {
        StrText::Latin1(core::slice::from_raw_parts(data, n))
    } else if kind == PyUnicode_2BYTE_KIND {
        StrText::Ucs2(core::slice::from_raw_parts(data.cast(), n))
    } else {
        StrText::Ucs4(core::slice::from_raw_parts(data.cast(), n))
    }
}

/// Returns the text of an exact or subclass `str` without allocating, so no
/// UTF-8 copy is left on the caller's object.
#[inline(always)]
pub(crate) unsafe fn text<'a>(op: *mut PyObject) -> StrText<'a> {
    #[cfg(not(Py_GIL_DISABLED))]
    if ENABLED.load(Ordering::Relaxed) {
        if let Some(t) = layout_text(op) {
            return t;
        }
    }
    api_text(op)
}

/// Reads a compact string's text from its header, preferring a cached UTF-8
/// copy, without the `ENABLED` gate; `None` for a non-compact one (an instance
/// of a `str` subclass).
#[cfg(not(Py_GIL_DISABLED))]
#[inline(always)]
unsafe fn layout_text<'a>(op: *mut PyObject) -> Option<StrText<'a>> {
    let st = (*op.cast::<PyASCIIObject>()).state;
    if st & (COMPACT | ASCII) == COMPACT {
        let c = op.cast::<PyCompactUnicodeObject>();
        if !(*c).utf8.is_null() {
            let utf8 = core::slice::from_raw_parts((*c).utf8, (*c).utf8_length as usize);
            return Some(StrText::Utf8(utf8));
        }
    }
    layout_chars(op)
}

/// Reads a compact string's characters as stored, ignoring any cached UTF-8.
#[cfg(not(Py_GIL_DISABLED))]
#[inline(always)]
unsafe fn layout_chars<'a>(op: *mut PyObject) -> Option<StrText<'a>> {
    let a = op.cast::<PyASCIIObject>();
    let st = (*a).state;
    let n = (*a).length as usize;
    if st & (COMPACT | ASCII) == (COMPACT | ASCII) {
        let data = op.cast::<u8>().add(core::mem::size_of::<PyASCIIObject>());
        return Some(StrText::Utf8(core::slice::from_raw_parts(data, n)));
    }
    if st & COMPACT == 0 {
        return None;
    }
    let data = op
        .cast::<u8>()
        .add(core::mem::size_of::<PyCompactUnicodeObject>());
    Some(text_of_kind((st >> KIND_SHIFT) & KIND_MASK, data, n))
}

unsafe fn api_text<'a>(op: *mut PyObject) -> StrText<'a> {
    let n = PyUnicode_GET_LENGTH(op) as usize;
    text_of_kind(PyUnicode_KIND(op), PyUnicode_DATA(op).cast(), n)
}

/// Returns the UTF-8 of a `str`, from the header when it is ASCII or cached,
/// else from `PyUnicode_AsUTF8AndSize`, which allocates a copy and keeps it on
/// the object. `None` means it is not encodable (lone surrogates); the error is
/// cleared.
pub(crate) unsafe fn as_utf8<'a>(op: *mut PyObject) -> Option<&'a [u8]> {
    if let StrText::Utf8(b) = text(op) {
        return Some(b);
    }
    let mut len: Py_ssize_t = 0;
    let p = PyUnicode_AsUTF8AndSize(op, &mut len);
    if p.is_null() {
        PyErr_Clear();
        return None;
    }
    Some(core::slice::from_raw_parts(p.cast(), len as usize))
}

/// Returns whether the header read of `o` agrees with the public API on its
/// kind and characters, and on its UTF-8 once that is cached. A probe may be a
/// shared singleton (`"é"`) whose UTF-8 an earlier check already cached.
#[cfg(not(Py_GIL_DISABLED))]
unsafe fn probe_agrees(o: *mut PyObject, expect: &[u8]) -> bool {
    let n = PyUnicode_GET_LENGTH(o) as usize;
    let data = PyUnicode_DATA(o).cast::<u8>().cast_const();
    let chars = match layout_chars(o) {
        Some(StrText::Utf8(b)) => expect.is_ascii() && b.as_ptr() == data && b == expect,
        Some(StrText::Latin1(s)) => {
            PyUnicode_KIND(o) == PyUnicode_1BYTE_KIND && s.as_ptr() == data && s.len() == n
        }
        Some(StrText::Ucs2(s)) => {
            PyUnicode_KIND(o) == PyUnicode_2BYTE_KIND && s.as_ptr().cast() == data && s.len() == n
        }
        Some(StrText::Ucs4(s)) => {
            PyUnicode_KIND(o) == PyUnicode_4BYTE_KIND && s.as_ptr().cast() == data && s.len() == n
        }
        None => false,
    };
    let mut len: Py_ssize_t = 0;
    let api = PyUnicode_AsUTF8AndSize(o, &mut len); // fills the cache
    let utf8 = match layout_text(o) {
        Some(StrText::Utf8(b)) => {
            !api.is_null()
                && b.as_ptr() == api.cast::<u8>()
                && b.len() == len as usize
                && b == expect
        }
        _ => false,
    };
    chars && utf8
}

/// Enables the fast path iff it agrees with the public API on every probe.
/// Returns -1 only if creating a probe string fails (Python error set).
pub(crate) unsafe fn self_check() -> core::ffi::c_int {
    #[cfg(Py_GIL_DISABLED)]
    {
        return 0;
    }
    #[cfg(not(Py_GIL_DISABLED))]
    {
        let probes: [&core::ffi::CStr; 7] = [
            c"",
            c"a",
            c"hello, isojson: a longer ascii probe string",
            c"\xc3\xa9",                 // é (1-byte kind, not ASCII)
            c"\xe6\x97\xa5\xe6\x9c\xac", // 日本 (2-byte kind)
            c"\xf0\x9f\x94\xa5x",        // 🔥x (4-byte kind)
            c"mixed \xc3\xa9 text",
        ];
        // Never toggles ENABLED on while checking: other interpreters may be
        // serializing on other threads right now.
        let mut ok = true;
        for p in probes {
            let o = PyUnicode_FromString(p.as_ptr());
            if o.is_null() {
                return -1;
            }
            ok &= probe_agrees(o, p.to_bytes());
            Py_DECREF(o);
        }
        // Deterministic for a given libpython, so every interpreter's check
        // stores the same value.
        ENABLED.store(ok, Ordering::Relaxed);
        0
    }
}

/// Returns whether the fast path is active in this process; only tests read it.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}
