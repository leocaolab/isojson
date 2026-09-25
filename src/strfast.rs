//! Reading a `str`'s UTF-8 without a function call.
//!
//! CPython does not expose this in its public API, so this module mirrors the
//! object header layout of CPython 3.14 (GIL builds): a compact ASCII
//! string stores its characters right after `PyASCIIObject`, and a compact
//! non-ASCII string keeps a cached UTF-8 pointer in `PyCompactUnicodeObject`.
//!
//! The layout is private, so it is not trusted blindly: `self_check` runs at
//! module import, compares this fast path against `PyUnicode_AsUTF8AndSize`
//! on probe strings, and enables it only if every probe agrees. If a CPython
//! version ever changes the layout, isojson falls back to the public API —
//! slower, never wrong. Free-threaded builds (different layout) never enable
//! it.

use core::sync::atomic::{AtomicBool, Ordering};
use pyo3_ffi::*;

/// Process-wide: the header layout is a property of the running libpython,
/// the same for every interpreter. Plain data, not a Python object.
static ENABLED: AtomicBool = AtomicBool::new(false);

#[repr(C)]
struct AsciiObject {
    ob_base: PyObject,
    length: Py_ssize_t,
    hash: Py_hash_t,
    state: u32,
}

#[repr(C)]
struct CompactUnicodeObject {
    base: AsciiObject,
    utf8_length: Py_ssize_t,
    utf8: *const u8,
}

// state bits (GIL builds, 3.14): interned:2, kind:3, compact:1, ascii:1
const COMPACT: u32 = 1 << 5;
const ASCII: u32 = 1 << 6;

/// UTF-8 of an exact or subclass `str` if it is available without a call:
/// compact ASCII, or compact non-ASCII whose UTF-8 is already cached.
#[inline(always)]
pub(crate) unsafe fn utf8<'a>(op: *mut PyObject) -> Option<&'a [u8]> {
    #[cfg(Py_GIL_DISABLED)]
    {
        let _ = op;
        return None;
    }
    #[cfg(not(Py_GIL_DISABLED))]
    {
        if !ENABLED.load(Ordering::Relaxed) {
            return None;
        }
        layout_utf8(op)
    }
}

/// The layout read itself, without the ENABLED gate (used by `self_check`).
#[cfg(not(Py_GIL_DISABLED))]
#[inline(always)]
unsafe fn layout_utf8<'a>(op: *mut PyObject) -> Option<&'a [u8]> {
    {
        let a = op.cast::<AsciiObject>();
        let st = (*a).state;
        if st & (COMPACT | ASCII) == (COMPACT | ASCII) {
            let data = op.cast::<u8>().add(core::mem::size_of::<AsciiObject>());
            return Some(core::slice::from_raw_parts(data, (*a).length as usize));
        }
        if st & COMPACT != 0 {
            let c = op.cast::<CompactUnicodeObject>();
            if !(*c).utf8.is_null() {
                return Some(core::slice::from_raw_parts(
                    (*c).utf8,
                    (*c).utf8_length as usize,
                ));
            }
        }
        None
    }
}

/// UTF-8 of a `str`: fast path, else `PyUnicode_AsUTF8AndSize` (which also
/// caches it on the object). `None` = not encodable (lone surrogates); the
/// Python error is cleared.
#[inline(always)]
pub(crate) unsafe fn as_utf8<'a>(op: *mut PyObject) -> Option<&'a [u8]> {
    if let Some(b) = utf8(op) {
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

/// Enable the fast path iff it agrees with the public API on every probe.
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
            let mut len: Py_ssize_t = 0;
            let api = PyUnicode_AsUTF8AndSize(o, &mut len); // also fills the cache
            let fast = layout_utf8(o);
            let expect_ascii = p.to_bytes().is_ascii();
            let is_ascii_bits =
                (*o.cast::<AsciiObject>()).state & (COMPACT | ASCII) == (COMPACT | ASCII);
            ok &= match fast {
                Some(b) => {
                    !api.is_null()
                        && b.as_ptr() == api.cast::<u8>()
                        && b.len() == len as usize
                        && b == p.to_bytes()
                        && is_ascii_bits == expect_ascii
                }
                None => false,
            };
            Py_DECREF(o);
        }
        // Deterministic for a given libpython, so every interpreter's check
        // stores the same value.
        ENABLED.store(ok, Ordering::Relaxed);
        0
    }
}

/// For tests: whether the fast path is active in this process.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}
