//! isojson — JSON for CPython that is safe under per-interpreter GIL
//! sub-interpreters (PEP 684).
//!
//! The one rule this crate is built around: **no Python object is ever held
//! in process-global storage.** Everything that must outlive a call lives in
//! the module's per-interpreter state (`ModState`), which CPython allocates
//! once per interpreter and tears down with it. The only process-global
//! pointers we touch are CPython's static builtin types and the immortal
//! singletons (`None`/`True`/`False`), which every interpreter shares by
//! design and never refcounts.
//!
//! Multi-phase init (PEP 489) + `Py_MOD_PER_INTERPRETER_GIL_SUPPORTED` means
//! the module loads in strict own-GIL sub-interpreters with no override.

#[cfg(Py_3_15)]
compile_error!("isojson does not support Python 3.15 yet (PyModExport init is not wired)");

mod decode;
mod encode;
mod float;
mod number;
mod out;
mod strfast;
mod swar;

use core::ffi::{c_int, c_void};
use core::ptr;
use pyo3_ffi::*;

/// Per-interpreter module state. One instance per interpreter that imports
/// isojson; never shared.
#[repr(C)]
pub(crate) struct ModState {
    /// `isojson.JSONDecodeError`, a subclass of *this interpreter's*
    /// `json.JSONDecodeError`.
    pub(crate) decode_error: *mut PyObject,
    /// Recently seen dict keys for `loads` — this interpreter's objects only.
    pub(crate) key_cache: *mut decode::KeyCache,
}

#[inline]
pub(crate) unsafe fn state(module: *mut PyObject) -> *mut ModState {
    PyModule_GetState(module).cast::<ModState>()
}

// Option bits. Values are identical to orjson's so callers can switch
// imports without touching flags.
pub(crate) const OPT_INDENT_2: u32 = 1;
pub(crate) const OPT_NAIVE_UTC: u32 = 2;
pub(crate) const OPT_NON_STR_KEYS: u32 = 4;
pub(crate) const OPT_OMIT_MICROSECONDS: u32 = 8;
pub(crate) const OPT_SERIALIZE_NUMPY: u32 = 16;
pub(crate) const OPT_SORT_KEYS: u32 = 32;
pub(crate) const OPT_STRICT_INTEGER: u32 = 64;
pub(crate) const OPT_UTC_Z: u32 = 128;
pub(crate) const OPT_PASSTHROUGH_SUBCLASS: u32 = 256;
pub(crate) const OPT_PASSTHROUGH_DATETIME: u32 = 512;
pub(crate) const OPT_APPEND_NEWLINE: u32 = 1024;
pub(crate) const OPT_PASSTHROUGH_DATACLASS: u32 = 2048;

unsafe fn add_int(m: *mut PyObject, name: &core::ffi::CStr, v: u32) -> c_int {
    PyModule_AddIntConstant(m, name.as_ptr(), v as core::ffi::c_long)
}

unsafe extern "C" fn module_exec(m: *mut PyObject) -> c_int {
    let st = state(m);
    (*st).decode_error = ptr::null_mut();
    (*st).key_cache = Box::into_raw(decode::KeyCache::new());

    // Subclass this interpreter's json.JSONDecodeError, so `except
    // json.JSONDecodeError` / `except ValueError` both catch ours.
    let json_mod = PyImport_ImportModule(c"json".as_ptr());
    if json_mod.is_null() {
        return -1;
    }
    let base = PyObject_GetAttrString(json_mod, c"JSONDecodeError".as_ptr());
    Py_DECREF(json_mod);
    if base.is_null() {
        return -1;
    }
    let exc = PyErr_NewException(c"isojson.JSONDecodeError".as_ptr(), base, ptr::null_mut());
    Py_DECREF(base);
    if exc.is_null() {
        return -1;
    }
    (*st).decode_error = exc; // state owns this reference
    if PyModule_AddObjectRef(m, c"JSONDecodeError".as_ptr(), exc) < 0 {
        return -1;
    }
    // orjson.JSONEncodeError *is* TypeError; mirror that exactly.
    if PyModule_AddObjectRef(m, c"JSONEncodeError".as_ptr(), PyExc_TypeError) < 0 {
        return -1;
    }

    for (name, v) in [
        (c"OPT_APPEND_NEWLINE", OPT_APPEND_NEWLINE),
        (c"OPT_INDENT_2", OPT_INDENT_2),
        (c"OPT_NAIVE_UTC", OPT_NAIVE_UTC),
        (c"OPT_NON_STR_KEYS", OPT_NON_STR_KEYS),
        (c"OPT_OMIT_MICROSECONDS", OPT_OMIT_MICROSECONDS),
        (c"OPT_PASSTHROUGH_DATACLASS", OPT_PASSTHROUGH_DATACLASS),
        (c"OPT_PASSTHROUGH_DATETIME", OPT_PASSTHROUGH_DATETIME),
        (c"OPT_PASSTHROUGH_SUBCLASS", OPT_PASSTHROUGH_SUBCLASS),
        (c"OPT_SERIALIZE_DATACLASS", 0), // orjson legacy no-op
        (c"OPT_SERIALIZE_NUMPY", OPT_SERIALIZE_NUMPY),
        (c"OPT_SERIALIZE_UUID", 0), // orjson legacy no-op
        (c"OPT_SORT_KEYS", OPT_SORT_KEYS),
        (c"OPT_STRICT_INTEGER", OPT_STRICT_INTEGER),
        (c"OPT_UTC_Z", OPT_UTC_Z),
    ] {
        if add_int(m, name, v) < 0 {
            return -1;
        }
    }
    if strfast::self_check() < 0 {
        return -1;
    }
    // private: lets the test suite assert the str fast path is active
    let fast = if strfast::enabled() {
        Py_True()
    } else {
        Py_False()
    };
    if PyModule_AddObjectRef(m, c"_str_fastpath".as_ptr(), fast) < 0 {
        return -1;
    }
    if PyModule_AddStringConstant(
        m,
        c"__version__".as_ptr(),
        concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast(),
    ) < 0
    {
        return -1;
    }
    0
}

unsafe extern "C" fn module_traverse(
    m: *mut PyObject,
    visit: visitproc,
    arg: *mut c_void,
) -> c_int {
    let st = state(m);
    if !st.is_null() && !(*st).decode_error.is_null() {
        let r = visit((*st).decode_error, arg);
        if r != 0 {
            return r;
        }
    }
    0
}

unsafe extern "C" fn module_clear(m: *mut PyObject) -> c_int {
    let st = state(m);
    if st.is_null() {
        return 0;
    }
    if !(*st).decode_error.is_null() {
        let e = (*st).decode_error;
        (*st).decode_error = ptr::null_mut();
        Py_DECREF(e);
    }
    if !(*st).key_cache.is_null() {
        (*(*st).key_cache).clear();
    }
    0
}

unsafe extern "C" fn module_free(m: *mut c_void) {
    let m: *mut PyObject = m.cast();
    module_clear(m);
    let st = state(m);
    if !st.is_null() && !(*st).key_cache.is_null() {
        drop(Box::from_raw((*st).key_cache));
        (*st).key_cache = ptr::null_mut();
    }
}

static mut METHODS: [PyMethodDef; 3] = [
    PyMethodDef {
        ml_name: c"dumps".as_ptr(),
        ml_meth: PyMethodDefPointer {
            PyCFunctionFastWithKeywords: encode::dumps,
        },
        ml_flags: METH_FASTCALL | METH_KEYWORDS,
        ml_doc:
            c"dumps(obj, /, default=None, option=None) -> bytes\n--\n\nSerialize obj to JSON bytes."
                .as_ptr(),
    },
    PyMethodDef {
        ml_name: c"loads".as_ptr(),
        ml_meth: PyMethodDefPointer {
            PyCFunction: decode::loads,
        },
        ml_flags: METH_O,
        ml_doc: c"loads(obj, /)\n--\n\nDeserialize JSON from bytes, bytearray, memoryview, or str."
            .as_ptr(),
    },
    PyMethodDef::zeroed(),
];

const SLOTS_LEN: usize = 3 + cfg!(Py_3_13) as usize;

static mut SLOTS: [PyModuleDef_Slot; SLOTS_LEN] = [
    PyModuleDef_Slot {
        slot: Py_mod_exec,
        value: module_exec as *mut c_void,
    },
    PyModuleDef_Slot {
        slot: Py_mod_multiple_interpreters,
        value: Py_MOD_PER_INTERPRETER_GIL_SUPPORTED,
    },
    // Free-threaded builds: we iterate dicts/lists with borrowed refs, which
    // is only safe under a GIL. Declare that honestly instead of claiming
    // GIL_NOT_USED.
    #[cfg(Py_3_13)]
    PyModuleDef_Slot {
        slot: Py_mod_gil,
        value: Py_MOD_GIL_USED,
    },
    PyModuleDef_Slot {
        slot: 0,
        value: ptr::null_mut(),
    },
];

static mut MODULE_DEF: PyModuleDef = PyModuleDef {
    m_base: PyModuleDef_HEAD_INIT,
    m_name: c"isojson.isojson".as_ptr(),
    m_doc: c"Fast JSON that is safe in per-interpreter-GIL sub-interpreters.".as_ptr(),
    m_size: core::mem::size_of::<ModState>() as Py_ssize_t,
    m_methods: (&raw mut METHODS).cast(),
    m_slots: (&raw mut SLOTS).cast(),
    m_traverse: Some(module_traverse),
    m_clear: Some(module_clear),
    m_free: Some(module_free),
};

/// Module entry point, called by CPython's import machinery.
///
/// # Safety
/// Must only be called by the CPython interpreter during import.
#[allow(non_snake_case)]
#[no_mangle]
pub unsafe extern "C" fn PyInit_isojson() -> *mut PyObject {
    PyModuleDef_Init(&raw mut MODULE_DEF)
}
