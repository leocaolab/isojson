//! Per-interpreter type cache (design C1, FR-1…3).
//!
//! Types isojson recognizes beyond the builtins are looked up in the calling
//! interpreter's `sys.modules`, never imported, and cached in that
//! interpreter's module state. Module attributes are read from the module's
//! `__dict__`, so no Python code runs during a lookup.
//!
//! Zero-validity: CPython zero-fills module state, and an all-null cache is
//! valid and empty. A group is Absent iff its owner pointer is null; null
//! names (before `init`, after `clear`) mean every group stays Absent.

use core::ffi::{c_int, c_void, CStr};
use core::ptr;
use pyo3_ffi::*;

use crate::{PyErrSet, R};

/// The datetime group: `_datetime`'s C types, which are static and shared by
/// every interpreter on 3.13+ and immortal, so reference counting them from
/// several own-GIL interpreters is race-free (E2E-6 has the tripwire).
#[derive(Clone, Copy)]
pub(crate) struct DtTypes {
    pub(crate) datetime: *mut PyTypeObject,
    pub(crate) date: *mut PyTypeObject,
    pub(crate) time: *mut PyTypeObject,
}

/// The interned names, created in `init`.
#[repr(C)]
pub(crate) struct Names {
    pub(crate) datetime_mod: *mut PyObject,
    pub(crate) numpy: *mut PyObject,
    pub(crate) datetime_capi: *mut PyObject,
    pub(crate) utcoffset: *mut PyObject,
    pub(crate) array_struct: *mut PyObject,
    pub(crate) dtype: *mut PyObject,
    pub(crate) str: *mut PyObject,
}

impl Names {
    fn slots(&mut self) -> [(&mut *mut PyObject, &'static CStr); 7] {
        [
            (&mut self.datetime_mod, c"_datetime"),
            (&mut self.numpy, c"numpy"),
            (&mut self.datetime_capi, c"datetime_CAPI"),
            (&mut self.utcoffset, c"utcoffset"),
            (&mut self.array_struct, c"__array_struct__"),
            (&mut self.dtype, c"dtype"),
            (&mut self.str, c"str"),
        ]
    }
}

#[repr(C)]
pub(crate) struct TypeCache {
    /// Owner of the datetime group: `_datetime.datetime_CAPI`.
    dt_capsule: *mut PyObject,
    dt: DtTypes,
    pub(crate) names: Names,
}

/// `d[key]` as a strong reference; `Ok(None)` if missing.
unsafe fn get_item(d: *mut PyObject, key: *mut PyObject) -> R<Option<*mut PyObject>> {
    let mut v = ptr::null_mut();
    match pyo3_ffi::compat::PyDict_GetItemRef(d, key, &mut v) {
        1 => Ok(Some(v)),
        0 => Ok(None),
        _ => Err(PyErrSet),
    }
}

/// Release a strong reference held in `slot`, leaving it null.
unsafe fn release<T>(slot: &mut *mut T) {
    let p = core::mem::replace(slot, ptr::null_mut());
    if !p.is_null() {
        Py_DECREF(p.cast());
    }
}

impl TypeCache {
    /// Create the interned names. Called from `module_exec`.
    pub(crate) unsafe fn init(&mut self) -> c_int {
        for (slot, name) in self.names.slots() {
            let s = PyUnicode_InternFromString(name.as_ptr());
            if s.is_null() {
                return -1;
            }
            *slot = s;
        }
        0
    }

    /// The datetime group, loaded on first use (design §8.1). Absent when
    /// `_datetime` is not in `sys.modules`, is not a module, or has no valid
    /// `datetime_CAPI` capsule; retried on the next call.
    pub(crate) unsafe fn datetime(&mut self) -> R<Option<DtTypes>> {
        if !self.dt_capsule.is_null() {
            return Ok(Some(self.dt));
        }
        if self.names.datetime_mod.is_null() {
            return Ok(None);
        }
        let Some(module) = get_item(PyImport_GetModuleDict(), self.names.datetime_mod)? else {
            return Ok(None);
        };
        let cap = if PyModule_Check(module) != 0 {
            get_item(PyModule_GetDict(module), self.names.datetime_capi)
        } else {
            Ok(None)
        };
        Py_DECREF(module);
        let Some(cap) = cap? else {
            return Ok(None);
        };
        let api = PyCapsule_GetPointer(cap, PyDateTime_CAPSULE_NAME.as_ptr());
        if api.is_null() {
            // not the datetime C-API capsule: the group stays Absent
            PyErr_Clear();
            Py_DECREF(cap);
            return Ok(None);
        }
        let api = api.cast::<PyDateTime_CAPI>();
        let dt = DtTypes {
            datetime: (*api).DateTimeType,
            date: (*api).DateType,
            time: (*api).TimeType,
        };
        for t in [dt.datetime, dt.date, dt.time] {
            Py_INCREF(t.cast());
        }
        self.dt = dt;
        self.dt_capsule = cap;
        Ok(Some(dt))
    }

    pub(crate) unsafe fn traverse(&self, visit: visitproc, arg: *mut c_void) -> c_int {
        let held: [*mut PyObject; 11] = [
            self.dt_capsule,
            self.dt.datetime.cast(),
            self.dt.date.cast(),
            self.dt.time.cast(),
            self.names.datetime_mod,
            self.names.numpy,
            self.names.datetime_capi,
            self.names.utcoffset,
            self.names.array_struct,
            self.names.dtype,
            self.names.str,
        ];
        for o in held {
            if !o.is_null() {
                let r = visit(o, arg);
                if r != 0 {
                    return r;
                }
            }
        }
        0
    }

    /// Release the groups and the names. A zeroed cache is left as it is.
    pub(crate) unsafe fn clear(&mut self) {
        release(&mut self.dt_capsule);
        release(&mut self.dt.datetime);
        release(&mut self.dt.date);
        release(&mut self.dt.time);
        for (slot, _) in self.names.slots() {
            release(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_cache_clear_is_a_no_op() {
        // CPython zero-fills module state; `clear` on it must not touch Python
        // (no interpreter exists in this test, so any FFI call would crash).
        let mut c: TypeCache = unsafe { core::mem::zeroed() };
        unsafe {
            c.clear();
            c.clear();
            assert!(matches!(c.datetime(), Ok(None)));
        }
        assert!(c.dt_capsule.is_null() && c.names.datetime_mod.is_null());
    }
}
