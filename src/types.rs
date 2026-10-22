//! Caches, per interpreter, the datetime and numpy types isojson recognizes.
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
use core::iter;
use core::ptr;
use pyo3_ffi::*;

use crate::decline::ElementType;
use crate::{PendingPyErr, PyResult};

/// The datetime group: `_datetime`'s C types, which are static, shared by
/// every interpreter and immortal, so reference counting them from
/// several own-GIL interpreters is race-free (E2E-6 has the tripwire).
#[derive(Clone, Copy)]
pub(crate) struct DtTypes {
    pub(crate) datetime: *mut PyTypeObject,
    pub(crate) date: *mut PyTypeObject,
    pub(crate) time: *mut PyTypeObject,
    /// `timedelta`: `utcoffset()` results are checked against it (FR-5).
    pub(crate) delta: *mut PyTypeObject,
}

/// The numpy group: `ndarray` and the scalar type of every `ElementType`,
/// read from the current interpreter's own numpy. Strong references, so a
/// cached pointer can't be reused by another object.
///
/// A copy must be re-fetched after any call that can run Python, because a
/// nested `dumps` may replace the group.
#[derive(Clone, Copy)]
pub(crate) struct NumpyTypes {
    pub(crate) ndarray: *mut PyTypeObject,
    /// Holds the scalar type of each `ElementType`, in `ElementType::ALL` order.
    scalars: [*mut PyTypeObject; ElementType::ALL.len()],
}

impl NumpyTypes {
    const EMPTY: Self = NumpyTypes {
        ndarray: ptr::null_mut(),
        scalars: [ptr::null_mut(); ElementType::ALL.len()],
    };

    fn all(&self) -> impl Iterator<Item = *mut PyTypeObject> {
        iter::once(self.ndarray).chain(self.scalars)
    }

    fn slots(&mut self) -> impl Iterator<Item = (&mut *mut PyTypeObject, &'static CStr)> {
        iter::once((&mut self.ndarray, c"ndarray")).chain(
            self.scalars
                .iter_mut()
                .zip(ElementType::ALL.map(numpy_name)),
        )
    }

    fn contains(&self, ty: *mut PyTypeObject) -> bool {
        self.ndarray == ty || self.scalars.contains(&ty)
    }

    /// Returns the element type whose numpy scalar type is `ty`, or `None`.
    pub(crate) fn scalar(&self, ty: *mut PyTypeObject) -> Option<ElementType> {
        self.scalars
            .iter()
            .zip(ElementType::ALL)
            .find_map(|(&t, e)| (t == ty).then_some(e))
    }
}

fn numpy_name(e: ElementType) -> &'static CStr {
    match e {
        ElementType::F64 => c"float64",
        ElementType::F32 => c"float32",
        ElementType::F16 => c"float16",
        ElementType::I64 => c"int64",
        ElementType::I32 => c"int32",
        ElementType::I16 => c"int16",
        ElementType::I8 => c"int8",
        ElementType::U64 => c"uint64",
        ElementType::U32 => c"uint32",
        ElementType::U16 => c"uint16",
        ElementType::U8 => c"uint8",
        ElementType::Bool => c"bool_",
        ElementType::Dt64 => c"datetime64",
    }
}

/// Names interned once in `TypeCache::init`, so lookups never build a string.
#[repr(C)]
pub(crate) struct InternedNames {
    pub(crate) datetime_mod: *mut PyObject,
    pub(crate) numpy: *mut PyObject,
    pub(crate) datetime_capi: *mut PyObject,
    pub(crate) utcoffset: *mut PyObject,
    pub(crate) array_struct: *mut PyObject,
    pub(crate) dtype: *mut PyObject,
    pub(crate) str: *mut PyObject,
}

impl InternedNames {
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
    /// Owner of the numpy group: the `sys.modules["numpy"]` it was read from.
    np_module: *mut PyObject,
    np: NumpyTypes,
    pub(crate) names: InternedNames,
}

/// Returns `d[key]` as a strong reference, or `None` when the key is missing.
unsafe fn get_item(d: *mut PyObject, key: *mut PyObject) -> PyResult<Option<*mut PyObject>> {
    let mut v = ptr::null_mut();
    match PyDict_GetItemRef(d, key, &mut v) {
        1 => Ok(Some(v)),
        0 => Ok(None),
        _ => Err(PendingPyErr),
    }
}

unsafe fn release<T>(slot: &mut *mut T) {
    let p = core::mem::replace(slot, ptr::null_mut());
    if !p.is_null() {
        Py_DECREF(p.cast());
    }
}

unsafe fn release_all(np: &mut NumpyTypes) {
    for (slot, _) in np.slots() {
        release(slot);
    }
}

/// Reads the numpy group from numpy's module dict; `None` if a name is missing.
unsafe fn read_numpy(d: *mut PyObject) -> PyResult<Option<NumpyTypes>> {
    let mut np = NumpyTypes::EMPTY;
    let mut found = Ok(true);
    for (slot, name) in np.slots() {
        match get_named(d, name) {
            Ok(Some(t)) => *slot = t.cast(),
            other => {
                found = other.map(|_| false);
                break;
            }
        }
    }
    match found {
        Ok(true) => Ok(Some(np)),
        other => {
            release_all(&mut np);
            other.map(|_| None)
        }
    }
}

unsafe fn get_named(d: *mut PyObject, name: &CStr) -> PyResult<Option<*mut PyObject>> {
    let key = PyUnicode_InternFromString(name.as_ptr());
    if key.is_null() {
        return Err(PendingPyErr);
    }
    let v = get_item(d, key);
    Py_DECREF(key);
    v
}

impl TypeCache {
    /// Returns -1 with a Python error set if interning a name fails.
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

    /// Returns the datetime group, loading it on first use. It is `None` while
    /// `_datetime` is not in `sys.modules`, is not a module, or has no valid
    /// `datetime_CAPI` capsule, and the load is retried on the next call.
    #[inline(always)]
    pub(crate) unsafe fn datetime(&mut self) -> PyResult<Option<DtTypes>> {
        if !self.dt_capsule.is_null() {
            return Ok(Some(self.dt));
        }
        self.load_datetime()
    }

    #[cold]
    #[inline(never)]
    unsafe fn load_datetime(&mut self) -> PyResult<Option<DtTypes>> {
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
            delta: (*api).DeltaType,
        };
        for t in [dt.datetime, dt.date, dt.time, dt.delta] {
            Py_INCREF(t.cast());
        }
        self.dt = dt;
        self.dt_capsule = cap;
        Ok(Some(dt))
    }

    /// Returns the cached numpy group if `ty` is one of its types, without a lookup.
    #[inline(always)]
    pub(crate) fn numpy_hit(&self, ty: *mut PyTypeObject) -> Option<NumpyTypes> {
        (!self.np_module.is_null() && self.np.contains(ty)).then_some(self.np)
    }

    /// Returns the numpy group if `ty` is one of its types.
    ///
    /// A hit is trusted. On a miss, the group is re-read only if
    /// `sys.modules["numpy"]` is no longer the module it was read from; if
    /// numpy is absent, not a module, or lacks a name, the result is `None`.
    pub(crate) unsafe fn numpy(&mut self, ty: *mut PyTypeObject) -> PyResult<Option<NumpyTypes>> {
        if let Some(np) = self.numpy_hit(ty) {
            return Ok(Some(np));
        }
        if self.names.numpy.is_null() {
            return Ok(None);
        }
        let Some(module) = get_item(PyImport_GetModuleDict(), self.names.numpy)? else {
            return Ok(None);
        };
        if module == self.np_module {
            Py_DECREF(module);
            return Ok(None);
        }
        self.clear_numpy();
        if PyModule_Check(module) == 0 {
            Py_DECREF(module);
            return Ok(None);
        }
        let np = match read_numpy(PyModule_GetDict(module)) {
            Ok(Some(np)) => np,
            other => {
                Py_DECREF(module);
                return other.map(|_| None);
            }
        };
        self.np = np;
        self.np_module = module;
        Ok(np.contains(ty).then_some(np))
    }

    unsafe fn clear_numpy(&mut self) {
        release(&mut self.np_module);
        release_all(&mut self.np);
    }

    pub(crate) unsafe fn traverse(&self, visit: visitproc, arg: *mut c_void) -> c_int {
        for t in self.np.all() {
            if !t.is_null() {
                let r = visit(t.cast(), arg);
                if r != 0 {
                    return r;
                }
            }
        }
        let held: [*mut PyObject; 13] = [
            self.np_module,
            self.dt_capsule,
            self.dt.datetime.cast(),
            self.dt.date.cast(),
            self.dt.time.cast(),
            self.dt.delta.cast(),
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

    /// Releases the groups and the names; on a zeroed cache it does nothing.
    pub(crate) unsafe fn clear(&mut self) {
        release(&mut self.dt_capsule);
        release(&mut self.dt.datetime);
        release(&mut self.dt.date);
        release(&mut self.dt.time);
        release(&mut self.dt.delta);
        self.clear_numpy();
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
        unsafe {
            assert!(matches!(c.numpy(ptr::null_mut()), Ok(None)));
        }
        assert!(c.dt_capsule.is_null() && c.names.datetime_mod.is_null() && c.np_module.is_null());
    }
}
