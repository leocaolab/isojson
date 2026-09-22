//! Output buffer that writes straight into the result `bytes` object.
//!
//! `dumps` used to write into a Rust `Vec` and copy it into a new `bytes` at
//! the end. Writing into the `bytes` object's own storage removes that final
//! copy. The initial capacity is the previous output size on this thread, so
//! repeated calls of a similar size rarely grow; at the end the object is
//! shrunk to the exact length.
//!
//! Growth allocates a new, larger `bytes` and copies into it, instead of
//! `_PyBytes_Resize`: on failure `_PyBytes_Resize` frees the object and the
//! bytes written so far would be lost. If even that allocation fails, writing
//! continues in a Rust `Vec` ("spill") and the result is built from it at the
//! end — the same behaviour the Vec-only buffer had.

use core::ptr;
use pyo3_ffi::*;
use std::cell::Cell;

thread_local! {
    /// Size of the previous output on this thread (plain data).
    static HINT: Cell<usize> = const { Cell::new(0) };
}

const MIN_CAP: usize = 64;

extern "C" {
    // Exported by libpython (not re-exported by pyo3-ffi). Resizes a bytes
    // object with refcount 1; on failure releases it, sets *obj = NULL and
    // raises MemoryError.
    fn _PyBytes_Resize(obj: *mut *mut PyObject, newsize: Py_ssize_t) -> core::ffi::c_int;
}

pub(crate) struct Out {
    /// The `bytes` being written, or null once spilled to `spill`.
    obj: *mut PyObject,
    spill: Vec<u8>,
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

impl Out {
    /// `None` if the first allocation fails (MemoryError is set).
    pub(crate) unsafe fn new() -> Option<Self> {
        let hint = HINT.get();
        let cap = (hint + hint / 8).max(MIN_CAP);
        let obj = PyBytes_FromStringAndSize(ptr::null(), cap as Py_ssize_t);
        if obj.is_null() {
            return None;
        }
        Some(Out {
            obj,
            spill: Vec::new(),
            ptr: PyBytes_AsString(obj).cast(),
            len: 0,
            cap,
        })
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// # Safety
    /// `len` bytes must have been written and `len <= capacity`.
    #[inline(always)]
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.len = len;
    }

    #[inline(always)]
    pub(crate) fn reserve(&mut self, n: usize) {
        if self.cap - self.len < n {
            self.grow(n);
        }
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, n: usize) {
        let want = (self.len + n).max(self.cap * 2);
        unsafe {
            if !self.obj.is_null() {
                let bigger = PyBytes_FromStringAndSize(ptr::null(), want as Py_ssize_t);
                if !bigger.is_null() {
                    let p: *mut u8 = PyBytes_AsString(bigger).cast();
                    ptr::copy_nonoverlapping(self.ptr, p, self.len);
                    Py_DECREF(self.obj);
                    self.obj = bigger;
                    self.ptr = p;
                    self.cap = want;
                    return;
                }
                // Out of memory for a bytes object: keep going in a Vec.
                PyErr_Clear();
                let mut v = Vec::with_capacity(want);
                v.extend_from_slice(core::slice::from_raw_parts(self.ptr, self.len));
                Py_DECREF(self.obj);
                self.obj = ptr::null_mut();
                self.spill = v;
            } else {
                self.spill.set_len(self.len);
                self.spill.reserve(want - self.len);
            }
            self.ptr = self.spill.as_mut_ptr();
            self.cap = self.spill.capacity();
        }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, b: u8) {
        self.reserve(1);
        unsafe {
            *self.ptr.add(self.len) = b;
        }
        self.len += 1;
    }

    #[inline(always)]
    pub(crate) fn extend_from_slice(&mut self, s: &[u8]) {
        self.reserve(s.len());
        unsafe {
            ptr::copy_nonoverlapping(s.as_ptr(), self.ptr.add(self.len), s.len());
        }
        self.len += s.len();
    }

    /// Hand the result to Python: a `bytes` of exactly `len`, or null with
    /// MemoryError set.
    pub(crate) unsafe fn finish(mut self) -> *mut PyObject {
        HINT.set(self.len);
        if self.obj.is_null() {
            return PyBytes_FromStringAndSize(self.spill.as_ptr().cast(), self.len as Py_ssize_t);
        }
        let mut obj = core::mem::replace(&mut self.obj, ptr::null_mut());
        if self.len != self.cap && _PyBytes_Resize(&mut obj, self.len as Py_ssize_t) < 0 {
            return ptr::null_mut(); // obj already released by CPython
        }
        obj
    }
}

impl Drop for Out {
    fn drop(&mut self) {
        if !self.obj.is_null() {
            unsafe { Py_DECREF(self.obj) };
        }
    }
}
