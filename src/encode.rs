//! `dumps`: Python object -> JSON bytes, written in a single pass.
//!
//! Multi-interpreter safety: the encoder only *borrows* objects for the
//! duration of one call, on the calling thread, under the calling
//! interpreter's GIL. Across calls, only two things are kept: a thread-local
//! output size hint (plain data), and the per-interpreter `TypeCache` in this
//! module's state, which holds types looked up in the calling interpreter's
//! own `sys.modules` (design C1).
//!
//! Reentrancy (design FR-13): `guard` is set when Python code can run during
//! the walk — a `default` was given, `OPT_SERIALIZE_NUMPY` is set, or the
//! datetime types are loaded (`utcoffset()` runs Python). While it is set,
//! every item and key taken from a list or dict holds a reference while it is
//! serialized, since that Python code could mutate the container. With
//! `guard` false no Python code runs and borrowing is safe.

use core::ffi::CStr;
use core::ptr;
use pyo3_ffi::*;

use crate::datetime::{fmt_hms, fmt_offset, fmt_ymd, Parts};
use crate::decline::{reason_message, reason_note, Reason};
use crate::float::{small_copy, write_f64};
use crate::numpy::Outcome;
use crate::out::Out;
use crate::types::{DtTypes, TypeCache};
use crate::*;

/// Nested containers at or beyond this depth fail (orjson: 254 ok, 255 fails).
const MAX_DEPTH: u32 = 254;
/// Chained `default` calls allowed before giving up (orjson: 255).
const MAX_DEFAULT_DEPTH: u32 = 255;

const ALL_OPTS: u32 = 4095;
/// Options that change *how* a type we serialize natively is written, and
/// that we do not implement. Silently ignoring them would produce output the
/// caller did not ask for, so they are rejected.
const UNSUPPORTED_OPTS: &[(u32, &str)] = &[(OPT_NON_STR_KEYS, "OPT_NON_STR_KEYS")];
// OPT_PASSTHROUGH_DATACLASS only affects dataclass objects, which isojson
// never serializes natively (they always go to `default`), so it is accepted
// and has no effect. The datetime options and OPT_SERIALIZE_NUMPY take effect
// (design FR-5, FR-11).

static HEX: &[u8; 16] = b"0123456789abcdef";

/// 0 = copy through; otherwise the escape letter (`u` = \u00XX form).
static ESCAPE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 0x20 {
        t[i] = b'u';
        i += 1;
    }
    t[0x08] = b'b';
    t[0x09] = b't';
    t[0x0a] = b'n';
    t[0x0c] = b'f';
    t[0x0d] = b'r';
    t[b'"' as usize] = b'"';
    t[b'\\' as usize] = b'\\';
    t
};

unsafe fn raise_type_error(msg: &str) {
    let c = std::ffi::CString::new(msg).unwrap_or_else(|_| c"isojson: error".into());
    PyErr_SetString(PyExc_TypeError, c.as_ptr());
}

unsafe fn type_name(obj: *mut PyObject) -> String {
    let ty = Py_TYPE(obj);
    CStr::from_ptr((*ty).tp_name).to_string_lossy().into_owned()
}

/// Raise `TypeError(msg)` with the currently raised exception as `__cause__`.
pub(crate) unsafe fn raise_type_error_from_current(msg: &str) {
    raise_type_error_caused_by(msg, PyErr_GetRaisedException());
}

/// Raise `TypeError(msg)` with `cause` (a strong reference, stolen; may be
/// null) as `__cause__`.
unsafe fn raise_type_error_caused_by(msg: &str, cause: *mut PyObject) {
    raise_type_error(msg);
    if !cause.is_null() {
        let exc = PyErr_GetRaisedException();
        PyException_SetCause(exc, cause); // steals `cause`
        PyErr_SetRaisedException(exc); // steals `exc`
    }
}

/// `<Exc>: <str(exc)>`, or `<Exc>` when the text is empty or `str()` fails
/// (the exception itself still travels as `__cause__`).
unsafe fn describe_exception(exc: *mut PyObject) -> String {
    let name = type_name(exc);
    let s = PyObject_Str(exc);
    if s.is_null() {
        PyErr_Clear();
        return name;
    }
    let text = crate::strfast::as_utf8(s).map(|b| String::from_utf8_lossy(b).into_owned());
    Py_DECREF(s);
    match text {
        Some(t) if !t.is_empty() => format!("{name}: {t}"),
        _ => {
            PyErr_Clear();
            name
        }
    }
}

fn write_ymd(out: &mut Out, y: u16, mo: u8, d: u8) {
    let mut b = [0u8; 10];
    let n = fmt_ymd(&mut b, y, mo, d);
    small_copy(out, &b[..n]);
}

fn write_offset(out: &mut Out, total_us: i64, opts: u32) {
    let mut b = [0u8; 16];
    let n = fmt_offset(&mut b, total_us, opts);
    small_copy(out, &b[..n]);
}

/// A quoted `YYYY-MM-DDTHH:MM:SS[.ffffff][offset]`, the datetime text of
/// design §1a for `datetime` and `datetime64` alike.
pub(crate) fn write_datetime(out: &mut Out, p: &Parts, offset: Option<i64>, opts: u32) {
    out.push(b'"');
    write_ymd(out, p.y, p.mo, p.d);
    out.push(b'T');
    let mut b = [0u8; 15];
    let n = fmt_hms(&mut b, p.h, p.mi, p.s, p.us, opts);
    small_copy(out, &b[..n]);
    if let Some(o) = offset {
        write_offset(out, o, opts);
    }
    out.push(b'"');
}

/// Attach `note` to the exception being raised (`exc.add_note(note)`). If
/// that itself fails, its error is cleared and the original exception stands.
unsafe fn add_note(note: &str) {
    let exc = PyErr_GetRaisedException();
    if exc.is_null() {
        return;
    }
    let name = PyUnicode_FromString(c"add_note".as_ptr());
    let text = PyUnicode_FromStringAndSize(note.as_ptr().cast(), note.len() as Py_ssize_t);
    let r = if name.is_null() || text.is_null() {
        ptr::null_mut()
    } else {
        PyObject_CallMethodOneArg(exc, name, text)
    };
    if r.is_null() {
        PyErr_Clear();
    } else {
        Py_DECREF(r);
    }
    for o in [name, text] {
        if !o.is_null() {
            Py_DECREF(o);
        }
    }
    PyErr_SetRaisedException(exc);
}

/// `None` → `None`; a timedelta → its total µs. Consumes the reference.
unsafe fn delta_us(r: *mut PyObject) -> Option<i64> {
    let out = if r == Py_None() {
        None
    } else {
        let days = PyDateTime_DELTA_GET_DAYS(r) as i64;
        let secs = PyDateTime_DELTA_GET_SECONDS(r) as i64;
        let us = PyDateTime_DELTA_GET_MICROSECONDS(r) as i64;
        Some((days * 86_400 + secs) * 1_000_000 + us)
    };
    Py_DECREF(r);
    out
}

/// An exact `str`, `int`, `float`, `bool` or `None`: serializing it reads the
/// object's own data and never runs Python code (no `__index__`, `__float__`,
/// `utcoffset()` or `default`), so nothing can mutate the container holding
/// it meanwhile, and the FR-13 guard needn't hold a reference to it (or to its
/// dict key). Holding one costs a refcount write to every leaf, which on
/// x86_64 made a list of 100k floats ~40% slower.
#[inline(always)]
unsafe fn runs_no_python(obj: *mut PyObject) -> bool {
    let ty = Py_TYPE(obj);
    ty == &raw mut PyFloat_Type
        || ty == &raw mut PyUnicode_Type
        || ty == &raw mut PyLong_Type
        || ty == &raw mut PyBool_Type
        || obj == Py_None()
}

#[inline]
pub(crate) fn write_int<I: itoa::Integer>(out: &mut Out, v: I) {
    let mut b = itoa::Buffer::new();
    crate::float::small_copy(out, b.format(v).as_bytes());
}

/// Why `invoke_default` produced no object. Both leave an exception set.
pub(crate) enum DefaultErr {
    /// `MAX_DEFAULT_DEPTH` chained calls.
    DepthLimit,
    /// `default` itself raised; the exception is the `TypeError`'s cause.
    Raised,
}

pub(crate) struct Encoder {
    pub(crate) out: Out,
    default: *mut PyObject,
    pub(crate) opts: u32,
    pub(crate) depth: u32,
    default_depth: u32,
    /// This interpreter's type cache (module state).
    pub(crate) cache: *mut TypeCache,
    /// FR-13: Python code can run during the walk.
    pub(crate) guard: bool,
    /// The datetime group, resolved at the start of `dumps`.
    dt: Option<DtTypes>,
    /// FR-3's `sys.modules["numpy"]` re-check already ran in this call.
    np_rechecked: bool,
}

impl Encoder {
    #[inline]
    pub(crate) fn indent(&mut self) -> bool {
        self.opts & OPT_INDENT_2 != 0
    }

    #[inline]
    pub(crate) fn newline_indent(&mut self, level: u32) {
        self.out.push(b'\n');
        for _ in 0..level {
            self.out.extend_from_slice(b"  ");
        }
    }

    unsafe fn enter(&mut self) -> bool {
        if self.depth >= MAX_DEPTH {
            raise_type_error("Recursion limit reached");
            return false;
        }
        self.depth += 1;
        true
    }

    unsafe fn serialize(&mut self, obj: *mut PyObject) -> bool {
        let ty = Py_TYPE(obj);
        if ty == &raw mut PyUnicode_Type {
            return self.str(obj);
        }
        if ty == &raw mut PyLong_Type {
            return self.int(obj);
        }
        if ty == &raw mut PyFloat_Type {
            write_f64(&mut self.out, PyFloat_AsDouble(obj));
            return true;
        }
        if ty == &raw mut PyBool_Type {
            // constant-length copy per branch (a slice chosen by `if` has a
            // non-constant length and would become a memcpy call)
            if obj == Py_True() {
                self.out.extend_from_slice(b"true");
            } else {
                self.out.extend_from_slice(b"false");
            }
            return true;
        }
        if obj == Py_None() {
            self.out.extend_from_slice(b"null");
            return true;
        }
        if ty == &raw mut PyDict_Type {
            return self.dict(obj);
        }
        if ty == &raw mut PyList_Type {
            return self.list(obj);
        }
        if ty == &raw mut PyTuple_Type {
            return self.tuple(obj);
        }
        self.serialize_other(obj, ty)
    }

    /// Everything past the exact builtins: datetimes, subclasses, numpy and
    /// `default`, out of line so `serialize` stays as small as 0.1's.
    #[inline(never)]
    unsafe fn serialize_other(&mut self, obj: *mut PyObject, ty: *mut PyTypeObject) -> bool {
        if let Some(dt) = self.dt {
            if let Some(ok) = self.datetime_like(obj, ty, dt) {
                return ok;
            }
        }
        if self.opts & OPT_PASSTHROUGH_SUBCLASS == 0 {
            if PyUnicode_Check(obj) != 0 {
                return self.str(obj);
            }
            if PyLong_Check(obj) != 0 {
                return self.int(obj);
            }
            if PyDict_Check(obj) != 0 {
                return self.dict(obj);
            }
            if PyList_Check(obj) != 0 {
                return self.list(obj);
            }
        }
        // FR-13: `datetime` may have been imported since `dumps` started (by
        // a `default`, say). Only possible while guarded, and only retried
        // for objects that missed every fast path.
        if self.dt.is_none() && self.guard && self.opts & OPT_PASSTHROUGH_DATETIME == 0 {
            match (*self.cache).datetime() {
                Ok(dt) => self.dt = dt,
                Err(PyErrSet) => return false,
            }
            if let Some(dt) = self.dt {
                if let Some(ok) = self.datetime_like(obj, ty, dt) {
                    return ok;
                }
            }
        }
        if self.opts & OPT_SERIALIZE_NUMPY != 0 {
            // A hit is trusted; a miss re-checks `sys.modules["numpy"]` once
            // per call (FR-3), not once per object bound for `default`
            let np = match (*self.cache).numpy_hit(ty) {
                Some(np) => Ok(Some(np)),
                None if !self.np_rechecked => {
                    self.np_rechecked = true;
                    (*self.cache).numpy(ty)
                }
                None => Ok(None),
            };
            match np {
                Err(PyErrSet) => return false,
                Ok(Some(np)) => {
                    let outcome = if ty == np.ndarray {
                        Some(crate::numpy::serialize_array(self, obj))
                    } else {
                        crate::numpy::serialize_scalar(self, obj, np)
                    };
                    match outcome {
                        Some(Outcome::Written) => return true,
                        Some(Outcome::Error(PyErrSet)) => return false,
                        Some(Outcome::Declined(reason)) => {
                            return self.call_default_declined(obj, reason)
                        }
                        // an unrecognized numpy type: the ordinary path (FR-8)
                        None => {}
                    }
                }
                Ok(None) => {}
            }
        }
        self.call_default(obj)
    }

    /// Exact `datetime` / `date` / `time`: `Some(ok)` once written (or
    /// failed); `None` if `obj` is none of them or `PASSTHROUGH_DATETIME`
    /// sends them to `default`.
    #[inline]
    unsafe fn datetime_like(
        &mut self,
        obj: *mut PyObject,
        ty: *mut PyTypeObject,
        dt: DtTypes,
    ) -> Option<bool> {
        if self.opts & OPT_PASSTHROUGH_DATETIME != 0 {
            return None;
        }
        if ty == dt.datetime {
            Some(self.datetime(obj, dt))
        } else if ty == dt.date {
            Some(self.date(obj))
        } else if ty == dt.time {
            Some(self.time(obj, dt))
        } else {
            None
        }
    }

    /// Serialize a borrowed container item. With `guard` false no Python code
    /// can run during serialization, so the container cannot be mutated and
    /// the borrow is safe as is. With it set, that code could mutate the
    /// container and drop the item, so hold a reference across the call —
    /// unless the item is a leaf that runs no Python itself (FR-13).
    #[inline]
    unsafe fn guarded(&mut self, item: *mut PyObject) -> bool {
        if !self.guard || runs_no_python(item) {
            return self.serialize(item);
        }
        Py_INCREF(item);
        let ok = self.serialize(item);
        Py_DECREF(item);
        ok
    }

    /// Call `default(obj)`. Only reached when a `default` was given.
    pub(crate) unsafe fn invoke_default(
        &mut self,
        obj: *mut PyObject,
    ) -> Result<*mut PyObject, DefaultErr> {
        debug_assert!(!self.default.is_null());
        if self.default_depth >= MAX_DEFAULT_DEPTH {
            raise_type_error("default serializer exceeds recursion limit");
            return Err(DefaultErr::DepthLimit);
        }
        debug_assert!(self.guard);
        let r = PyObject_CallOneArg(self.default, obj);
        if r.is_null() {
            raise_type_error_from_current(&format!(
                "Type is not JSON serializable: {}",
                type_name(obj)
            ));
            return Err(DefaultErr::Raised);
        }
        Ok(r)
    }

    /// Serialize what `default` returned, and release it.
    unsafe fn serialize_default_result(&mut self, r: *mut PyObject) -> bool {
        self.default_depth += 1;
        let ok = self.serialize(r);
        self.default_depth -= 1;
        Py_DECREF(r);
        ok
    }

    unsafe fn call_default(&mut self, obj: *mut PyObject) -> bool {
        if self.default.is_null() {
            raise_type_error(&format!(
                "Type is not JSON serializable: {}",
                type_name(obj)
            ));
            return false;
        }
        match self.invoke_default(obj) {
            Ok(r) => self.serialize_default_result(r),
            Err(DefaultErr::DepthLimit | DefaultErr::Raised) => false,
        }
    }

    /// A numpy object isojson declined (FR-7): the whole object goes to
    /// `default` when one is given. Without one, the reason's message is
    /// raised. Either way, when an error results from the decline itself or
    /// from `default` raising, `reason_note` is attached with `add_note`, so
    /// the evidence isn't lost. `DepthLimit` gets no note.
    pub(crate) unsafe fn call_default_declined(
        &mut self,
        obj: *mut PyObject,
        reason: Reason,
    ) -> bool {
        if self.default.is_null() {
            raise_type_error(&reason_message(&reason));
            add_note(&reason_note(&reason));
            return false;
        }
        match self.invoke_default(obj) {
            Ok(r) => self.serialize_default_result(r),
            Err(DefaultErr::Raised) => {
                add_note(&reason_note(&reason));
                false
            }
            Err(DefaultErr::DepthLimit) => false,
        }
    }

    /// `obj.utcoffset()` as total µs; `None` when naive (design §1a). The
    /// tzinfo field is read first, so naive objects call no method (D7).
    ///
    /// This is CPython's own `call_tzinfo_method` done inline:
    /// `tzinfo.utcoffset(arg)` (`arg` is the datetime, or `None` for a
    /// `time`), then `None` passes, and a timedelta (or subclass) must lie
    /// strictly between −24 h and 24 h. `datetime.utcoffset()` builds that call
    /// from a format string, which costs about 40 ns per value.
    ///
    /// A raising tzinfo is exactly what `obj.utcoffset()` would propagate;
    /// it becomes DV-4b's `TypeError`, with the exception as `__cause__`. For
    /// an invalid result (DV-15), `obj.utcoffset()` itself is asked, so the
    /// error is CPython's own, word for word; that path calls the tzinfo a
    /// second time.
    unsafe fn utcoffset_of(
        &mut self,
        obj: *mut PyObject,
        tzinfo: *mut PyObject,
        arg: *mut PyObject,
        what: &str,
        delta: *mut PyTypeObject,
    ) -> R<Option<i64>> {
        if tzinfo == Py_None() {
            return Ok(None);
        }
        debug_assert!(self.guard);
        let names = &(*self.cache).names;
        let r = PyObject_CallMethodOneArg(tzinfo, names.utcoffset, arg);
        if r.is_null() {
            return Err(self.utcoffset_raised(what));
        }
        if r == Py_None() {
            return Ok(delta_us(r));
        }
        // CPython's bounds: strictly between -timedelta(1) and timedelta(1)
        if PyObject_TypeCheck(r, delta) != 0 {
            let days = PyDateTime_DELTA_GET_DAYS(r);
            if days == 0
                || (days == -1
                    && (PyDateTime_DELTA_GET_SECONDS(r) != 0
                        || PyDateTime_DELTA_GET_MICROSECONDS(r) != 0))
            {
                return Ok(delta_us(r));
            }
        }
        Py_DECREF(r);
        // invalid: ask `obj.utcoffset()`, whose answer is the truth (§1a) —
        // normally CPython's own error for it
        let r = PyObject_CallMethodNoArgs(obj, names.utcoffset);
        if r.is_null() {
            return Err(self.utcoffset_raised(what));
        }
        Ok(delta_us(r))
    }

    /// DV-4b: `TypeError("<what>.utcoffset() raised <Exc>: <msg>")` from the
    /// exception being raised, which becomes its `__cause__`.
    unsafe fn utcoffset_raised(&mut self, what: &str) -> PyErrSet {
        let cause = PyErr_GetRaisedException();
        let msg = format!("{what}.utcoffset() raised {}", describe_exception(cause));
        raise_type_error_caused_by(&msg, cause);
        PyErrSet
    }

    /// `dt.isoformat()` after `NAIVE_UTC` / `OMIT_MICROSECONDS`, with a zero
    /// offset written `Z` under `UTC_Z` (design §1a, FR-5).
    unsafe fn datetime(&mut self, obj: *mut PyObject, dt: DtTypes) -> bool {
        let tzinfo = PyDateTime_DATE_GET_TZINFO(obj);
        let offset = match self.utcoffset_of(obj, tzinfo, obj, "datetime", dt.delta) {
            Ok(Some(o)) => Some(o),
            Ok(None) if self.opts & OPT_NAIVE_UTC != 0 => Some(0),
            Ok(None) => None,
            Err(PyErrSet) => return false,
        };
        let p = Parts {
            y: PyDateTime_GET_YEAR(obj) as u16,
            mo: PyDateTime_GET_MONTH(obj) as u8,
            d: PyDateTime_GET_DAY(obj) as u8,
            h: PyDateTime_DATE_GET_HOUR(obj) as u8,
            mi: PyDateTime_DATE_GET_MINUTE(obj) as u8,
            s: PyDateTime_DATE_GET_SECOND(obj) as u8,
            us: PyDateTime_DATE_GET_MICROSECOND(obj) as u32,
        };
        write_datetime(&mut self.out, &p, offset, self.opts);
        true
    }

    /// `d.isoformat()`. No options apply.
    unsafe fn date(&mut self, obj: *mut PyObject) -> bool {
        self.out.push(b'"');
        write_ymd(
            &mut self.out,
            PyDateTime_GET_YEAR(obj) as u16,
            PyDateTime_GET_MONTH(obj) as u8,
            PyDateTime_GET_DAY(obj) as u8,
        );
        self.out.push(b'"');
        true
    }

    /// `t.isoformat()` after `OMIT_MICROSECONDS`, with its offset if it has
    /// one (DV-12). `NAIVE_UTC` and `UTC_Z` don't apply to `time`.
    unsafe fn time(&mut self, obj: *mut PyObject, dt: DtTypes) -> bool {
        let opts = self.opts & !(OPT_NAIVE_UTC | OPT_UTC_Z);
        let tzinfo = PyDateTime_TIME_GET_TZINFO(obj);
        let offset = match self.utcoffset_of(obj, tzinfo, Py_None(), "time", dt.delta) {
            Ok(o) => o,
            Err(PyErrSet) => return false,
        };
        self.out.push(b'"');
        let mut b = [0u8; 15];
        let n = fmt_hms(
            &mut b,
            PyDateTime_TIME_GET_HOUR(obj) as u8,
            PyDateTime_TIME_GET_MINUTE(obj) as u8,
            PyDateTime_TIME_GET_SECOND(obj) as u8,
            PyDateTime_TIME_GET_MICROSECOND(obj) as u32,
            opts,
        );
        small_copy(&mut self.out, &b[..n]);
        if let Some(o) = offset {
            write_offset(&mut self.out, o, opts);
        }
        self.out.push(b'"');
        true
    }

    unsafe fn str(&mut self, obj: *mut PyObject) -> bool {
        match crate::strfast::as_utf8(obj) {
            Some(b) => {
                write_escaped(&mut self.out, b);
                true
            }
            None => {
                raise_type_error("str is not valid UTF-8: surrogates not allowed");
                false
            }
        }
    }

    unsafe fn int(&mut self, obj: *mut PyObject) -> bool {
        let mut overflow = 0;
        let v = PyLong_AsLongLongAndOverflow(obj, &mut overflow);
        if overflow == 0 {
            if v == -1 && !PyErr_Occurred().is_null() {
                return false;
            }
            if self.opts & OPT_STRICT_INTEGER != 0
                && !(-9007199254740991..=9007199254740991).contains(&v)
            {
                raise_type_error("Integer exceeds 53-bit range");
                return false;
            }
            write_int(&mut self.out, v);
            return true;
        }
        if overflow > 0 && self.opts & OPT_STRICT_INTEGER == 0 {
            let u = PyLong_AsUnsignedLongLong(obj);
            if u != u64::MAX || PyErr_Occurred().is_null() {
                write_int(&mut self.out, u);
                return true;
            }
            PyErr_Clear();
        }
        if self.opts & OPT_STRICT_INTEGER != 0 {
            raise_type_error("Integer exceeds 53-bit range");
        } else {
            raise_type_error("Integer exceeds 64-bit range");
        }
        false
    }

    unsafe fn list(&mut self, obj: *mut PyObject) -> bool {
        let n = PyList_GET_SIZE(obj);
        if n == 0 {
            if !self.enter() {
                return false;
            }
            self.depth -= 1;
            self.out.extend_from_slice(b"[]");
            return true;
        }
        if !self.enter() {
            return false;
        }
        self.out.push(b'[');
        let mut i = 0;
        // Re-read the size every step: a `default` callback may mutate the list.
        while i < PyList_GET_SIZE(obj) {
            if i > 0 {
                self.out.push(b',');
            }
            if self.indent() {
                self.newline_indent(self.depth);
            }
            let item = PyList_GET_ITEM(obj, i);
            if !self.guarded(item) {
                return false;
            }
            i += 1;
        }
        self.depth -= 1;
        if self.indent() {
            self.newline_indent(self.depth);
        }
        self.out.push(b']');
        true
    }

    unsafe fn tuple(&mut self, obj: *mut PyObject) -> bool {
        let n = PyTuple_GET_SIZE(obj);
        if !self.enter() {
            return false;
        }
        if n == 0 {
            self.depth -= 1;
            self.out.extend_from_slice(b"[]");
            return true;
        }
        self.out.push(b'[');
        for i in 0..n {
            if i > 0 {
                self.out.push(b',');
            }
            if self.indent() {
                self.newline_indent(self.depth);
            }
            // tuples are immutable: the borrowed item lives as long as `obj`
            if !self.serialize(PyTuple_GET_ITEM(obj, i)) {
                return false;
            }
        }
        self.depth -= 1;
        if self.indent() {
            self.newline_indent(self.depth);
        }
        self.out.push(b']');
        true
    }

    unsafe fn key_utf8<'a>(key: *mut PyObject) -> Option<&'a [u8]> {
        if PyUnicode_Check(key) == 0 {
            raise_type_error("Dict key must be str");
            return None;
        }
        let k = crate::strfast::as_utf8(key);
        if k.is_none() {
            raise_type_error("str is not valid UTF-8: surrogates not allowed");
        }
        k
    }

    unsafe fn write_key(&mut self, k: &[u8], first: bool) {
        if !first {
            self.out.push(b',');
        }
        if self.indent() {
            self.newline_indent(self.depth);
        }
        write_escaped(&mut self.out, k);
        if self.indent() {
            self.out.extend_from_slice(b": ");
        } else {
            self.out.push(b':');
        }
    }

    unsafe fn dict(&mut self, obj: *mut PyObject) -> bool {
        if !self.enter() {
            return false;
        }
        if PyDict_Size(obj) == 0 {
            self.depth -= 1;
            self.out.extend_from_slice(b"{}");
            return true;
        }
        self.out.push(b'{');
        let ok = if self.opts & OPT_SORT_KEYS != 0 {
            self.dict_sorted(obj)
        } else {
            self.dict_in_order(obj)
        };
        if !ok {
            return false;
        }
        self.depth -= 1;
        if self.indent() {
            self.newline_indent(self.depth);
        }
        self.out.push(b'}');
        true
    }

    unsafe fn dict_in_order(&mut self, obj: *mut PyObject) -> bool {
        let mut pos: Py_ssize_t = 0;
        let mut key: *mut PyObject = ptr::null_mut();
        let mut value: *mut PyObject = ptr::null_mut();
        let mut first = true;
        while PyDict_Next(obj, &mut pos, &mut key, &mut value) != 0 {
            let Some(k) = Self::key_utf8(key) else {
                return false;
            };
            self.write_key(k, first);
            if !self.guard || runs_no_python(value) {
                if !self.serialize(value) {
                    return false;
                }
            } else {
                // Python code may run (FR-13) and mutate this dict; keep the
                // key (whose UTF-8 we already wrote) and value alive.
                Py_INCREF(key);
                let ok = self.guarded(value);
                Py_DECREF(key);
                if !ok {
                    return false;
                }
            }
            first = false;
        }
        true
    }

    unsafe fn dict_sorted(&mut self, obj: *mut PyObject) -> bool {
        let mut items: Vec<(*mut PyObject, *mut PyObject)> =
            Vec::with_capacity(PyDict_Size(obj) as usize);
        let mut pos: Py_ssize_t = 0;
        let mut key: *mut PyObject = ptr::null_mut();
        let mut value: *mut PyObject = ptr::null_mut();
        while PyDict_Next(obj, &mut pos, &mut key, &mut value) != 0 {
            Py_INCREF(key);
            Py_INCREF(value);
            items.push((key, value));
        }
        let release = |items: &[(*mut PyObject, *mut PyObject)]| {
            for &(k, v) in items {
                Py_DECREF(v);
                Py_DECREF(k);
            }
        };
        // Validate + fetch UTF-8 for every key first; UTF-8 byte order equals
        // code point order, which is what Python's str comparison uses.
        let mut keyed: Vec<(&[u8], *mut PyObject)> = Vec::with_capacity(items.len());
        for &(k, v) in &items {
            match Self::key_utf8(k) {
                Some(b) => keyed.push((b, v)),
                None => {
                    release(&items);
                    return false;
                }
            }
        }
        keyed.sort_by(|a, b| a.0.cmp(b.0));
        let mut ok = true;
        for (i, &(k, v)) in keyed.iter().enumerate() {
            self.write_key(k, i == 0);
            if !self.serialize(v) {
                ok = false;
                break;
            }
        }
        release(&items);
        ok
    }
}

/// Copy bytes of `s` from `i` to `dst` until the first byte that needs
/// escaping (or the end). Returns its index; `dst` is advanced past the copied
/// bytes. May store up to 15 bytes beyond what it reports as copied, so the
/// caller must have 16 bytes of slack.
#[inline(always)]
unsafe fn copy_plain(s: &[u8], mut i: usize, dst: &mut *mut u8) -> usize {
    let n = s.len();
    let p = s.as_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        // SSE2 is part of the x86_64 baseline: no runtime detection needed.
        let quote = _mm_set1_epi8(b'"' as i8);
        let bslash = _mm_set1_epi8(b'\\' as i8);
        let x1f = _mm_set1_epi8(0x1f);
        while i + 16 <= n {
            let v = _mm_loadu_si128(p.add(i).cast());
            _mm_storeu_si128((*dst).cast(), v);
            // v <= 0x1f  <=>  min(v, 0x1f) == v   (unsigned)
            let ctrl = _mm_cmpeq_epi8(_mm_min_epu8(v, x1f), v);
            let hit = _mm_or_si128(
                ctrl,
                _mm_or_si128(_mm_cmpeq_epi8(v, quote), _mm_cmpeq_epi8(v, bslash)),
            );
            let m = _mm_movemask_epi8(hit) as u32;
            if m != 0 {
                let k = m.trailing_zeros() as usize;
                *dst = dst.add(k);
                return i + k;
            }
            *dst = dst.add(16);
            i += 16;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        // NEON is part of the aarch64 baseline.
        let quote = vdupq_n_u8(b'"');
        let bslash = vdupq_n_u8(b'\\');
        let x20 = vdupq_n_u8(0x20);
        while i + 16 <= n {
            let v = vld1q_u8(p.add(i));
            vst1q_u8(*dst, v);
            let hit = vorrq_u8(
                vcltq_u8(v, x20),
                vorrq_u8(vceqq_u8(v, quote), vceqq_u8(v, bslash)),
            );
            // narrow each 0x00/0xFF byte to a 4-bit nibble of one u64
            let nib = vget_lane_u64(
                vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(hit), 4)),
                0,
            );
            if nib != 0 {
                let k = (nib.trailing_zeros() / 4) as usize;
                *dst = dst.add(k);
                return i + k;
            }
            *dst = dst.add(16);
            i += 16;
        }
    }

    // 8 bytes at a time (portable), then byte by byte.
    while i + 8 <= n {
        let w = crate::swar::load(s, i);
        if crate::swar::has_special(w) {
            break;
        }
        core::ptr::write_unaligned((*dst).cast::<u64>(), w);
        *dst = dst.add(8);
        i += 8;
    }
    while i < n && ESCAPE[*p.add(i) as usize] == 0 {
        **dst = *p.add(i);
        *dst = dst.add(1);
        i += 1;
    }
    i
}

#[inline]
fn write_escaped(out: &mut Out, s: &[u8]) {
    let n = s.len();
    // 2 quotes + 16 bytes of slack for copy_plain's full-width stores
    out.reserve(n + 2 + 16);
    unsafe {
        let mut dst = out.as_mut_ptr().add(out.len());
        *dst = b'"';
        dst = dst.add(1);
        let mut i = copy_plain(s, 0, &mut dst);
        while i < n {
            let b = *s.get_unchecked(i);
            let e = ESCAPE[b as usize];
            if e == b'u' {
                core::ptr::copy_nonoverlapping(
                    [
                        b'\\',
                        b'u',
                        b'0',
                        b'0',
                        HEX[(b >> 4) as usize],
                        HEX[(b & 0xf) as usize],
                    ]
                    .as_ptr(),
                    dst,
                    6,
                );
                dst = dst.add(6);
            } else {
                *dst = b'\\';
                *dst.add(1) = e;
                dst = dst.add(2);
            }
            i += 1;
            // Escapes grow the output: make sure the rest still fits (plus
            // one more escape, the closing quote, and the slack).
            let written = dst.offset_from(out.as_mut_ptr()) as usize;
            out.set_len(written);
            out.reserve(n - i + 6 + 1 + 16);
            dst = out.as_mut_ptr().add(written);
            i = copy_plain(s, i, &mut dst);
        }
        *dst = b'"';
        dst = dst.add(1);
        let written = dst.offset_from(out.as_mut_ptr()) as usize;
        out.set_len(written);
    }
}

unsafe fn kw_is(name: *mut PyObject, s: &CStr) -> bool {
    PyUnicode_CompareWithASCIIString(name, s.as_ptr()) == 0
}

pub(crate) unsafe extern "C" fn dumps(
    module: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
    kwnames: *mut PyObject,
) -> *mut PyObject {
    let nargs = PyVectorcall_NARGS(nargs as usize);
    if nargs < 1 {
        raise_type_error("dumps() missing required argument 'obj' (pos 1)");
        return ptr::null_mut();
    }
    if nargs > 3 {
        raise_type_error(&format!(
            "dumps() takes at most 3 positional arguments ({nargs} given)"
        ));
        return ptr::null_mut();
    }
    let obj = *args;
    let mut default: *mut PyObject = if nargs >= 2 {
        *args.add(1)
    } else {
        ptr::null_mut()
    };
    let mut option: *mut PyObject = if nargs >= 3 {
        *args.add(2)
    } else {
        ptr::null_mut()
    };
    if !kwnames.is_null() {
        let nkw = PyTuple_GET_SIZE(kwnames);
        for i in 0..nkw {
            let name = PyTuple_GET_ITEM(kwnames, i);
            let val = *args.add(nargs as usize + i as usize);
            if kw_is(name, c"default") && nargs < 2 {
                default = val;
            } else if kw_is(name, c"option") && nargs < 3 {
                option = val;
            } else {
                raise_type_error("dumps() got an unexpected or duplicate keyword argument");
                return ptr::null_mut();
            }
        }
    }
    if default == Py_None() {
        default = ptr::null_mut();
    }
    if !default.is_null() && PyCallable_Check(default) == 0 {
        raise_type_error("default must be a callable");
        return ptr::null_mut();
    }

    let mut opts: u32 = 0;
    if !option.is_null() && option != Py_None() {
        if PyLong_Check(option) == 0 {
            raise_type_error("Invalid opts");
            return ptr::null_mut();
        }
        let v = PyLong_AsLongLong(option);
        if !(0..=ALL_OPTS as i64).contains(&v) {
            PyErr_Clear();
            raise_type_error("Invalid opts");
            return ptr::null_mut();
        }
        opts = v as u32;
        for &(bit, name) in UNSUPPORTED_OPTS {
            if opts & bit != 0 {
                raise_type_error(&format!("isojson does not support {name}"));
                return ptr::null_mut();
            }
        }
    }

    // FR-13: resolve the datetime group first, so `guard` knows whether a
    // `utcoffset()` could run during the walk.
    let cache = &raw mut (*state(module)).types;
    let dt = match (*cache).datetime() {
        Ok(dt) => dt,
        Err(PyErrSet) => return ptr::null_mut(),
    };
    let guard = !default.is_null() || opts & OPT_SERIALIZE_NUMPY != 0 || dt.is_some();

    let Some(out) = Out::new() else {
        return ptr::null_mut();
    };
    let mut enc = Encoder {
        out,
        default,
        opts,
        depth: 0,
        default_depth: 0,
        cache,
        guard,
        dt,
        np_rechecked: false,
    };
    if !enc.serialize(obj) {
        return ptr::null_mut(); // `enc.out` releases the partial bytes
    }
    if opts & OPT_APPEND_NEWLINE != 0 {
        enc.out.push(b'\n');
    }
    enc.out.finish()
}
