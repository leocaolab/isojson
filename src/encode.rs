//! `dumps`: Python object -> JSON bytes, written in a single pass.
//!
//! Multi-interpreter safety: the encoder only *borrows* objects for the
//! duration of one call, on the calling thread, under the calling
//! interpreter's GIL (items are additionally INCREF'd while a user `default`
//! callback could run). Nothing is cached across calls except a thread-local
//! byte buffer, which holds no Python objects.

use core::ffi::CStr;
use core::ptr;
use pyo3_ffi::*;

use crate::float::write_f64;
use crate::out::Out;
use crate::*;

/// Nested containers at or beyond this depth fail (orjson: 254 ok, 255 fails).
const MAX_DEPTH: u32 = 254;
/// Chained `default` calls allowed before giving up (orjson: 255).
const MAX_DEFAULT_DEPTH: u32 = 255;

const ALL_OPTS: u32 = 4095;
/// Options that change *how* a type we serialize natively is written, and
/// that we do not implement. Silently ignoring them would produce output the
/// caller did not ask for, so they are rejected.
const UNSUPPORTED_OPTS: &[(u32, &str)] = &[
    (OPT_NON_STR_KEYS, "OPT_NON_STR_KEYS"),
    (OPT_SERIALIZE_NUMPY, "OPT_SERIALIZE_NUMPY"),
];
// OPT_NAIVE_UTC / OPT_OMIT_MICROSECONDS / OPT_UTC_Z / OPT_PASSTHROUGH_DATETIME /
// OPT_PASSTHROUGH_DATACLASS only affect datetime and dataclass objects, which
// isojson never serializes natively (they always go to `default`), so they
// are accepted and have no effect.

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
unsafe fn raise_type_error_from_current(msg: &str) {
    let cause = PyErr_GetRaisedException();
    raise_type_error(msg);
    if !cause.is_null() {
        let exc = PyErr_GetRaisedException();
        PyException_SetCause(exc, cause); // steals `cause`
        PyErr_SetRaisedException(exc); // steals `exc`
    }
}

struct Encoder {
    out: Out,
    default: *mut PyObject,
    opts: u32,
    depth: u32,
    default_depth: u32,
}

impl Encoder {
    #[inline]
    fn indent(&mut self) -> bool {
        self.opts & OPT_INDENT_2 != 0
    }

    #[inline]
    fn newline_indent(&mut self, level: u32) {
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
        self.call_default(obj)
    }

    /// Serialize a borrowed container item. Without a `default` callback no
    /// Python code can run during serialization, so the container cannot be
    /// mutated and the borrow is safe as is. With one, `default` could mutate
    /// the container and drop the item, so hold a reference across the call.
    #[inline]
    unsafe fn guarded(&mut self, item: *mut PyObject) -> bool {
        if self.default.is_null() {
            return self.serialize(item);
        }
        Py_INCREF(item);
        let ok = self.serialize(item);
        Py_DECREF(item);
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
        if self.default_depth >= MAX_DEFAULT_DEPTH {
            raise_type_error("default serializer exceeds recursion limit");
            return false;
        }
        let r = PyObject_CallOneArg(self.default, obj);
        if r.is_null() {
            raise_type_error_from_current(&format!(
                "Type is not JSON serializable: {}",
                type_name(obj)
            ));
            return false;
        }
        self.default_depth += 1;
        let ok = self.serialize(r);
        self.default_depth -= 1;
        Py_DECREF(r);
        ok
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
            let mut b = itoa::Buffer::new();
            crate::float::small_copy(&mut self.out, b.format(v).as_bytes());
            return true;
        }
        if overflow > 0 && self.opts & OPT_STRICT_INTEGER == 0 {
            let u = PyLong_AsUnsignedLongLong(obj);
            if u != u64::MAX || PyErr_Occurred().is_null() {
                let mut b = itoa::Buffer::new();
                crate::float::small_copy(&mut self.out, b.format(u).as_bytes());
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
            if self.default.is_null() {
                if !self.serialize(value) {
                    return false;
                }
            } else {
                // `default` may run arbitrary Python and mutate this dict;
                // keep the key (whose UTF-8 we already wrote) and value alive.
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
    _module: *mut PyObject,
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

    let Some(out) = Out::new() else {
        return ptr::null_mut();
    };
    let mut enc = Encoder {
        out,
        default,
        opts,
        depth: 0,
        default_depth: 0,
    };
    if !enc.serialize(obj) {
        return ptr::null_mut(); // `enc.out` releases the partial bytes
    }
    if opts & OPT_APPEND_NEWLINE != 0 {
        enc.out.push(b'\n');
    }
    enc.out.finish()
}
