//! `loads`: JSON -> Python objects.
//!
//! Parsing is done by simd-json (a pure-Rust port of simdjson), which turns
//! the document into a flat "tape" of nodes where every array and object
//! carries its element count. isojson then walks the tape and builds Python
//! objects: lists are created at their exact size, and nesting is handled
//! with an explicit stack rather than recursion.
//!
//! Multi-interpreter safety: simd-json never sees a Python object; its work
//! buffers are thread-local plain bytes. Every Python object is created by the
//! calling interpreter for the calling interpreter. Dict keys go through a
//! `KeyCache` that lives in the module's *per-interpreter* state — a
//! process-global key cache is exactly the cross-interpreter sharing this
//! crate exists to avoid.

use core::cell::RefCell;
use core::ptr;
use pyo3_ffi::*;
use simd_json::{Buffers, ErrorType, Node, StaticNode};

use crate::swar;
use crate::{state, PyErrSet, R};

thread_local! {
    /// Per-thread parser scratch: a mutable copy of the input (simd-json
    /// unescapes strings in place, and Python `bytes` are immutable) and
    /// simd-json's work buffers. Plain bytes only.
    static SCRATCH: RefCell<(Vec<u8>, Buffers)> = RefCell::new((Vec::new(), Buffers::new(0)));
}

const CACHE_SLOTS: usize = 2048;
const CACHE_MAX_KEY: usize = 64;

/// Direct-mapped cache of recently seen ASCII dict keys, one per interpreter.
/// A hit returns the same `str` object (hash already computed), which saves
/// both the allocation and the hashing in `PyDict_SetItem`.
pub(crate) struct KeyCache {
    slots: Box<[(u64, *mut PyObject)]>,
}

impl KeyCache {
    pub(crate) fn new() -> Box<Self> {
        Box::new(KeyCache {
            slots: vec![(0u64, ptr::null_mut()); CACHE_SLOTS].into_boxed_slice(),
        })
    }

    /// Release every cached key. Must run while the owning interpreter is
    /// alive and holds its GIL (module clear/free satisfy both).
    pub(crate) unsafe fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            if !slot.1.is_null() {
                let o = slot.1;
                slot.1 = ptr::null_mut();
                Py_DECREF(o);
            }
        }
    }

    #[inline]
    fn hash(b: &[u8]) -> u64 {
        let mut h: u64 = b.len() as u64;
        let mut i = 0;
        while i + 8 <= b.len() {
            h = (h.rotate_left(5) ^ swar::load(b, i)).wrapping_mul(0x517c_c1b7_2722_0a95);
            i += 8;
        }
        let mut tail = [0u8; 8];
        tail[..b.len() - i].copy_from_slice(&b[i..]);
        (h.rotate_left(5) ^ u64::from_le_bytes(tail)).wrapping_mul(0x517c_c1b7_2722_0a95)
    }

    /// New reference to a `str` equal to `b` (which the caller has checked is
    /// ASCII and at most CACHE_MAX_KEY bytes).
    #[inline]
    unsafe fn get(&mut self, b: &[u8]) -> R<*mut PyObject> {
        let h = Self::hash(b);
        let slot = &mut self.slots[(h as usize) & (CACHE_SLOTS - 1)];
        if slot.0 == h && !slot.1.is_null() {
            let o = slot.1;
            let n = PyUnicode_GET_LENGTH(o) as usize;
            if n == b.len() && core::slice::from_raw_parts(PyUnicode_DATA(o).cast::<u8>(), n) == b {
                Py_INCREF(o);
                return Ok(o);
            }
        }
        let o = new_ascii(b)?;
        // Pre-compute the hash so every later hit skips it.
        PyObject_Hash(o);
        Py_INCREF(o); // one for the cache, one for the caller
        if !slot.1.is_null() {
            Py_DECREF(slot.1);
        }
        *slot = (h, o);
        Ok(o)
    }
}

#[inline]
unsafe fn new_ascii(b: &[u8]) -> R<*mut PyObject> {
    let o = PyUnicode_New(b.len() as Py_ssize_t, 127);
    if o.is_null() {
        return Err(PyErrSet);
    }
    ptr::copy_nonoverlapping(b.as_ptr(), PyUnicode_DATA(o).cast::<u8>(), b.len());
    Ok(o)
}

/// orjson: 1024 nested containers ok, 1025 fails.
#[inline]
unsafe fn new_ref(p: *mut PyObject) -> R<*mut PyObject> {
    if p.is_null() {
        Err(PyErrSet)
    } else {
        Ok(p)
    }
}

/// `str` from UTF-8 that simd-json has already validated.
#[inline]
unsafe fn make_str(b: &[u8]) -> R<*mut PyObject> {
    if swar::is_ascii(b) {
        return new_ascii(b);
    }
    new_ref(PyUnicode_FromStringAndSize(
        b.as_ptr().cast(),
        b.len() as Py_ssize_t,
    ))
}

/// A container being filled: `obj` owns the items inserted so far.
struct Open {
    obj: *mut PyObject,
    is_list: bool,
    /// next list index / dict entries inserted so far
    filled: usize,
    len: usize,
    /// dict only: key awaiting its value (owned), or null
    key: *mut PyObject,
}

/// Releases every partially built container if building fails midway.
/// (A list with unfilled NULL slots is safe to release.)
struct OpenStack(Vec<Open>);

impl Drop for OpenStack {
    fn drop(&mut self) {
        for o in self.0.drain(..) {
            unsafe {
                if !o.key.is_null() {
                    Py_DECREF(o.key);
                }
                Py_DECREF(o.obj);
            }
        }
    }
}

/// Build the Python value for `nodes` (one complete JSON document).
unsafe fn build(nodes: &[Node<'_>], keys: *mut KeyCache) -> R<*mut PyObject> {
    let mut stack = OpenStack(Vec::new());
    let mut i = 0;
    loop {
        // In an object, the next node is a key.
        if let Some(top) = stack.0.last_mut() {
            if !top.is_list && top.key.is_null() {
                let Node::String(k) = *nodes.get_unchecked(i) else {
                    unreachable!("simd-json object keys are strings")
                };
                i += 1;
                let k = k.as_bytes();
                top.key = if k.len() <= CACHE_MAX_KEY && !keys.is_null() && swar::is_ascii(k) {
                    (*keys).get(k)?
                } else {
                    make_str(k)?
                };
            }
        }

        let node = *nodes.get_unchecked(i);
        i += 1;
        let mut val = match node {
            Node::String(s) => make_str(s.as_bytes())?,
            Node::Static(StaticNode::Null) => {
                Py_INCREF(Py_None());
                Py_None()
            }
            Node::Static(StaticNode::Bool(b)) => {
                let o = if b { Py_True() } else { Py_False() };
                Py_INCREF(o);
                o
            }
            Node::Static(StaticNode::I64(v)) => new_ref(PyLong_FromLongLong(v))?,
            Node::Static(StaticNode::U64(v)) => new_ref(PyLong_FromUnsignedLongLong(v))?,
            Node::Static(StaticNode::F64(v)) => new_ref(PyFloat_FromDouble(v))?,
            Node::Array { len, .. } => {
                let l = new_ref(PyList_New(len as Py_ssize_t))?;
                if len > 0 {
                    stack.0.push(Open {
                        obj: l,
                        is_list: true,
                        filled: 0,
                        len,
                        key: ptr::null_mut(),
                    });
                    continue;
                }
                l
            }
            Node::Object { len, .. } => {
                let d = new_ref(PyDict_New())?;
                if len > 0 {
                    stack.0.push(Open {
                        obj: d,
                        is_list: false,
                        filled: 0,
                        len,
                        key: ptr::null_mut(),
                    });
                    continue;
                }
                d
            }
        };

        // Insert the finished value into its parent, closing every
        // container this completes.
        loop {
            let Some(top) = stack.0.last_mut() else {
                return Ok(val);
            };
            if top.is_list {
                PyList_SET_ITEM(top.obj, top.filled as Py_ssize_t, val); // steals
            } else {
                let rc = PyDict_SetItem(top.obj, top.key, val);
                Py_DECREF(val);
                Py_DECREF(top.key);
                top.key = ptr::null_mut();
                if rc < 0 {
                    return Err(PyErrSet);
                }
            }
            top.filled += 1;
            if top.filled < top.len {
                break;
            }
            let done = stack.0.pop().unwrap();
            val = done.obj;
        }
    }
}

/// Human-readable message for a simd-json error (never its internal
/// variant name).
fn describe(e: &ErrorType) -> String {
    match e {
        ErrorType::Eof => "unexpected end of data".into(),
        ErrorType::TrailingData => "unexpected content after document".into(),
        ErrorType::UnterminatedString => "unterminated string".into(),
        ErrorType::InvalidEscape => "invalid escape sequence in string".into(),
        ErrorType::InvalidUnicodeEscape => "invalid \\u escape in string".into(),
        ErrorType::InvalidUnicodeCodepoint => {
            "invalid unicode code point (lone surrogate?) in string".into()
        }
        ErrorType::InvalidUtf8 => "input is not valid UTF-8".into(),
        ErrorType::InvalidNumber => "invalid number".into(),
        ErrorType::InvalidExponent => "invalid number exponent".into(),
        ErrorType::DepthLimitExceeded => "depth limit exceeded".into(),
        ErrorType::ExpectedArrayComma => "expected ',' or ']' in array".into(),
        ErrorType::ExpectedMapComma => "expected ',' or '}' in object".into(),
        ErrorType::ExpectedObjectColon => "expected ':' after object key".into(),
        ErrorType::ExpectedObjectKey | ErrorType::KeyMustBeAString | ErrorType::BadKeyType => {
            "object key must be a string".into()
        }
        ErrorType::ExpectedMapEnd => "expected '}' at end of object".into(),
        ErrorType::ExpectedArrayContent | ErrorType::ExpectedObjectContent => {
            "unexpected character, expected a JSON value".into()
        }
        ErrorType::ExpectedNull | ErrorType::ExpectedTrue | ErrorType::ExpectedFalse => {
            "unexpected character, expected a JSON value".into()
        }
        ErrorType::UnexpectedCharacter | ErrorType::NoStructure | ErrorType::Syntax => {
            "unexpected character".into()
        }
        ErrorType::InputTooLarge => "input is too large (simd-json limit: 4 GiB)".into(),
        // Anything else is not expected from parsing to a tape; keep the
        // parser's own wording rather than inventing one.
        other => format!("invalid JSON ({other:?})"),
    }
}

/// Raise this interpreter's `isojson.JSONDecodeError(msg, doc, pos)`.
/// `pos` is converted from a byte offset to a character offset, which is
/// what `json.JSONDecodeError` expects.
unsafe fn raise_decode_error(
    module: *mut PyObject,
    msg: &str,
    input: &[u8],
    doc: *mut PyObject,
    byte_pos: usize,
) {
    let exc_type = (*state(module)).decode_error;
    let upto = &input[..byte_pos.min(input.len())];
    let char_pos = upto.iter().filter(|&&b| (b & 0xC0) != 0x80).count();

    let owned_doc = if doc.is_null() {
        PyUnicode_DecodeUTF8(
            input.as_ptr().cast(),
            input.len() as Py_ssize_t,
            c"replace".as_ptr(),
        )
    } else {
        Py_INCREF(doc);
        doc
    };
    if owned_doc.is_null() {
        return;
    }
    let py_msg = PyUnicode_FromStringAndSize(msg.as_ptr().cast(), msg.len() as Py_ssize_t);
    let py_pos = PyLong_FromSize_t(char_pos);
    if !py_msg.is_null() && !py_pos.is_null() {
        let args = [py_msg, owned_doc, py_pos];
        let exc = PyObject_Vectorcall(exc_type, args.as_ptr(), 3, ptr::null_mut());
        if !exc.is_null() {
            PyErr_SetObject(exc_type, exc);
            Py_DECREF(exc);
        }
    }
    if !py_msg.is_null() {
        Py_DECREF(py_msg);
    }
    if !py_pos.is_null() {
        Py_DECREF(py_pos);
    }
    Py_DECREF(owned_doc);
}

unsafe fn run(module: *mut PyObject, input: &[u8], doc: *mut PyObject) -> *mut PyObject {
    if input
        .iter()
        .all(|&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
    {
        raise_decode_error(
            module,
            "Input is a zero-length, empty document",
            input,
            doc,
            0,
        );
        return ptr::null_mut();
    }
    let keys = (*state(module)).key_cache;
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let (buf, buffers) = &mut *scratch;
        buf.clear();
        buf.extend_from_slice(input);
        match simd_json::to_tape_with_buffers(buf, buffers) {
            Err(e) => {
                let msg = describe(e.error());
                raise_decode_error(module, &msg, input, doc, e.index());
                ptr::null_mut()
            }
            Ok(tape) => match build(&tape.0, keys) {
                Ok(o) => o,
                Err(PyErrSet) => ptr::null_mut(),
            },
        }
    })
}

pub(crate) unsafe extern "C" fn loads(module: *mut PyObject, obj: *mut PyObject) -> *mut PyObject {
    if PyBytes_Check(obj) != 0 {
        let p = PyBytes_AsString(obj);
        let n = PyBytes_Size(obj);
        return run(
            module,
            core::slice::from_raw_parts(p.cast(), n as usize),
            ptr::null_mut(),
        );
    }
    if PyUnicode_Check(obj) != 0 {
        let mut n: Py_ssize_t = 0;
        let p = PyUnicode_AsUTF8AndSize(obj, &mut n);
        if p.is_null() {
            PyErr_Clear();
            raise_decode_error(
                module,
                "str is not valid UTF-8: surrogates not allowed",
                b"",
                obj,
                0,
            );
            return ptr::null_mut();
        }
        return run(
            module,
            core::slice::from_raw_parts(p.cast(), n as usize),
            obj,
        );
    }
    if PyByteArray_Check(obj) != 0 {
        // The parser never calls back into Python code, so the bytearray
        // cannot be resized underneath us.
        let p = PyByteArray_AsString(obj);
        let n = PyByteArray_Size(obj);
        return run(
            module,
            core::slice::from_raw_parts(p.cast(), n as usize),
            ptr::null_mut(),
        );
    }
    if PyMemoryView_Check(obj) != 0 {
        let mut view: Py_buffer = core::mem::zeroed();
        if PyObject_GetBuffer(obj, &mut view, PyBUF_C_CONTIGUOUS) < 0 {
            return ptr::null_mut();
        }
        let r = run(
            module,
            core::slice::from_raw_parts(view.buf.cast(), view.len as usize),
            ptr::null_mut(),
        );
        PyBuffer_Release(&mut view);
        return r;
    }
    raise_decode_error(
        module,
        "Input must be bytes, bytearray, memoryview, or str",
        b"",
        ptr::null_mut(),
        0,
    );
    ptr::null_mut()
}
