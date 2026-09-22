//! `loads`: JSON -> Python objects, built directly (no intermediate tree).
//!
//! The parser is iterative: nesting lives on an explicit heap stack, not the
//! C stack, so a 1024-deep document is safe even on threads with small stacks.
//!
//! Multi-interpreter safety: every object is created by the calling
//! interpreter for the calling interpreter. Dict keys go through a
//! `KeyCache`, which lives in the module's *per-interpreter* state — a
//! process-global key cache is exactly the cross-interpreter sharing this
//! crate exists to avoid.

use core::ptr;
use pyo3_ffi::*;

use crate::state;
use crate::swar;

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
        return Err(Error::Python);
    }
    ptr::copy_nonoverlapping(b.as_ptr(), PyUnicode_DATA(o).cast::<u8>(), b.len());
    Ok(o)
}

/// orjson: 1024 nested containers ok, 1025 fails.
const MAX_DEPTH: usize = 1024;

enum Error {
    /// JSON syntax error: message + byte offset into the input.
    Syntax(String, usize),
    /// A Python exception is already set (e.g. MemoryError).
    Python,
}

type R<T> = Result<T, Error>;

fn syntax<T>(msg: impl Into<String>, pos: usize) -> R<T> {
    Err(Error::Syntax(msg.into(), pos))
}

enum Frame {
    /// A list whose items are `Stack::items[start..]` (owned); the list
    /// object itself is built on `]`, sized exactly.
    List(usize),
    /// dict + the key awaiting its value (owned)
    Dict(*mut PyObject, *mut PyObject),
}

/// Owns every partially built container; releases them on error.
/// List items for all open lists share one `items` vector (one allocation
/// reused across the whole document instead of one per list).
struct Stack(Vec<Frame>, Vec<*mut PyObject>);

impl Drop for Stack {
    fn drop(&mut self) {
        for o in self.1.drain(..) {
            unsafe { Py_DECREF(o) };
        }
        for f in self.0.drain(..) {
            unsafe {
                match f {
                    Frame::List(_) => {}
                    Frame::Dict(d, k) => {
                        Py_DECREF(d);
                        if !k.is_null() {
                            Py_DECREF(k);
                        }
                    }
                }
            }
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    scratch: Vec<u8>,
    keys: *mut KeyCache,
}

/// Where a scanned string's unescaped bytes are.
enum Str {
    /// `s[a..b]` of the input (no escapes)
    Input(usize, usize),
    /// `scratch`
    Scratch,
}

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[inline]
unsafe fn new_ref(p: *mut PyObject) -> R<*mut PyObject> {
    if p.is_null() {
        Err(Error::Python)
    } else {
        Ok(p)
    }
}

impl<'a> Parser<'a> {
    #[inline]
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && is_ws(self.s[self.i]) {
            self.i += 1;
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    unsafe fn parse(&mut self) -> R<*mut PyObject> {
        let mut stack = Stack(Vec::new(), Vec::new());
        self.skip_ws();
        if self.i >= self.s.len() {
            return syntax("Input is a zero-length, empty document", 0);
        }
        'value: loop {
            self.skip_ws();
            let start = self.i;
            let mut val = match self.peek() {
                Some(b'{') => {
                    if stack.0.len() >= MAX_DEPTH {
                        return syntax("depth limit exceeded", start);
                    }
                    self.i += 1;
                    let d = new_ref(PyDict_New())?;
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                        d
                    } else {
                        stack.0.push(Frame::Dict(d, ptr::null_mut()));
                        let k = self.key()?;
                        if let Some(Frame::Dict(_, slot)) = stack.0.last_mut() {
                            *slot = k;
                        }
                        continue 'value;
                    }
                }
                Some(b'[') => {
                    if stack.0.len() >= MAX_DEPTH {
                        return syntax("depth limit exceeded", start);
                    }
                    self.i += 1;
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        self.i += 1;
                        new_ref(PyList_New(0))?
                    } else {
                        stack.0.push(Frame::List(stack.1.len()));
                        continue 'value;
                    }
                }
                Some(b'"') => self.string()?,
                Some(b'-' | b'0'..=b'9') => self.number()?,
                Some(b't') => self.literal(b"true", Py_True())?,
                Some(b'f') => self.literal(b"false", Py_False())?,
                Some(b'n') => self.literal(b"null", Py_None())?,
                Some(b']' | b'}')
                    if self.s[..start].iter().rev().find(|&&b| !is_ws(b)) == Some(&b',') =>
                {
                    return syntax("trailing comma is not allowed", start)
                }
                Some(_) => return syntax("unexpected character, expected a JSON value", start),
                None => return syntax("unexpected end of data, expected a JSON value", start),
            };

            // Attach the completed value to its parent, closing as many
            // containers as the input closes.
            loop {
                match stack.0.last_mut() {
                    None => {
                        self.skip_ws();
                        if self.i != self.s.len() {
                            Py_DECREF(val);
                            return syntax("unexpected content after document", self.i);
                        }
                        return Ok(val);
                    }
                    Some(Frame::List(_)) => {
                        stack.1.push(val);
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.i += 1;
                                // (a `]` right here is reported as a trailing
                                // comma by the value parser)
                                continue 'value;
                            }
                            Some(b']') => {
                                self.i += 1;
                                let Some(Frame::List(start)) = stack.0.pop() else {
                                    unreachable!()
                                };
                                val = build_list(&mut stack.1, start)?;
                            }
                            _ => return syntax("expected ',' or ']'", self.i),
                        }
                    }
                    Some(Frame::Dict(d, k)) => {
                        let rc = PyDict_SetItem(*d, *k, val);
                        Py_DECREF(val);
                        Py_DECREF(*k);
                        *k = ptr::null_mut();
                        if rc < 0 {
                            return Err(Error::Python);
                        }
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.i += 1;
                                self.skip_ws();
                                if self.peek() == Some(b'}') {
                                    return syntax("trailing comma is not allowed", self.i);
                                }
                                let nk = self.key()?;
                                if let Some(Frame::Dict(_, slot)) = stack.0.last_mut() {
                                    *slot = nk;
                                }
                                continue 'value;
                            }
                            Some(b'}') => {
                                self.i += 1;
                                let Some(Frame::Dict(d, _)) = stack.0.pop() else {
                                    unreachable!()
                                };
                                val = d;
                            }
                            _ => return syntax("expected ',' or '}'", self.i),
                        }
                    }
                }
            }
        }
    }

    /// `"key"` followed by `:`; leaves the cursor on the value.
    unsafe fn key(&mut self) -> R<*mut PyObject> {
        self.skip_ws();
        if self.peek() != Some(b'"') {
            return syntax("key must be a string", self.i);
        }
        let open = self.i;
        let where_ = self.scan_string()?;
        let bytes: &[u8] = match where_ {
            Str::Input(a, b) => &self.s[a..b],
            Str::Scratch => &self.scratch,
        };
        let k = if bytes.len() <= CACHE_MAX_KEY && !self.keys.is_null() && swar::is_ascii(bytes) {
            (*self.keys).get(bytes)?
        } else {
            make_str(bytes, open + 1)?
        };
        self.skip_ws();
        if self.peek() != Some(b':') {
            Py_DECREF(k);
            return syntax("expected ':'", self.i);
        }
        self.i += 1;
        Ok(k)
    }

    unsafe fn literal(&mut self, word: &[u8], obj: *mut PyObject) -> R<*mut PyObject> {
        if self.s[self.i..].starts_with(word) {
            self.i += word.len();
            Py_INCREF(obj);
            Ok(obj)
        } else {
            syntax("unexpected character, expected a JSON value", self.i)
        }
    }

    unsafe fn string(&mut self) -> R<*mut PyObject> {
        let open = self.i;
        match self.scan_string()? {
            Str::Input(a, b) => make_str(&self.s[a..b], open + 1),
            Str::Scratch => make_str(&self.scratch, open + 1),
        }
    }

    /// Scan a string starting at the opening quote; leave the cursor after
    /// the closing quote.
    fn scan_string(&mut self) -> R<Str> {
        let s = self.s;
        let open = self.i;
        let start = open + 1;
        let mut j = match swar::find_special(s, start) {
            None => return syntax("unexpected end of data in string", open),
            Some(j) => j,
        };
        match s[j] {
            b'"' => {
                self.i = j + 1;
                return Ok(Str::Input(start, j));
            }
            b'\\' => {}
            _ => return syntax("control character in string", j),
        }
        // Slow path: unescape into scratch.
        let buf = &mut self.scratch;
        buf.clear();
        buf.extend_from_slice(&s[start..j]);
        loop {
            match s.get(j) {
                None => return syntax("unexpected end of data in string", open),
                Some(b'"') => {
                    self.i = j + 1;
                    return Ok(Str::Scratch);
                }
                Some(b'\\') => {
                    let esc = j;
                    let c = match s.get(j + 1) {
                        Some(&c) => c,
                        None => return syntax("unexpected end of data in string", esc),
                    };
                    j += 2;
                    match c {
                        b'"' => buf.push(b'"'),
                        b'\\' => buf.push(b'\\'),
                        b'/' => buf.push(b'/'),
                        b'b' => buf.push(0x08),
                        b'f' => buf.push(0x0c),
                        b'n' => buf.push(b'\n'),
                        b'r' => buf.push(b'\r'),
                        b't' => buf.push(b'\t'),
                        b'u' => {
                            let hi = hex4(s, j)
                                .ok_or(Error::Syntax("invalid \\u escape".into(), esc))?;
                            j += 4;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if s.get(j) != Some(&b'\\') || s.get(j + 1) != Some(&b'u') {
                                    return syntax("no low surrogate in string", esc);
                                }
                                let lo = hex4(s, j + 2)
                                    .ok_or(Error::Syntax("invalid \\u escape".into(), j))?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return syntax("no low surrogate in string", esc);
                                }
                                j += 6;
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return syntax("lone trailing surrogate in string", esc);
                            } else {
                                hi
                            };
                            let ch = char::from_u32(cp)
                                .ok_or(Error::Syntax("invalid \\u escape".into(), esc))?;
                            let mut tmp = [0u8; 4];
                            buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                        }
                        _ => return syntax("invalid escape", esc),
                    }
                }
                Some(&b) if b < 0x20 => return syntax("control character in string", j),
                Some(_) => {
                    // copy the run up to the next special byte in one go
                    let next = swar::find_special(s, j).unwrap_or(s.len());
                    buf.extend_from_slice(&s[j..next]);
                    j = next;
                }
            }
        }
    }

    unsafe fn number(&mut self) -> R<*mut PyObject> {
        let s = self.s;
        let st = self.i;
        let mut j = st;
        let neg = s[j] == b'-';
        if neg {
            j += 1;
        }
        match s.get(j) {
            Some(b'0') => j += 1,
            Some(b'1'..=b'9') => {
                while matches!(s.get(j), Some(b'0'..=b'9')) {
                    j += 1;
                }
            }
            _ => return syntax("invalid number", st),
        }
        let mut is_float = false;
        if s.get(j) == Some(&b'.') {
            j += 1;
            if !matches!(s.get(j), Some(b'0'..=b'9')) {
                return syntax("invalid number", st);
            }
            while matches!(s.get(j), Some(b'0'..=b'9')) {
                j += 1;
            }
            is_float = true;
        }
        if matches!(s.get(j), Some(b'e' | b'E')) {
            j += 1;
            if matches!(s.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if !matches!(s.get(j), Some(b'0'..=b'9')) {
                return syntax("invalid number", st);
            }
            while matches!(s.get(j), Some(b'0'..=b'9')) {
                j += 1;
            }
            is_float = true;
        }
        self.i = j;
        // SAFETY: the grammar above admits only ASCII.
        let text = core::str::from_utf8_unchecked(&s[st..j]);
        if !is_float {
            // Up to 18 digits always fits in i64: accumulate directly.
            let digits = if neg { &s[st + 1..j] } else { &s[st..j] };
            if digits.len() <= 18 {
                let mut v: i64 = 0;
                for &b in digits {
                    v = v * 10 + (b - b'0') as i64;
                }
                return new_ref(PyLong_FromLongLong(if neg { -v } else { v }));
            }
            if let Ok(v) = text.parse::<i64>() {
                return new_ref(PyLong_FromLongLong(v));
            }
            if let Ok(v) = text.parse::<u64>() {
                return new_ref(PyLong_FromUnsignedLongLong(v));
            }
            // Beyond 64 bits: orjson returns a float, and so do we.
        }
        match fast_float2::parse::<f64, _>(text) {
            Ok(v) if v.is_finite() => new_ref(PyFloat_FromDouble(v)),
            _ => syntax("number is infinity when parsed as double", st),
        }
    }
}

unsafe fn make_str(bytes: &[u8], pos: usize) -> R<*mut PyObject> {
    if swar::is_ascii(bytes) {
        return new_ascii(bytes);
    }
    let o = PyUnicode_FromStringAndSize(bytes.as_ptr().cast(), bytes.len() as Py_ssize_t);
    if o.is_null() {
        if PyErr_ExceptionMatches(PyExc_UnicodeDecodeError) != 0 {
            PyErr_Clear();
            return syntax("str is not valid UTF-8: surrogates not allowed", pos);
        }
        return Err(Error::Python);
    }
    Ok(o)
}

/// Move `items[start..]` into a new exactly-sized list.
unsafe fn build_list(items: &mut Vec<*mut PyObject>, start: usize) -> R<*mut PyObject> {
    let n = items.len() - start;
    let l = PyList_New(n as Py_ssize_t);
    if l.is_null() {
        return Err(Error::Python); // items stay owned by the stack, released on drop
    }
    for (i, &o) in items[start..].iter().enumerate() {
        PyList_SET_ITEM(l, i as Py_ssize_t, o); // steals
    }
    items.truncate(start);
    Ok(l)
}

fn hex4(s: &[u8], at: usize) -> Option<u32> {
    let h = s.get(at..at + 4)?;
    let mut v = 0u32;
    for &b in h {
        v = v * 16
            + match b {
                b'0'..=b'9' => (b - b'0') as u32,
                b'a'..=b'f' => (b - b'a' + 10) as u32,
                b'A'..=b'F' => (b - b'A' + 10) as u32,
                _ => return None,
            };
    }
    Some(v)
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
    let mut p = Parser {
        s: input,
        i: 0,
        scratch: Vec::new(),
        keys: (*state(module)).key_cache,
    };
    match p.parse() {
        Ok(o) => o,
        Err(Error::Python) => ptr::null_mut(),
        Err(Error::Syntax(msg, pos)) => {
            raise_decode_error(module, &msg, input, doc, pos);
            ptr::null_mut()
        }
    }
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
