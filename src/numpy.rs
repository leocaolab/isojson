//! Writes numpy arrays and scalars.
//!
//! Arrays are read through `__array_struct__` (PyArrayInterface v2): no numpy
//! headers, no `import_array`, no numpy C-API global. The capsule is held for
//! the walk. Leaves are read with `ptr::read_unaligned`, so unaligned
//! C-contiguous arrays are read correctly.
//!
//! Whatever isojson can't write is *declined* (FR-7): `NumpyError::Declined`
//! carries a `Decline` with the raw evidence, and `Encoder::call_default_declined`
//! handles it in one place. An element declined mid-array first rolls the
//! output back to the array's start, so `default` sees the whole array.

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use pyo3_ffi::*;

use crate::datetime::{dt64_to_parts, parse_dt64, Dt64Error, Dt64Unit};
use crate::decline::{classify, ArrayDecline, Decline, ElementType};
use crate::encode::{write_datetime, write_int, Encoder};
use crate::float::{f16_to_f32, write_f32, write_f64};
use crate::out::OutputBuffer;
use crate::{PendingPyErr, PyResult};

/// numpy's `PyArrayInterface` (`__array_struct__`, version 2).
#[repr(C)]
struct PyArrayInterface {
    two: c_int,
    nd: c_int,
    typekind: c_char,
    itemsize: c_int,
    flags: c_int,
    shape: *mut Py_intptr_t,
    strides: *mut Py_intptr_t,
    data: *mut c_void,
    descr: *mut PyObject,
}

/// Why a numpy object wasn't written.
pub(crate) enum NumpyError {
    /// isojson won't write it; the encoder hands the whole object to `default`.
    Declined(Decline),
    Py(PendingPyErr),
}

impl From<PendingPyErr> for NumpyError {
    fn from(e: PendingPyErr) -> Self {
        NumpyError::Py(e)
    }
}

impl From<Decline> for NumpyError {
    fn from(d: Decline) -> Self {
        NumpyError::Declined(d)
    }
}

/// Holds `obj.__array_struct__` and releases it on drop.
struct ArrayStruct {
    capsule: *mut PyObject,
    iface: *const PyArrayInterface,
}

impl Drop for ArrayStruct {
    fn drop(&mut self) {
        unsafe { Py_DECREF(self.capsule) };
    }
}

unsafe fn array_struct(enc: &mut Encoder, obj: *mut PyObject) -> PyResult<ArrayStruct> {
    debug_assert!(enc.guard);
    let capsule = PyObject_GetAttr(obj, (*enc.cache).names.array_struct);
    if capsule.is_null() {
        return Err(PendingPyErr);
    }
    let iface = PyCapsule_GetPointer(capsule, ptr::null()).cast::<PyArrayInterface>();
    if iface.is_null() || (*iface).two != 2 {
        Py_DECREF(capsule);
        crate::encode::raise_type_error_from_current("numpy array is malformed");
        return Err(PendingPyErr);
    }
    Ok(ArrayStruct { capsule, iface })
}

/// Returns the raw bytes of `obj.dtype.str`, which are both the decline
/// evidence and the datetime64 unit.
unsafe fn dtype_str(enc: &mut Encoder, obj: *mut PyObject) -> PyResult<Vec<u8>> {
    debug_assert!(enc.guard);
    let names = &(*enc.cache).names;
    let dtype = PyObject_GetAttr(obj, names.dtype);
    if dtype.is_null() {
        return Err(PendingPyErr);
    }
    let s = PyObject_GetAttr(dtype, names.str);
    Py_DECREF(dtype);
    if s.is_null() {
        return Err(PendingPyErr);
    }
    let mut n: Py_ssize_t = 0;
    let p = PyUnicode_AsUTF8AndSize(s, &mut n);
    let out = if p.is_null() {
        Err(PendingPyErr)
    } else {
        Ok(core::slice::from_raw_parts(p.cast::<u8>(), n as usize).to_vec())
    };
    Py_DECREF(s);
    out
}

#[inline(always)]
unsafe fn rd<T>(p: *const u8) -> T {
    ptr::read_unaligned(p.cast::<T>())
}

/// Writes one datetime64 value (`null` for NaT), or returns why it can't.
#[inline]
fn write_dt64(enc: &mut Encoder, v: i64, unit: Dt64Unit) -> Result<(), Decline> {
    match dt64_to_parts(v, unit) {
        Ok(None) => {
            enc.out.extend_from_slice(b"null");
            Ok(())
        }
        Ok(Some(p)) => {
            let offset = (enc.opts & crate::OPT_NAIVE_UTC != 0).then_some(0);
            write_datetime(&mut enc.out, &p, offset, enc.opts);
            Ok(())
        }
        Err(Dt64Error::GenericUnit) => Err(Decline::GenericUnit { v }),
        Err(Dt64Error::Unrepresentable) => Err(Decline::Unrepresentable { v, unit }),
    }
}

#[inline]
fn write_bool(enc: &mut Encoder, b: u8) {
    if b != 0 {
        enc.out.extend_from_slice(b"true");
    } else {
        enc.out.extend_from_slice(b"false");
    }
}

/// Expands to one closure per element type that writes the element at `p`,
/// so each walk is monomorphic. `unit` is the datetime64 unit; it is `None` only for
/// an element-less array, whose walk visits no leaf.
macro_rules! leaf {
    ($elem:expr, $unit:expr, $body:ident) => {
        match $elem {
            ElementType::Bool => $body!(|e: &mut Encoder, p| {
                write_bool(e, rd::<u8>(p));
                Ok(())
            }),
            ElementType::F16 => $body!(|e: &mut Encoder, p| {
                write_f32(&mut e.out, f16_to_f32(rd::<u16>(p)));
                Ok(())
            }),
            ElementType::F32 => $body!(|e: &mut Encoder, p| {
                write_f32(&mut e.out, rd::<f32>(p));
                Ok(())
            }),
            ElementType::F64 => $body!(|e: &mut Encoder, p| {
                write_f64(&mut e.out, rd::<f64>(p));
                Ok(())
            }),
            ElementType::I8 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i8>(p));
                Ok(())
            }),
            ElementType::I16 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i16>(p));
                Ok(())
            }),
            ElementType::I32 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i32>(p));
                Ok(())
            }),
            ElementType::I64 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i64>(p));
                Ok(())
            }),
            ElementType::U8 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u8>(p));
                Ok(())
            }),
            ElementType::U16 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u16>(p));
                Ok(())
            }),
            ElementType::U32 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u32>(p));
                Ok(())
            }),
            ElementType::U64 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u64>(p));
                Ok(())
            }),
            ElementType::Dt64 => match $unit {
                Some(unit) => $body!(|e: &mut Encoder, p| write_dt64(e, rd::<i64>(p), unit)),
                None => $body!(|_: &mut Encoder, _| Ok(())),
            },
        }
    };
}

/// Elements per capacity reservation in a compact row.
const ROW_CHUNK: isize = 1024;

/// Writes a whole compact row `[a,b,…]` of `n` elements `stride` apart.
type RowFn = unsafe fn(&mut OutputBuffer, *const u8, isize, isize);

/// Writes a compact integer row through a raw pointer after one reserve, with
/// no per-element capacity check or `Result`; integers can't be declined.
unsafe fn write_int_row<T: itoa::Integer>(
    out: &mut OutputBuffer,
    p: *const u8,
    n: isize,
    stride: isize,
) {
    // the longest integer text is 20 bytes, plus a comma, plus the brackets
    out.reserve(n as usize * 21 + 2);
    let base = out.as_mut_ptr();
    let mut dst = base.add(out.len());
    *dst = b'[';
    dst = dst.add(1);
    let mut buf = itoa::Buffer::new();
    for i in 0..n {
        if i > 0 {
            *dst = b',';
            dst = dst.add(1);
        }
        let s = buf.format(rd::<T>(p.offset(i * stride)));
        dst = copy_digits(s.as_bytes(), dst);
    }
    *dst = b']';
    dst = dst.add(1);
    out.set_len(dst.offset_from(base) as usize);
}

/// Copies `s` (at most 20 bytes) to `dst` with fixed-size, possibly
/// overlapping copies, as `small_copy` does, and return the end. Never reads
/// or writes outside `s` / `dst[..s.len()]`.
#[inline(always)]
unsafe fn copy_digits(s: &[u8], dst: *mut u8) -> *mut u8 {
    let n = s.len();
    let src = s.as_ptr();
    if n >= 8 {
        ptr::copy_nonoverlapping(src, dst, 8);
        ptr::copy_nonoverlapping(src.add(n - 8), dst.add(n - 8), 8);
        if n > 16 {
            ptr::copy_nonoverlapping(src.add(8), dst.add(8), 8);
        }
    } else if n >= 4 {
        ptr::copy_nonoverlapping(src, dst, 4);
        ptr::copy_nonoverlapping(src.add(n - 4), dst.add(n - 4), 4);
    } else {
        for i in 0..n {
            *dst.add(i) = *src.add(i);
        }
    }
    dst.add(n)
}

fn int_row(elem: ElementType) -> Option<RowFn> {
    Some(match elem {
        ElementType::I8 => write_int_row::<i8>,
        ElementType::I16 => write_int_row::<i16>,
        ElementType::I32 => write_int_row::<i32>,
        ElementType::I64 => write_int_row::<i64>,
        ElementType::U8 => write_int_row::<u8>,
        ElementType::U16 => write_int_row::<u16>,
        ElementType::U32 => write_int_row::<u32>,
        ElementType::U64 => write_int_row::<u64>,
        _ => return None,
    })
}

/// Returns the longest text, in bytes, that one element of `elem` can have.
fn max_text(elem: ElementType) -> usize {
    match elem {
        ElementType::Bool => 5,
        ElementType::I8 | ElementType::U8 => 4,
        ElementType::I16 | ElementType::U16 => 6,
        ElementType::I32 | ElementType::U32 => 11,
        ElementType::I64 | ElementType::U64 => 20,
        ElementType::F16 | ElementType::F32 => 16,
        ElementType::F64 => 24,
        // "YYYY-MM-DDTHH:MM:SS.ffffff+00:00" quoted
        ElementType::Dt64 => 34,
    }
}

/// Writes nested dims as nested lists and a 0-length dim as `[]` with no
/// indent; under `INDENT_2` each dim is indented like a nested list (the
/// encoder's depth plus the dim). Depth is at most `nd` (numpy caps it at 64)
/// and doesn't count toward `MAX_DEPTH`.
///
/// # Safety
/// `a.shape` and `a.strides` must hold `a.nd` entries and `p` must point at
/// the data they describe, with `dim < a.nd`.
unsafe fn walk<L>(
    enc: &mut Encoder,
    a: &PyArrayInterface,
    row_bytes: usize,
    fast_row: Option<RowFn>,
    dim: usize,
    p: *const u8,
    leaf: &mut L,
) -> Result<(), Decline>
where
    L: FnMut(&mut Encoder, *const u8) -> Result<(), Decline>,
{
    let n = *a.shape.add(dim);
    if n == 0 {
        enc.out.extend_from_slice(b"[]");
        return Ok(());
    }
    let stride = *a.strides.add(dim);
    let last = dim + 1 == a.nd as usize;
    let indent = enc.indent();
    let level = enc.depth + dim as u32;
    if last && !indent {
        if let Some(row) = fast_row {
            row(&mut enc.out, p, n, stride);
            return Ok(());
        }
    }
    enc.out.push(b'[');
    for i in 0..n {
        // one reserve per chunk of a compact row: the leaves' own capacity
        // checks never grow it, and a very long row doesn't allocate its
        // whole worst case up front
        if last && !indent && i % ROW_CHUNK == 0 {
            enc.out
                .reserve((n - i).min(ROW_CHUNK) as usize * row_bytes + 1);
        }
        if i > 0 {
            enc.out.push(b',');
        }
        if indent {
            enc.newline_indent(level + 1);
        }
        let q = p.offset(i * stride);
        if last {
            leaf(enc, q)?;
        } else {
            walk(enc, a, row_bytes, fast_row, dim + 1, q, leaf)?;
        }
    }
    if indent {
        enc.newline_indent(level);
    }
    enc.out.push(b']');
    Ok(())
}

/// Returns a datetime64 dtype's unit, or a decline if its dtype string doesn't parse.
unsafe fn dt64_unit(enc: &mut Encoder, obj: *mut PyObject) -> Result<Dt64Unit, NumpyError> {
    let dtype = dtype_str(enc, obj)?;
    Ok(parse_dt64(&dtype).ok_or(Decline::UnparseableDtype { dtype })?)
}

/// Writes an exact `ndarray`. On an element decline it first rolls the output
/// back, so `default` sees the whole array.
pub(crate) unsafe fn serialize_array(
    enc: &mut Encoder,
    obj: *mut PyObject,
) -> Result<(), NumpyError> {
    let held = array_struct(enc, obj)?;
    let a = &*held.iface;
    let elem = match classify(a.flags, a.nd, a.typekind as u8, a.itemsize) {
        Ok(elem) => elem,
        Err(check) => {
            let reason = match check {
                ArrayDecline::NotContiguous => Decline::NotContiguous {
                    flags: a.flags,
                    shape: core::slice::from_raw_parts(a.shape, a.nd.max(0) as usize).to_vec(),
                },
                _ => {
                    let dtype = dtype_str(enc, obj)?;
                    match check {
                        ArrayDecline::NotNative => Decline::NotNative { dtype },
                        ArrayDecline::ZeroDim => Decline::ZeroDim { dtype },
                        _ => Decline::UnsupportedDtype { dtype },
                    }
                }
            };
            return Err(reason.into());
        }
    };
    let shape = core::slice::from_raw_parts(a.shape, a.nd as usize);
    // element-less arrays are walked without reading their unit, as orjson
    // writes them (`[]`, `[[],[]]`) whatever the dtype string says
    let unit = if elem == ElementType::Dt64 && !shape.contains(&0) {
        Some(dt64_unit(enc, obj)?)
    } else {
        None
    };
    let start = enc.out.len();
    let data = a.data as *const u8;
    macro_rules! run {
        ($leaf:expr) => {{
            let mut leaf = $leaf;
            walk(
                enc,
                a,
                max_text(elem) + 1,
                int_row(elem),
                0,
                data,
                &mut leaf,
            )
        }};
    }
    let r = leaf!(elem, unit, run);
    drop(held);
    r.map_err(|reason| {
        enc.out.set_len(start);
        reason.into()
    })
}

/// Writes an exact numpy scalar whose element type is `elem`.
pub(crate) unsafe fn serialize_scalar(
    enc: &mut Encoder,
    obj: *mut PyObject,
    elem: ElementType,
) -> Result<(), NumpyError> {
    let unit = if elem == ElementType::Dt64 {
        Some(dt64_unit(enc, obj)?)
    } else {
        None
    };
    // the value is read from the scalar's layout, right after the
    // object header (numpy's `Py<Type>ScalarObject { PyObject_HEAD; obval }`),
    // as orjson does; measured several times faster than a capsule per scalar
    let data = obj
        .cast::<u8>()
        .add(core::mem::size_of::<PyObject>())
        .cast_const();
    macro_rules! one {
        ($leaf:expr) => {{
            let leaf = $leaf;
            leaf(enc, data)
        }};
    }
    Ok(leaf!(elem, unit, one)?)
}
