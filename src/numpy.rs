//! numpy arrays and scalars (design C4, FR-6…9).
//!
//! Arrays are read through `__array_struct__` (PyArrayInterface v2): no numpy
//! headers, no `import_array`, no numpy C-API global. The capsule is held for
//! the walk. Leaves are read with `ptr::read_unaligned`, so unaligned
//! C-contiguous arrays are read correctly.
//!
//! Whatever isojson can't write is *declined* (FR-7): `Outcome::Declined`
//! carries a `Reason` with the raw evidence, and `Encoder::call_default_declined`
//! handles it in one place. An element declined mid-array first rolls the
//! output back to the array's start, so `default` sees the whole array.

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use pyo3_ffi::*;

use crate::datetime::{dt64_to_parts, parse_dt64, Dt64Err, Dt64Unit};
use crate::decline::{classify, Check, Elem, Reason};
use crate::encode::{write_datetime, write_int, Encoder};
use crate::float::{f16_to_f32, write_f32, write_f64};
use crate::out::Out;
use crate::types::NumpyTypes;
use crate::{PyErrSet, R};

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

pub(crate) enum Outcome {
    Written,
    Declined(Reason),
    Error(PyErrSet),
}

/// `obj.__array_struct__`, held until dropped.
struct Interface {
    capsule: *mut PyObject,
    iface: *const PyArrayInterface,
}

impl Drop for Interface {
    fn drop(&mut self) {
        unsafe { Py_DECREF(self.capsule) };
    }
}

unsafe fn interface(enc: &mut Encoder, obj: *mut PyObject) -> R<Interface> {
    debug_assert!(enc.guard);
    let capsule = PyObject_GetAttr(obj, (*enc.cache).names.array_struct);
    if capsule.is_null() {
        return Err(PyErrSet);
    }
    let iface = PyCapsule_GetPointer(capsule, ptr::null()).cast::<PyArrayInterface>();
    if iface.is_null() || (*iface).two != 2 {
        Py_DECREF(capsule);
        crate::encode::raise_type_error_from_current("numpy array is malformed");
        return Err(PyErrSet);
    }
    Ok(Interface { capsule, iface })
}

/// `obj.dtype.str`, the raw bytes (the evidence FR-7 reports, and the
/// datetime64 unit).
unsafe fn dtype_str(enc: &mut Encoder, obj: *mut PyObject) -> R<Vec<u8>> {
    debug_assert!(enc.guard);
    let names = &(*enc.cache).names;
    let dtype = PyObject_GetAttr(obj, names.dtype);
    if dtype.is_null() {
        return Err(PyErrSet);
    }
    let s = PyObject_GetAttr(dtype, names.str);
    Py_DECREF(dtype);
    if s.is_null() {
        return Err(PyErrSet);
    }
    let mut n: Py_ssize_t = 0;
    let p = PyUnicode_AsUTF8AndSize(s, &mut n);
    let out = if p.is_null() {
        Err(PyErrSet)
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

/// One datetime64 value: `null` for NaT, the §1a text, or the FR-7 reason.
#[inline]
fn write_dt64(enc: &mut Encoder, v: i64, unit: Dt64Unit) -> Result<(), Reason> {
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
        Err(Dt64Err::GenericValue) => Err(Reason::GenericValue { v }),
        Err(Dt64Err::Unrepresentable) => Err(Reason::Unrepresentable { v, unit }),
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

/// Write one element of type `elem` at `p`, one closure per dtype so each
/// walk is monomorphic. `unit` is the datetime64 unit; it is `None` only for
/// an element-less array, whose walk visits no leaf.
macro_rules! leaf {
    ($elem:expr, $unit:expr, $body:ident) => {
        match $elem {
            Elem::Bool => $body!(|e: &mut Encoder, p| {
                write_bool(e, rd::<u8>(p));
                Ok(())
            }),
            Elem::F16 => $body!(|e: &mut Encoder, p| {
                write_f32(&mut e.out, f16_to_f32(rd::<u16>(p)));
                Ok(())
            }),
            Elem::F32 => $body!(|e: &mut Encoder, p| {
                write_f32(&mut e.out, rd::<f32>(p));
                Ok(())
            }),
            Elem::F64 => $body!(|e: &mut Encoder, p| {
                write_f64(&mut e.out, rd::<f64>(p));
                Ok(())
            }),
            Elem::I8 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i8>(p));
                Ok(())
            }),
            Elem::I16 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i16>(p));
                Ok(())
            }),
            Elem::I32 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i32>(p));
                Ok(())
            }),
            Elem::I64 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<i64>(p));
                Ok(())
            }),
            Elem::U8 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u8>(p));
                Ok(())
            }),
            Elem::U16 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u16>(p));
                Ok(())
            }),
            Elem::U32 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u32>(p));
                Ok(())
            }),
            Elem::U64 => $body!(|e: &mut Encoder, p| {
                write_int(&mut e.out, rd::<u64>(p));
                Ok(())
            }),
            Elem::Dt64 => match $unit {
                Some(unit) => $body!(|e: &mut Encoder, p| write_dt64(e, rd::<i64>(p), unit)),
                None => $body!(|_: &mut Encoder, _| Ok(())),
            },
        }
    };
}

/// Elements per capacity reservation in a compact row.
const ROW_CHUNK: isize = 1024;

/// Writes a whole compact row `[a,b,…]` of `n` elements `stride` apart.
type RowFn = unsafe fn(&mut Out, *const u8, isize, isize);

/// A compact integer row, written through a raw pointer after one reserve:
/// no per-element capacity check or `Result`. Integers can't be declined.
unsafe fn write_int_row<T: itoa::Integer>(out: &mut Out, p: *const u8, n: isize, stride: isize) {
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

/// Copy `s` (at most 20 bytes) to `dst` with fixed-size, possibly
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

fn int_row(elem: Elem) -> Option<RowFn> {
    Some(match elem {
        Elem::I8 => write_int_row::<i8>,
        Elem::I16 => write_int_row::<i16>,
        Elem::I32 => write_int_row::<i32>,
        Elem::I64 => write_int_row::<i64>,
        Elem::U8 => write_int_row::<u8>,
        Elem::U16 => write_int_row::<u16>,
        Elem::U32 => write_int_row::<u32>,
        Elem::U64 => write_int_row::<u64>,
        _ => return None,
    })
}

/// The longest text one element of `elem` can have.
fn max_text(elem: Elem) -> usize {
    match elem {
        Elem::Bool => 5,
        Elem::I8 | Elem::U8 => 4,
        Elem::I16 | Elem::U16 => 6,
        Elem::I32 | Elem::U32 => 11,
        Elem::I64 | Elem::U64 => 20,
        Elem::F16 | Elem::F32 => 16,
        Elem::F64 => 24,
        // "YYYY-MM-DDTHH:MM:SS.ffffff+00:00" quoted
        Elem::Dt64 => 34,
    }
}

/// The FR-6 walk: nested dims → nested lists, a 0-length dim → `[]` with no
/// indent, and under `INDENT_2` each dim indented like a nested list (the
/// encoder's depth + the dim). Depth is at most `nd` (numpy caps it at 64)
/// and doesn't count toward `MAX_DEPTH`.
unsafe fn walk<L>(
    enc: &mut Encoder,
    a: &PyArrayInterface,
    row_bytes: usize,
    fast_row: Option<RowFn>,
    dim: usize,
    p: *const u8,
    leaf: &mut L,
) -> Result<(), Reason>
where
    L: FnMut(&mut Encoder, *const u8) -> Result<(), Reason>,
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

/// A datetime64 dtype's unit, or the FR-7 (e) decline.
unsafe fn dt64_unit(enc: &mut Encoder, obj: *mut PyObject) -> Result<Dt64Unit, Outcome> {
    let dtype = dtype_str(enc, obj).map_err(Outcome::Error)?;
    parse_dt64(&dtype).ok_or(Outcome::Declined(Reason::UnparseableDtype { dtype }))
}

/// An exact `ndarray` (FR-6, FR-7). Rolls the output back before any
/// element decline.
pub(crate) unsafe fn serialize_array(enc: &mut Encoder, obj: *mut PyObject) -> Outcome {
    let held = match interface(enc, obj) {
        Ok(i) => i,
        Err(e) => return Outcome::Error(e),
    };
    let a = &*held.iface;
    let elem = match classify(a.flags, a.nd, a.typekind as u8, a.itemsize) {
        Ok(elem) => elem,
        Err(check) => {
            let reason = match check {
                Check::NotContiguous => Reason::NotContiguous {
                    flags: a.flags,
                    shape: core::slice::from_raw_parts(a.shape, a.nd.max(0) as usize).to_vec(),
                },
                _ => {
                    let dtype = match dtype_str(enc, obj) {
                        Ok(d) => d,
                        Err(e) => return Outcome::Error(e),
                    };
                    match check {
                        Check::NotNative => Reason::NotNative { dtype },
                        Check::ZeroDim => Reason::ZeroDim { dtype },
                        _ => Reason::Dtype { dtype },
                    }
                }
            };
            return Outcome::Declined(reason);
        }
    };
    let shape = core::slice::from_raw_parts(a.shape, a.nd as usize);
    // element-less arrays are walked without reading their unit, as orjson
    // writes them (`[]`, `[[],[]]`) whatever the dtype string says
    let unit = if elem == Elem::Dt64 && !shape.contains(&0) {
        match dt64_unit(enc, obj) {
            Ok(u) => Some(u),
            Err(o) => return o,
        }
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
    match r {
        Ok(()) => Outcome::Written,
        Err(reason) => {
            enc.out.set_len(start);
            Outcome::Declined(reason)
        }
    }
}

/// The FR-8 scalar types; `None` for anything else.
fn scalar_elem(np: &NumpyTypes, ty: *mut PyTypeObject) -> Option<Elem> {
    let table = [
        (np.float64, Elem::F64),
        (np.float32, Elem::F32),
        (np.float16, Elem::F16),
        (np.int64, Elem::I64),
        (np.int32, Elem::I32),
        (np.int16, Elem::I16),
        (np.int8, Elem::I8),
        (np.uint64, Elem::U64),
        (np.uint32, Elem::U32),
        (np.uint16, Elem::U16),
        (np.uint8, Elem::U8),
        (np.bool_, Elem::Bool),
        (np.datetime64, Elem::Dt64),
    ];
    table.into_iter().find(|&(t, _)| t == ty).map(|(_, e)| e)
}

/// An exact numpy scalar of an FR-8 type; `None` if `obj` is not one.
pub(crate) unsafe fn serialize_scalar(
    enc: &mut Encoder,
    obj: *mut PyObject,
    np: NumpyTypes,
) -> Option<Outcome> {
    let elem = scalar_elem(&np, Py_TYPE(obj))?;
    // `np` is not used past this point: the calls below run Python (C1)
    let unit = if elem == Elem::Dt64 {
        match dt64_unit(enc, obj) {
            Ok(u) => Some(u),
            Err(o) => return Some(o),
        }
    } else {
        None
    };
    // O-2: the value is read from the scalar's layout, right after the
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
    let r = leaf!(elem, unit, one);
    Some(match r {
        Ok(()) => Outcome::Written,
        Err(reason) => Outcome::Declined(reason),
    })
}
