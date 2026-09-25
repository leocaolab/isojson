//! The decline rule (design FR-7): which numpy objects isojson won't write,
//! and what it says about them. Pure: no Python, no output buffer.
//!
//! `classify` applies checks (a)–(d) in orjson's order. The caller builds a
//! `Reason` from the rule it names plus the raw evidence it reads (`dtype.str`,
//! flags, shape, value); `reason_message` is the error text (orjson's where
//! orjson has one) and `reason_note` the `add_note` text with that evidence.

use crate::datetime::{Dt64Unit, Unit};

/// `NPY_ARRAY_C_CONTIGUOUS`.
pub(crate) const C_CONTIGUOUS: i32 = 0x1;
/// `NPY_ARRAY_NOTSWAPPED`.
pub(crate) const NOTSWAPPED: i32 = 0x200;

/// FR-7 (a)–(d), the checks made on the array interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Check {
    NotContiguous,
    NotNative,
    ZeroDim,
    Dtype,
}

/// The element types isojson writes: the dtype table's single home.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elem {
    Bool,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Dt64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    NotContiguous { flags: i32, shape: Vec<isize> },
    NotNative { dtype: Vec<u8> },
    ZeroDim { dtype: Vec<u8> },
    Dtype { dtype: Vec<u8> },
    GenericValue { v: i64 },
    UnparseableDtype { dtype: Vec<u8> },
    Unrepresentable { v: i64, unit: Dt64Unit },
}

/// FR-7 (a)–(d) in orjson's order (`array.rs:69-100`), first match wins;
/// otherwise the element type. numpy never reports `nd < 0`; it is treated
/// as 0-d so that no walk indexes a negative shape.
pub(crate) fn classify(flags: i32, nd: i32, kind: u8, itemsize: i32) -> Result<Elem, Check> {
    if flags & C_CONTIGUOUS == 0 {
        return Err(Check::NotContiguous);
    }
    if flags & NOTSWAPPED == 0 {
        return Err(Check::NotNative);
    }
    if nd <= 0 {
        return Err(Check::ZeroDim);
    }
    Ok(match (kind, itemsize) {
        (b'b', 1) => Elem::Bool,
        (b'f', 2) => Elem::F16,
        (b'f', 4) => Elem::F32,
        (b'f', 8) => Elem::F64,
        (b'i', 1) => Elem::I8,
        (b'i', 2) => Elem::I16,
        (b'i', 4) => Elem::I32,
        (b'i', 8) => Elem::I64,
        (b'u', 1) => Elem::U8,
        (b'u', 2) => Elem::U16,
        (b'u', 4) => Elem::U32,
        (b'u', 8) => Elem::U64,
        (b'M', 8) => Elem::Dt64,
        _ => return Err(Check::Dtype),
    })
}

/// orjson's word for a unit in its datetime64 messages.
fn unit_word(base: Unit) -> &'static str {
    match base {
        Unit::Year => "years",
        Unit::Month => "months",
        Unit::Week => "weeks",
        Unit::Day => "days",
        Unit::Hour => "hours",
        Unit::Minute => "minutes",
        Unit::Second => "seconds",
        Unit::Milli => "milliseconds",
        Unit::Micro => "microseconds",
        Unit::Nano => "nanoseconds",
        Unit::Pico => "picoseconds",
        Unit::Femto => "femtoseconds",
        Unit::Atto => "attoseconds",
        Unit::Generic => "generic",
    }
}

/// The error text when no `default` takes the object (FR-7).
pub(crate) fn reason_message(r: &Reason) -> String {
    match r {
        Reason::NotContiguous { .. } => {
            "numpy array is not C contiguous; use ndarray.tolist() in default".into()
        }
        Reason::NotNative { .. } => "numpy array is not native-endianness".into(),
        Reason::ZeroDim { .. } | Reason::Dtype { .. } => {
            "unsupported datatype in numpy array".into()
        }
        Reason::GenericValue { .. } => "unsupported numpy.datetime64 unit: generic".into(),
        Reason::UnparseableDtype { dtype } => format!(
            "unsupported numpy.datetime64 dtype: {}",
            String::from_utf8_lossy(dtype)
        ),
        Reason::Unrepresentable { v, unit } => {
            let mut m = format!(
                "unrepresentable numpy.datetime64: {v} {}",
                unit_word(unit.base)
            );
            if unit.mult != 1 {
                m.push_str(&format!(" × {}", unit.mult));
            }
            m
        }
    }
}

/// The `add_note` text: the message plus the raw evidence it was decided on.
pub(crate) fn reason_note(r: &Reason) -> String {
    let dtype = |d: &[u8]| format!("dtype.str='{}'", String::from_utf8_lossy(d));
    let evidence = match r {
        Reason::NotContiguous { flags, shape } => {
            let dims: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            let tuple = match dims.len() {
                1 => format!("({},)", dims[0]),
                _ => format!("({})", dims.join(", ")),
            };
            Some(format!("flags={flags:#x}, shape={tuple}"))
        }
        Reason::NotNative { dtype: d } | Reason::Dtype { dtype: d } => Some(dtype(d)),
        Reason::ZeroDim { dtype: d } => Some(format!("0-d array, {}", dtype(d))),
        Reason::GenericValue { v } => Some(format!(
            "value={v}; a generic-unit datetime64 has no unit, so its value has no meaning"
        )),
        // the message already quotes dtype.str
        Reason::UnparseableDtype { .. } => None,
        Reason::Unrepresentable { .. } => {
            Some("outside 0000-01-01T00:00:00 … 9999-12-31T23:59:59.999999".into())
        }
    };
    let mut note = format!(
        "isojson can't write this numpy object natively: {}",
        reason_message(r)
    );
    if let Some(e) = evidence {
        note.push_str(&format!(" ({e})"));
    }
    note
}

#[cfg(test)]
mod tests;
