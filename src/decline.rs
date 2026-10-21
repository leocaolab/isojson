//! The decline rule (design FR-7): which numpy objects isojson won't write,
//! and what it says about them. Pure: no Python, no output buffer.
//!
//! `classify` applies checks (a)–(d) in orjson's order. The caller builds a
//! `Decline` from the rule it names plus the raw evidence it reads (`dtype.str`,
//! flags, shape, value); `decline_message` is the error text (orjson's where
//! orjson has one) and `decline_note` the `add_note` text with that evidence.

use crate::datetime::{Dt64Unit, TimeUnit};

/// `NPY_ARRAY_C_CONTIGUOUS`.
pub(crate) const C_CONTIGUOUS: i32 = 0x1;
/// `NPY_ARRAY_NOTSWAPPED`.
pub(crate) const NOTSWAPPED: i32 = 0x200;

/// The reason an array is declined, decided from its interface alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArrayDecline {
    NotContiguous,
    NotNative,
    ZeroDim,
    UnsupportedDtype,
}

/// The element types isojson writes, in the order a scalar's type is looked up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ElementType {
    F64,
    F32,
    F16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Dt64,
}

impl ElementType {
    /// Lists every element type; `ALL[i] as usize == i`, checked at compile time.
    pub(crate) const ALL: [ElementType; 13] = [
        ElementType::F64,
        ElementType::F32,
        ElementType::F16,
        ElementType::I64,
        ElementType::I32,
        ElementType::I16,
        ElementType::I8,
        ElementType::U64,
        ElementType::U32,
        ElementType::U16,
        ElementType::U8,
        ElementType::Bool,
        ElementType::Dt64,
    ];
}

const _: () = {
    let mut i = 0;
    while i < ElementType::ALL.len() {
        assert!(ElementType::ALL[i] as usize == i);
        i += 1;
    }
};

/// Why isojson won't write a numpy object, with the raw evidence it was
/// decided on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Decline {
    NotContiguous { flags: i32, shape: Vec<isize> },
    NotNative { dtype: Vec<u8> },
    ZeroDim { dtype: Vec<u8> },
    UnsupportedDtype { dtype: Vec<u8> },
    GenericUnit { v: i64 },
    UnparseableDtype { dtype: Vec<u8> },
    Unrepresentable { v: i64, unit: Dt64Unit },
}

/// Returns the element type, or the first failing check in orjson's order
/// (`array.rs:69-100`). numpy never reports `nd < 0`; it is treated
/// as 0-d so that no walk indexes a negative shape.
pub(crate) fn classify(
    flags: i32,
    nd: i32,
    kind: u8,
    itemsize: i32,
) -> Result<ElementType, ArrayDecline> {
    if flags & C_CONTIGUOUS == 0 {
        return Err(ArrayDecline::NotContiguous);
    }
    if flags & NOTSWAPPED == 0 {
        return Err(ArrayDecline::NotNative);
    }
    if nd <= 0 {
        return Err(ArrayDecline::ZeroDim);
    }
    Ok(match (kind, itemsize) {
        (b'b', 1) => ElementType::Bool,
        (b'f', 2) => ElementType::F16,
        (b'f', 4) => ElementType::F32,
        (b'f', 8) => ElementType::F64,
        (b'i', 1) => ElementType::I8,
        (b'i', 2) => ElementType::I16,
        (b'i', 4) => ElementType::I32,
        (b'i', 8) => ElementType::I64,
        (b'u', 1) => ElementType::U8,
        (b'u', 2) => ElementType::U16,
        (b'u', 4) => ElementType::U32,
        (b'u', 8) => ElementType::U64,
        (b'M', 8) => ElementType::Dt64,
        _ => return Err(ArrayDecline::UnsupportedDtype),
    })
}

/// Returns orjson's word for a unit in its datetime64 messages.
fn unit_word(base: TimeUnit) -> &'static str {
    match base {
        TimeUnit::Year => "years",
        TimeUnit::Month => "months",
        TimeUnit::Week => "weeks",
        TimeUnit::Day => "days",
        TimeUnit::Hour => "hours",
        TimeUnit::Minute => "minutes",
        TimeUnit::Second => "seconds",
        TimeUnit::Milli => "milliseconds",
        TimeUnit::Micro => "microseconds",
        TimeUnit::Nano => "nanoseconds",
        TimeUnit::Pico => "picoseconds",
        TimeUnit::Femto => "femtoseconds",
        TimeUnit::Atto => "attoseconds",
        TimeUnit::Generic => "generic",
    }
}

/// Returns the error text raised when no `default` takes the object.
pub(crate) fn decline_message(r: &Decline) -> String {
    match r {
        Decline::NotContiguous { .. } => {
            "numpy array is not C contiguous; use ndarray.tolist() in default".into()
        }
        Decline::NotNative { .. } => "numpy array is not native-endianness".into(),
        Decline::ZeroDim { .. } | Decline::UnsupportedDtype { .. } => {
            "unsupported datatype in numpy array".into()
        }
        Decline::GenericUnit { .. } => "unsupported numpy.datetime64 unit: generic".into(),
        Decline::UnparseableDtype { dtype } => format!(
            "unsupported numpy.datetime64 dtype: {}",
            String::from_utf8_lossy(dtype)
        ),
        Decline::Unrepresentable { v, unit } => {
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

/// Returns the `add_note` text: the message plus the raw evidence it was decided on.
pub(crate) fn decline_note(r: &Decline) -> String {
    let dtype = |d: &[u8]| format!("dtype.str='{}'", String::from_utf8_lossy(d));
    let evidence = match r {
        Decline::NotContiguous { flags, shape } => {
            let dims: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            let tuple = match dims.len() {
                1 => format!("({},)", dims[0]),
                _ => format!("({})", dims.join(", ")),
            };
            Some(format!("flags={flags:#x}, shape={tuple}"))
        }
        Decline::NotNative { dtype: d } | Decline::UnsupportedDtype { dtype: d } => Some(dtype(d)),
        Decline::ZeroDim { dtype: d } => Some(format!("0-d array, {}", dtype(d))),
        Decline::GenericUnit { v } => Some(format!(
            "value={v}; a generic-unit datetime64 has no unit, so its value has no meaning"
        )),
        // the message already quotes dtype.str
        Decline::UnparseableDtype { .. } => None,
        Decline::Unrepresentable { .. } => {
            Some("outside 0000-01-01T00:00:00 … 9999-12-31T23:59:59.999999".into())
        }
    };
    let mut note = format!(
        "isojson can't write this numpy object natively: {}",
        decline_message(r)
    );
    if let Some(e) = evidence {
        note.push_str(&format!(" ({e})"));
    }
    note
}

#[cfg(test)]
mod tests;
