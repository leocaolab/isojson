//! Tests for the decline rule (design C6): `classify` in orjson's order, and
//! `reason_message` / `reason_note` for every `Reason`.

use super::*;

const OK_FLAGS: i32 = C_CONTIGUOUS | NOTSWAPPED;

const TABLE: &[(u8, i32, Elem)] = &[
    (b'b', 1, Elem::Bool),
    (b'f', 2, Elem::F16),
    (b'f', 4, Elem::F32),
    (b'f', 8, Elem::F64),
    (b'i', 1, Elem::I8),
    (b'i', 2, Elem::I16),
    (b'i', 4, Elem::I32),
    (b'i', 8, Elem::I64),
    (b'u', 1, Elem::U8),
    (b'u', 2, Elem::U16),
    (b'u', 4, Elem::U32),
    (b'u', 8, Elem::U64),
    (b'M', 8, Elem::Dt64),
];

#[test]
fn classify_accepts_exactly_the_table() {
    for kind in 0..=255u8 {
        for itemsize in [0, 1, 2, 3, 4, 8, 12, 16, 32] {
            let want = TABLE
                .iter()
                .find(|&&(k, n, _)| k == kind && n == itemsize)
                .map(|&(_, _, e)| e)
                .ok_or(Check::Dtype);
            for nd in [1, 2, 64] {
                assert_eq!(
                    classify(OK_FLAGS, nd, kind, itemsize),
                    want,
                    "{kind} {itemsize}"
                );
            }
        }
    }
    // numpy kinds that look close but are not written: complex, timedelta,
    // object, str, bytes, void, longdouble on Linux x86_64
    for (kind, itemsize) in [
        (b'c', 16),
        (b'm', 8),
        (b'O', 8),
        (b'U', 12),
        (b'S', 3),
        (b'V', 8),
        (b'f', 16),
    ] {
        assert_eq!(classify(OK_FLAGS, 1, kind, itemsize), Err(Check::Dtype));
    }
}

#[test]
fn classify_order_first_match_wins() {
    let bad = (b'c', 16);
    // (a) before everything
    assert_eq!(classify(0, 0, bad.0, bad.1), Err(Check::NotContiguous));
    assert_eq!(classify(NOTSWAPPED, 1, b'f', 8), Err(Check::NotContiguous));
    // (b) before (c) and (d)
    assert_eq!(
        classify(C_CONTIGUOUS, 0, bad.0, bad.1),
        Err(Check::NotNative)
    );
    assert_eq!(classify(C_CONTIGUOUS, 1, b'f', 8), Err(Check::NotNative));
    // (c) before (d)
    assert_eq!(classify(OK_FLAGS, 0, bad.0, bad.1), Err(Check::ZeroDim));
    assert_eq!(classify(OK_FLAGS, 0, b'f', 8), Err(Check::ZeroDim));
    // (d)
    assert_eq!(classify(OK_FLAGS, 1, bad.0, bad.1), Err(Check::Dtype));
}

#[test]
fn classify_ignores_other_flag_bits() {
    // ALIGNED 0x100, WRITEABLE 0x400, F_CONTIGUOUS 0x2, OWNDATA 0x4
    let extra = 0x100 | 0x400 | 0x2 | 0x4;
    assert_eq!(classify(OK_FLAGS | extra, 2, b'i', 8), Ok(Elem::I64));
    assert_eq!(
        classify(NOTSWAPPED | extra, 2, b'i', 8),
        Err(Check::NotContiguous)
    );
}

#[test]
fn classify_negative_nd_is_not_walked() {
    assert_eq!(classify(OK_FLAGS, -1, b'f', 8), Err(Check::ZeroDim));
}

fn u(base: Unit, mult: i64) -> Dt64Unit {
    Dt64Unit { base, mult }
}

#[test]
fn messages_match_fr7() {
    let cases: Vec<(Reason, &str)> = vec![
        (
            Reason::NotContiguous {
                flags: 0x502,
                shape: vec![3, 2],
            },
            "numpy array is not C contiguous; use ndarray.tolist() in default",
        ),
        (
            Reason::NotNative {
                dtype: b">f8".to_vec(),
            },
            "numpy array is not native-endianness",
        ),
        (
            Reason::ZeroDim {
                dtype: b"<f8".to_vec(),
            },
            "unsupported datatype in numpy array",
        ),
        (
            Reason::Dtype {
                dtype: b"<U3".to_vec(),
            },
            "unsupported datatype in numpy array",
        ),
        (
            Reason::GenericValue { v: 5 },
            "unsupported numpy.datetime64 unit: generic",
        ),
        (
            Reason::UnparseableDtype {
                dtype: b"<M8[xx]".to_vec(),
            },
            "unsupported numpy.datetime64 dtype: <M8[xx]",
        ),
        (
            Reason::Unrepresentable {
                v: 307_445_734_561_825_861,
                unit: u(Unit::Minute, 1),
            },
            "unrepresentable numpy.datetime64: 307445734561825861 minutes",
        ),
        (
            Reason::Unrepresentable {
                v: -5,
                unit: u(Unit::Milli, 10),
            },
            "unrepresentable numpy.datetime64: -5 milliseconds × 10",
        ),
    ];
    for (r, want) in cases {
        assert_eq!(reason_message(&r), want, "{r:?}");
    }
}

#[test]
fn unrepresentable_names_every_unit_as_orjson() {
    let words = [
        (Unit::Year, "years"),
        (Unit::Month, "months"),
        (Unit::Week, "weeks"),
        (Unit::Day, "days"),
        (Unit::Hour, "hours"),
        (Unit::Minute, "minutes"),
        (Unit::Second, "seconds"),
        (Unit::Milli, "milliseconds"),
        (Unit::Micro, "microseconds"),
        (Unit::Nano, "nanoseconds"),
        (Unit::Pico, "picoseconds"),
        (Unit::Femto, "femtoseconds"),
        (Unit::Atto, "attoseconds"),
        (Unit::Generic, "generic"),
    ];
    for (base, word) in words {
        let r = Reason::Unrepresentable {
            v: 1,
            unit: u(base, 1),
        };
        assert_eq!(
            reason_message(&r),
            format!("unrepresentable numpy.datetime64: 1 {word}")
        );
    }
}

#[test]
fn notes_carry_the_raw_evidence() {
    let cases: Vec<(Reason, &str)> = vec![
        (
            Reason::NotContiguous {
                flags: 0x502,
                shape: vec![3, 2],
            },
            "numpy array is not C contiguous; use ndarray.tolist() in default \
             (flags=0x502, shape=(3, 2))",
        ),
        (
            Reason::NotContiguous {
                flags: 0x0,
                shape: vec![7],
            },
            "numpy array is not C contiguous; use ndarray.tolist() in default \
             (flags=0x0, shape=(7,))",
        ),
        (
            Reason::NotNative {
                dtype: b">f8".to_vec(),
            },
            "numpy array is not native-endianness (dtype.str='>f8')",
        ),
        (
            Reason::ZeroDim {
                dtype: b"<f8".to_vec(),
            },
            "unsupported datatype in numpy array (0-d array, dtype.str='<f8')",
        ),
        (
            Reason::Dtype {
                dtype: b"<U3".to_vec(),
            },
            "unsupported datatype in numpy array (dtype.str='<U3')",
        ),
        (
            Reason::GenericValue { v: 5 },
            "unsupported numpy.datetime64 unit: generic (value=5; a generic-unit \
             datetime64 has no unit, so its value has no meaning)",
        ),
        (
            Reason::UnparseableDtype {
                dtype: b"<M8[xx]".to_vec(),
            },
            "unsupported numpy.datetime64 dtype: <M8[xx]",
        ),
        (
            Reason::Unrepresentable {
                v: -5,
                unit: u(Unit::Milli, 10),
            },
            "unrepresentable numpy.datetime64: -5 milliseconds × 10 (outside \
             0000-01-01T00:00:00 … 9999-12-31T23:59:59.999999)",
        ),
    ];
    for (r, want) in cases {
        assert_eq!(
            reason_note(&r),
            format!("isojson can't write this numpy object natively: {want}"),
            "{r:?}"
        );
    }
}

#[test]
fn non_utf8_dtype_is_shown_not_dropped() {
    let r = Reason::Dtype {
        dtype: vec![b'<', 0xff, b'8'],
    };
    assert!(reason_note(&r).contains("dtype.str='<\u{fffd}8'"));
}
