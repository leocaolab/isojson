//! Date/time text and calendar math. Pure: no Python, no output buffer.
//!
//! The formatters fill caller-owned fixed-size arrays and return the length;
//! `encode.rs` and `numpy.rs` copy the bytes into `Out`. Every piece is at
//! most 16 bytes, within `small_copy`'s limit.
//!
//! The text is `isoformat()`'s (design §1a, §8.2). `datetime64` values are
//! turned into calendar parts by the meaning numpy's API defines: `v × mult`
//! units since 1970-01-01T00:00, sub-µs units floored to µs, months by floor
//! division, all arithmetic checked (design FR-9, §8.3).

use crate::{OPT_OMIT_MICROSECONDS, OPT_UTC_Z};

/// Days from 1970-01-01 to 0000-01-01 and to 9999-12-31: the range
/// `civil_from_days` is proven over.
const MIN_DAYS: i64 = -719_528;
const MAX_DAYS: i64 = 2_932_896;

const US_PER_DAY: i128 = 86_400_000_000;

#[inline(always)]
fn two(buf: &mut [u8], at: usize, v: u32) {
    buf[at] = b'0' + (v / 10 % 10) as u8;
    buf[at + 1] = b'0' + (v % 10) as u8;
}

#[inline(always)]
fn six(buf: &mut [u8], at: usize, v: u32) {
    let mut v = v;
    for i in (0..6).rev() {
        buf[at + i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
}

/// `YYYY-MM-DD`, the year zero-padded to four digits as `isoformat()` does.
pub(crate) fn fmt_ymd(buf: &mut [u8; 10], y: u16, mo: u8, d: u8) -> usize {
    debug_assert!(y <= 9999);
    two(buf, 0, y as u32 / 100);
    two(buf, 2, y as u32 % 100);
    buf[4] = b'-';
    two(buf, 5, mo as u32);
    buf[7] = b'-';
    two(buf, 8, d as u32);
    10
}

/// `HH:MM:SS[.ffffff]`: the fraction only when `us` is non-zero and
/// `OMIT_MICROSECONDS` is not set.
pub(crate) fn fmt_hms(buf: &mut [u8; 15], h: u8, mi: u8, s: u8, us: u32, opts: u32) -> usize {
    two(buf, 0, h as u32);
    buf[2] = b':';
    two(buf, 3, mi as u32);
    buf[5] = b':';
    two(buf, 6, s as u32);
    if us == 0 || opts & OPT_OMIT_MICROSECONDS != 0 {
        return 8;
    }
    buf[8] = b'.';
    six(buf, 9, us);
    15
}

/// `isoformat()`'s offset text for a UTC offset of `total_us` (design §8.2):
/// `±HH:MM[:SS[.ffffff]]`, or `+00:00` / `Z` (`UTC_Z`) for zero. The caller
/// passes an offset CPython validated, so `|total_us|` is under 24 hours.
pub(crate) fn fmt_offset(buf: &mut [u8; 16], total_us: i64, opts: u32) -> usize {
    if total_us == 0 {
        if opts & OPT_UTC_Z != 0 {
            buf[0] = b'Z';
            return 1;
        }
        buf[..6].copy_from_slice(b"+00:00");
        return 6;
    }
    debug_assert!(total_us.unsigned_abs() < 86_400_000_000);
    buf[0] = if total_us < 0 { b'-' } else { b'+' };
    let m = total_us.unsigned_abs();
    let (hh, r) = (m / 3_600_000_000, m % 3_600_000_000);
    let (mm, r) = (r / 60_000_000, r % 60_000_000);
    let (ss, us) = (r / 1_000_000, r % 1_000_000);
    two(buf, 1, hh as u32);
    buf[3] = b':';
    two(buf, 4, mm as u32);
    if ss == 0 && us == 0 {
        return 6;
    }
    buf[6] = b':';
    two(buf, 7, ss as u32);
    if us == 0 {
        return 9;
    }
    buf[9] = b'.';
    six(buf, 10, us as u32);
    16
}

/// The proleptic Gregorian date `days` after 1970-01-01 (Howard Hinnant's
/// `civil_from_days`). Proven exhaustively over `MIN_DAYS..=MAX_DAYS`, the
/// range the caller checks first.
pub(crate) fn civil_from_days(days: i64) -> (i32, u8, u8) {
    debug_assert!((MIN_DAYS..=MAX_DAYS).contains(&days));
    let z = days + 719_468; // days since 0000-03-01
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], from March 1
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (mo <= 2) as i64;
    (y as i32, mo as u8, d as u8)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unit {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Milli,
    Micro,
    Nano,
    Pico,
    Femto,
    Atto,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dt64Unit {
    pub(crate) base: Unit,
    pub(crate) mult: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Parts {
    pub(crate) y: u16,
    pub(crate) mo: u8,
    pub(crate) d: u8,
    pub(crate) h: u8,
    pub(crate) mi: u8,
    pub(crate) s: u8,
    pub(crate) us: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dt64Err {
    /// Overflow, or a year outside 0000–9999: FR-7 (f).
    Unrepresentable,
    /// A generic-unit value that is not NaT: FR-7 (e).
    GenericValue,
}

/// The unit and multiplier of a `datetime64` dtype from its `dtype.str`:
/// a byte-order character, `M8`, then nothing (generic) or `[<mult><unit>]`.
/// `None` for anything else (FR-7 (e)).
pub(crate) fn parse_dt64(s: &[u8]) -> Option<Dt64Unit> {
    let rest = match s {
        [b'<' | b'>' | b'=' | b'|', b'M', b'8', rest @ ..] => rest,
        _ => return None,
    };
    if rest.is_empty() {
        return Some(Dt64Unit {
            base: Unit::Generic,
            mult: 1,
        });
    }
    let inner = rest.strip_prefix(b"[")?.strip_suffix(b"]")?;
    let digits = inner.iter().take_while(|c| c.is_ascii_digit()).count();
    let (num, code) = inner.split_at(digits);
    let mult = if num.is_empty() {
        1
    } else {
        let mut m: i64 = 0;
        for &c in num {
            m = m.checked_mul(10)?.checked_add((c - b'0') as i64)?;
        }
        if m == 0 {
            return None;
        }
        m
    };
    let base = match code {
        b"Y" => Unit::Year,
        b"M" => Unit::Month,
        b"W" => Unit::Week,
        b"D" => Unit::Day,
        b"h" => Unit::Hour,
        b"m" => Unit::Minute,
        b"s" => Unit::Second,
        b"ms" => Unit::Milli,
        b"us" => Unit::Micro,
        b"ns" => Unit::Nano,
        b"ps" => Unit::Pico,
        b"fs" => Unit::Femto,
        b"as" => Unit::Atto,
        _ => return None,
    };
    Some(Dt64Unit { base, mult })
}

/// The calendar parts of the `datetime64` value `v` in unit `u` (design §8.3):
/// `Ok(None)` for NaT, `Err` when the value has no representation.
pub(crate) fn dt64_to_parts(v: i64, u: Dt64Unit) -> Result<Option<Parts>, Dt64Err> {
    if v == i64::MIN {
        return Ok(None);
    }
    let date_only = |y: i128, mo: i128| {
        if (0..=9999).contains(&y) {
            Ok(Some(Parts {
                y: y as u16,
                mo: mo as u8,
                d: 1,
                h: 0,
                mi: 0,
                s: 0,
                us: 0,
            }))
        } else {
            Err(Dt64Err::Unrepresentable)
        }
    };
    let n = (v as i128)
        .checked_mul(u.mult as i128)
        .ok_or(Dt64Err::Unrepresentable)?;
    let times = |k: i128| n.checked_mul(k).ok_or(Dt64Err::Unrepresentable);
    let us = match u.base {
        Unit::Generic => return Err(Dt64Err::GenericValue),
        Unit::Year => return date_only(1970 + n, 1),
        Unit::Month => return date_only(1970 + n.div_euclid(12), n.rem_euclid(12) + 1),
        Unit::Week => times(7 * US_PER_DAY)?,
        Unit::Day => times(US_PER_DAY)?,
        Unit::Hour => times(3_600_000_000)?,
        Unit::Minute => times(60_000_000)?,
        Unit::Second => times(1_000_000)?,
        Unit::Milli => times(1_000)?,
        Unit::Micro => n,
        // sub-µs units are floored to µs (§1a)
        Unit::Nano => n.div_euclid(1_000),
        Unit::Pico => n.div_euclid(1_000_000),
        Unit::Femto => n.div_euclid(1_000_000_000),
        Unit::Atto => n.div_euclid(1_000_000_000_000),
    };
    let days = us.div_euclid(US_PER_DAY);
    if !(MIN_DAYS as i128..=MAX_DAYS as i128).contains(&days) {
        return Err(Dt64Err::Unrepresentable);
    }
    let (y, mo, d) = civil_from_days(days as i64);
    let rem = us.rem_euclid(US_PER_DAY) as u64;
    Ok(Some(Parts {
        y: y as u16,
        mo,
        d,
        h: (rem / 3_600_000_000) as u8,
        mi: (rem / 60_000_000 % 60) as u8,
        s: (rem / 1_000_000 % 60) as u8,
        us: (rem % 1_000_000) as u32,
    }))
}

#[cfg(test)]
mod tests;
