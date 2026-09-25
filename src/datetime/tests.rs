//! Tests for the pure date/time core (design C6).
//!
//! The oracle for `dt64_to_parts` is `model`, an exact-integer rendering of
//! design §1a written independently of §8.3: total µs from a unit ratio table,
//! the date by walking years and months (not Hinnant's formula), and the
//! 0000–9999 range by comparing µs against the two walked boundaries.

use super::*;
use crate::{OPT_NAIVE_UTC, OPT_OMIT_MICROSECONDS, OPT_UTC_Z};

// ---- formatters ------------------------------------------------------------

fn ymd(y: u16, mo: u8, d: u8) -> String {
    let mut b = [0u8; 10];
    let n = fmt_ymd(&mut b, y, mo, d);
    String::from_utf8(b[..n].to_vec()).unwrap()
}

fn hms(h: u8, mi: u8, s: u8, us: u32, opts: u32) -> String {
    let mut b = [0u8; 15];
    let n = fmt_hms(&mut b, h, mi, s, us, opts);
    String::from_utf8(b[..n].to_vec()).unwrap()
}

fn off(total_us: i64, opts: u32) -> String {
    let mut b = [0u8; 16];
    let n = fmt_offset(&mut b, total_us, opts);
    String::from_utf8(b[..n].to_vec()).unwrap()
}

const H: i64 = 3_600_000_000;
const M: i64 = 60_000_000;
const S: i64 = 1_000_000;

#[test]
fn fmt_ymd_pads_every_field() {
    assert_eq!(ymd(0, 1, 1), "0000-01-01");
    assert_eq!(ymd(7, 3, 9), "0007-03-09");
    assert_eq!(ymd(999, 10, 10), "0999-10-10");
    assert_eq!(ymd(2026, 9, 24), "2026-09-24");
    assert_eq!(ymd(9999, 12, 31), "9999-12-31");
}

#[test]
fn fmt_hms_fraction_only_when_nonzero_and_not_omitted() {
    assert_eq!(hms(0, 0, 0, 0, 0), "00:00:00");
    assert_eq!(hms(1, 2, 3, 0, 0), "01:02:03");
    assert_eq!(hms(1, 2, 3, 10, 0), "01:02:03.000010");
    assert_eq!(hms(23, 59, 59, 999_999, 0), "23:59:59.999999");
    assert_eq!(hms(12, 0, 0, 100_000, 0), "12:00:00.100000");
    assert_eq!(hms(23, 59, 59, 999_999, OPT_OMIT_MICROSECONDS), "23:59:59");
    // options that don't concern the time of day change nothing
    assert_eq!(
        hms(1, 2, 3, 4, OPT_UTC_Z | OPT_NAIVE_UTC),
        "01:02:03.000004"
    );
}

#[test]
fn fmt_offset_is_isoformats() {
    assert_eq!(off(0, 0), "+00:00");
    assert_eq!(off(0, OPT_UTC_Z), "Z");
    assert_eq!(off(8 * H, 0), "+08:00");
    assert_eq!(off(8 * H, OPT_UTC_Z), "+08:00");
    assert_eq!(off(-5 * H, 0), "-05:00");
    assert_eq!(off(5 * H + 30 * M, 0), "+05:30");
    // DV-1: seconds and microseconds are written, the sign is kept
    assert_eq!(off(5 * H + 59 * M + 30 * S, 0), "+05:59:30");
    assert_eq!(off(5 * H + 1, 0), "+05:00:00.000001");
    assert_eq!(off(-S, 0), "-00:00:01");
    assert_eq!(off(-(24 * H - 1), 0), "-23:59:59.999999");
    assert_eq!(off(-(4 * H + 56 * M + 2 * S), 0), "-04:56:02");
    assert_eq!(off(1, 0), "+00:00:00.000001");
    assert_eq!(off(-1, OPT_UTC_Z), "-00:00:00.000001");
    // OMIT_MICROSECONDS is `replace(microsecond=0)` on the value, not the offset
    assert_eq!(off(5 * H + 1, OPT_OMIT_MICROSECONDS), "+05:00:00.000001");
}

// ---- civil_from_days -------------------------------------------------------

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

fn month_len(y: i64, mo: i64) -> i64 {
    match mo {
        2 if is_leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Every day from 0000-01-01 to 9999-12-31, against a date advanced one day at
/// a time. Also pins the range constants to those two dates.
#[test]
fn civil_from_days_exhaustive() {
    let (mut y, mut mo, mut d) = (0i64, 1i64, 1i64);
    for days in MIN_DAYS..=MAX_DAYS {
        let got = civil_from_days(days);
        assert_eq!(
            (got.0 as i64, got.1 as i64, got.2 as i64),
            (y, mo, d),
            "day {days}"
        );
        if days == 0 {
            assert_eq!((y, mo, d), (1970, 1, 1));
        }
        d += 1;
        if d > month_len(y, mo) {
            d = 1;
            mo += 1;
            if mo > 12 {
                mo = 1;
                y += 1;
            }
        }
    }
    assert_eq!((y, mo, d), (10000, 1, 1));
}

// ---- parse_dt64 ------------------------------------------------------------

fn unit(base: Unit, mult: i64) -> Dt64Unit {
    Dt64Unit { base, mult }
}

#[test]
fn parse_dt64_valid() {
    let cases: &[(&[u8], Unit)] = &[
        (b"<M8[Y]", Unit::Year),
        (b"<M8[M]", Unit::Month),
        (b"<M8[W]", Unit::Week),
        (b"<M8[D]", Unit::Day),
        (b"<M8[h]", Unit::Hour),
        (b"<M8[m]", Unit::Minute),
        (b"<M8[s]", Unit::Second),
        (b"<M8[ms]", Unit::Milli),
        (b"<M8[us]", Unit::Micro),
        (b"<M8[ns]", Unit::Nano),
        (b"<M8[ps]", Unit::Pico),
        (b"<M8[fs]", Unit::Femto),
        (b"<M8[as]", Unit::Atto),
    ];
    for &(s, base) in cases {
        assert_eq!(parse_dt64(s), Some(unit(base, 1)), "{s:?}");
    }
    assert_eq!(parse_dt64(b"<M8"), Some(unit(Unit::Generic, 1)));
    assert_eq!(parse_dt64(b">M8[ns]"), Some(unit(Unit::Nano, 1)));
    assert_eq!(parse_dt64(b"=M8[D]"), Some(unit(Unit::Day, 1)));
    assert_eq!(parse_dt64(b"|M8[s]"), Some(unit(Unit::Second, 1)));
}

#[test]
fn parse_dt64_multiplied() {
    assert_eq!(parse_dt64(b"<M8[10ms]"), Some(unit(Unit::Milli, 10)));
    assert_eq!(parse_dt64(b"<M8[2D]"), Some(unit(Unit::Day, 2)));
    assert_eq!(parse_dt64(b"<M8[1ns]"), Some(unit(Unit::Nano, 1)));
    assert_eq!(parse_dt64(b"<M8[1000Y]"), Some(unit(Unit::Year, 1000)));
    assert_eq!(
        parse_dt64(b"<M8[2147483647as]"),
        Some(unit(Unit::Atto, 2_147_483_647))
    );
}

#[test]
fn parse_dt64_unparseable() {
    for s in [
        &b""[..],
        b"M8[ns]",
        b"<m8[ns]",
        b"<M4[ns]",
        b"<M8[",
        b"<M8[ns",
        b"<M8[]",
        b"<M8[xx]",
        b"<M8[0ns]",
        b"<M8[10]",
        b"<M8[ns]x",
        b"<M8ns",
        b"<M8[-1ns]",
        b"<M8[99999999999999999999ns]",
        b"<M8[generic]",
        b"<f8",
    ] {
        assert_eq!(parse_dt64(s), None, "{:?}", String::from_utf8_lossy(s));
    }
}

// ---- dt64_to_parts: design boundaries --------------------------------------

fn parts(y: u16, mo: u8, d: u8, h: u8, mi: u8, s: u8, us: u32) -> Parts {
    Parts {
        y,
        mo,
        d,
        h,
        mi,
        s,
        us,
    }
}

fn ok(v: i64, base: Unit, mult: i64) -> Parts {
    match dt64_to_parts(v, unit(base, mult)) {
        Ok(Some(p)) => p,
        other => panic!("{v} {base:?}×{mult}: {other:?}"),
    }
}

fn unrepresentable(v: i64, base: Unit, mult: i64) {
    assert_eq!(
        dt64_to_parts(v, unit(base, mult)),
        Err(Dt64Err::Unrepresentable),
        "{v} {base:?}×{mult}"
    );
}

const ALL_UNITS: [Unit; 14] = [
    Unit::Year,
    Unit::Month,
    Unit::Week,
    Unit::Day,
    Unit::Hour,
    Unit::Minute,
    Unit::Second,
    Unit::Milli,
    Unit::Micro,
    Unit::Nano,
    Unit::Pico,
    Unit::Femto,
    Unit::Atto,
    Unit::Generic,
];

#[test]
fn nat_is_none_in_every_unit() {
    // DV-5/6/7, generic NaT included
    for base in ALL_UNITS {
        for mult in [1, 10] {
            assert_eq!(dt64_to_parts(i64::MIN, unit(base, mult)), Ok(None));
        }
    }
}

#[test]
fn generic_value_is_declined() {
    // DV-13 / FR-7 (e)
    for v in [0, 5, -1, i64::MAX, i64::MIN + 1] {
        assert_eq!(
            dt64_to_parts(v, unit(Unit::Generic, 1)),
            Err(Dt64Err::GenericValue)
        );
    }
}

#[test]
fn dv8_multiplied_units() {
    assert_eq!(ok(1, Unit::Milli, 10), parts(1970, 1, 1, 0, 0, 0, 10_000));
    assert_eq!(ok(1, Unit::Day, 2), parts(1970, 1, 3, 0, 0, 0, 0));
    assert_eq!(ok(-1, Unit::Day, 2), parts(1969, 12, 30, 0, 0, 0, 0));
    assert_eq!(ok(3, Unit::Minute, 7), parts(1970, 1, 1, 0, 21, 0, 0));
}

#[test]
fn dv9_months_floor() {
    assert_eq!(ok(-1, Unit::Month, 1), parts(1969, 12, 1, 0, 0, 0, 0));
    assert_eq!(ok(-2, Unit::Month, 1), parts(1969, 11, 1, 0, 0, 0, 0));
    assert_eq!(ok(-12, Unit::Month, 1), parts(1969, 1, 1, 0, 0, 0, 0));
    assert_eq!(ok(-13, Unit::Month, 1), parts(1968, 12, 1, 0, 0, 0, 0));
    assert_eq!(ok(11, Unit::Month, 1), parts(1970, 12, 1, 0, 0, 0, 0));
    assert_eq!(ok(-1, Unit::Month, 3), parts(1969, 10, 1, 0, 0, 0, 0));
    // 0000-01 and 9999-12 are the ends
    assert_eq!(ok(-1970 * 12, Unit::Month, 1), parts(0, 1, 1, 0, 0, 0, 0));
    unrepresentable(-1970 * 12 - 1, Unit::Month, 1);
    assert_eq!(
        ok(8030 * 12 - 1, Unit::Month, 1),
        parts(9999, 12, 1, 0, 0, 0, 0)
    );
    unrepresentable(8030 * 12, Unit::Month, 1);
    unrepresentable(i64::MAX, Unit::Month, 1);
}

#[test]
fn years_range() {
    assert_eq!(ok(0, Unit::Year, 1), parts(1970, 1, 1, 0, 0, 0, 0));
    assert_eq!(ok(-1970, Unit::Year, 1), parts(0, 1, 1, 0, 0, 0, 0));
    unrepresentable(-1971, Unit::Year, 1);
    assert_eq!(ok(8029, Unit::Year, 1), parts(9999, 1, 1, 0, 0, 0, 0));
    unrepresentable(8030, Unit::Year, 1);
    // numpy's astype wraps this to 1969 (§1a); the value is out of range
    unrepresentable(i64::MAX, Unit::Year, 1);
    unrepresentable(i64::MIN + 1, Unit::Year, 1);
}

#[test]
fn dv10_last_day() {
    assert_eq!(ok(MAX_DAYS, Unit::Day, 1), parts(9999, 12, 31, 0, 0, 0, 0));
    unrepresentable(MAX_DAYS + 1, Unit::Day, 1);
    assert_eq!(ok(MIN_DAYS, Unit::Day, 1), parts(0, 1, 1, 0, 0, 0, 0));
    unrepresentable(MIN_DAYS - 1, Unit::Day, 1);
    let last_us = (MAX_DAYS + 1) * 86_400_000_000 - 1;
    assert_eq!(
        ok(last_us, Unit::Micro, 1),
        parts(9999, 12, 31, 23, 59, 59, 999_999)
    );
    unrepresentable(last_us + 1, Unit::Micro, 1);
    let first_s = MIN_DAYS * 86_400;
    assert_eq!(ok(first_s, Unit::Second, 1), parts(0, 1, 1, 0, 0, 0, 0));
    unrepresentable(first_s - 1, Unit::Second, 1);
}

#[test]
fn dv11_overflowing_minutes() {
    unrepresentable(307_445_734_561_825_861, Unit::Minute, 1);
    unrepresentable(i64::MAX, Unit::Week, 1);
    unrepresentable(i64::MIN + 1, Unit::Day, 1);
    unrepresentable(i64::MAX, Unit::Hour, 1000);
}

#[test]
fn dv16_sub_microsecond_floors() {
    assert_eq!(ok(5, Unit::Pico, 1), parts(1970, 1, 1, 0, 0, 0, 0));
    assert_eq!(
        ok(-1, Unit::Pico, 1),
        parts(1969, 12, 31, 23, 59, 59, 999_999)
    );
    assert_eq!(
        ok(i64::MAX, Unit::Pico, 1),
        parts(1970, 4, 17, 18, 2, 52, 36_854)
    );
    assert_eq!(
        ok(-1, Unit::Nano, 1),
        parts(1969, 12, 31, 23, 59, 59, 999_999)
    );
    assert_eq!(ok(1_999, Unit::Nano, 1), parts(1970, 1, 1, 0, 0, 0, 1));
    assert_eq!(
        ok(-1, Unit::Atto, 1),
        parts(1969, 12, 31, 23, 59, 59, 999_999)
    );
    // M8[ns] i64 MIN+1: numpy's astype overflows here (§1a); the meaning is defined
    assert_eq!(
        ok(i64::MIN + 1, Unit::Nano, 1),
        parts(1677, 9, 21, 0, 12, 43, 145_224)
    );
    // M8[10ns] 1e18: numpy's datetime_as_string overflows (R10-B1)
    assert_eq!(
        ok(1_000_000_000_000_000_000, Unit::Nano, 10),
        parts(2286, 11, 20, 17, 46, 40, 0)
    );
}

// ---- dt64_to_parts: the exact-integer model --------------------------------

/// µs per `n` units as a ratio `num / den`.
fn ratio(base: Unit) -> (i128, i128) {
    match base {
        Unit::Week => (604_800_000_000, 1),
        Unit::Day => (86_400_000_000, 1),
        Unit::Hour => (3_600_000_000, 1),
        Unit::Minute => (60_000_000, 1),
        Unit::Second => (1_000_000, 1),
        Unit::Milli => (1_000, 1),
        Unit::Micro => (1, 1),
        Unit::Nano => (1, 1_000),
        Unit::Pico => (1, 1_000_000),
        Unit::Femto => (1, 1_000_000_000),
        Unit::Atto => (1, 1_000_000_000_000),
        Unit::Year | Unit::Month | Unit::Generic => unreachable!(),
    }
}

fn year_len(y: i128) -> i128 {
    if is_leap(y as i64) {
        366
    } else {
        365
    }
}

/// Days from 1970-01-01 to `y`-01-01, by summing year lengths.
fn days_to_year(y: i128) -> i128 {
    let mut d = 0;
    if y >= 1970 {
        for yy in 1970..y {
            d += year_len(yy);
        }
    } else {
        for yy in y..1970 {
            d -= year_len(yy);
        }
    }
    d
}

/// The date `days` after 1970-01-01, by 400-year cycles then a year and
/// month walk.
fn walk_date(days: i128) -> (i128, i128, i128) {
    let mut y = 1970i128;
    let mut d = days;
    let cycles = d.div_euclid(146_097);
    d -= cycles * 146_097;
    y += cycles * 400;
    while d >= year_len(y) {
        d -= year_len(y);
        y += 1;
    }
    let mut mo = 1;
    while d >= month_len(y as i64, mo) as i128 {
        d -= month_len(y as i64, mo) as i128;
        mo += 1;
    }
    (y, mo as i128, d + 1)
}

struct Model {
    lo_us: i128,
    hi_us: i128,
}

impl Model {
    fn new() -> Self {
        Model {
            lo_us: days_to_year(0) * US_PER_DAY,
            hi_us: days_to_year(10_000) * US_PER_DAY,
        }
    }

    fn eval(&self, v: i64, u: Dt64Unit) -> Result<Option<Parts>, Dt64Err> {
        if v == i64::MIN {
            return Ok(None);
        }
        let n = v as i128 * u.mult as i128;
        let p = |y: i128, mo: i128, d: i128, rem: i128| Parts {
            y: y as u16,
            mo: mo as u8,
            d: d as u8,
            h: (rem / 3_600_000_000) as u8,
            mi: (rem / 60_000_000 % 60) as u8,
            s: (rem / 1_000_000 % 60) as u8,
            us: (rem % 1_000_000) as u32,
        };
        match u.base {
            Unit::Generic => Err(Dt64Err::GenericValue),
            Unit::Year => {
                let y = 1970 + n;
                if (0..10_000).contains(&y) {
                    Ok(Some(p(y, 1, 1, 0)))
                } else {
                    Err(Dt64Err::Unrepresentable)
                }
            }
            Unit::Month => {
                let t = 1970 * 12 + n; // months since 0000-01
                if (0..10_000 * 12).contains(&t) {
                    Ok(Some(p(t / 12, t % 12 + 1, 1, 0)))
                } else {
                    Err(Dt64Err::Unrepresentable)
                }
            }
            base => {
                let (num, den) = ratio(base);
                let Some(scaled) = n.checked_mul(num) else {
                    return Err(Dt64Err::Unrepresentable);
                };
                let us = scaled.div_euclid(den);
                if us < self.lo_us || us >= self.hi_us {
                    return Err(Dt64Err::Unrepresentable);
                }
                let (y, mo, d) = walk_date(us.div_euclid(US_PER_DAY));
                Ok(Some(p(y, mo, d, us.rem_euclid(US_PER_DAY))))
            }
        }
    }

    /// `v` values whose result crosses 0000-01-01 or 10000-01-01, ±2.
    fn crossings(&self, u: Dt64Unit) -> Vec<i64> {
        let targets: [i128; 2] = match u.base {
            Unit::Year => [-1970, 8030],
            Unit::Month => [-1970 * 12, 8030 * 12],
            Unit::Generic => return vec![],
            base => {
                let (num, den) = ratio(base);
                [self.lo_us * den / num, self.hi_us * den / num]
            }
        };
        let mut out = vec![];
        for t in targets {
            let v0 = t.div_euclid(u.mult as i128);
            for dv in -2..=2 {
                if let Ok(v) = i64::try_from(v0 + dv) {
                    out.push(v);
                }
            }
        }
        out
    }
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The round-12 cross-check: 13 units × {1,2,3,7,10,1000} × boundary values
/// plus seeded random values, against the exact-integer model.
#[test]
fn dt64_to_parts_matches_exact_integer_model() {
    let model = Model::new();
    let mut rng = SplitMix(0x1503_2026_0924);
    let (mut cases, mut declined) = (0usize, 0usize);
    for base in ALL_UNITS {
        for mult in [1i64, 2, 3, 7, 10, 1000] {
            let u = unit(base, mult);
            let mut vs = vec![i64::MIN, i64::MIN + 1, i64::MAX, i64::MAX - 1];
            vs.extend(-2..=2);
            vs.extend(model.crossings(u));
            for _ in 0..300 {
                vs.push(rng.next() as i64);
            }
            // small magnitudes, where most in-range values of coarse units live
            for k in 1..19 {
                let m = 10u64.pow(k);
                for _ in 0..20 {
                    let x = (rng.next() % m) as i64;
                    vs.push(if rng.next() & 1 == 0 { x } else { -x });
                }
            }
            for v in vs {
                let want = model.eval(v, u);
                assert_eq!(dt64_to_parts(v, u), want, "{v} {base:?}×{mult}");
                cases += 1;
                declined += want.is_err() as usize;
            }
        }
    }
    // the sample must exercise both outcomes, not just one
    assert!(cases > 50_000, "{cases}");
    assert!(
        declined > 5_000 && cases - declined > 5_000,
        "{declined}/{cases}"
    );
}
