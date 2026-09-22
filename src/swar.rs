//! Word-at-a-time byte scanning (8 bytes per step, portable, no SIMD intrinsics).

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

#[inline(always)]
fn has_zero(w: u64) -> u64 {
    w.wrapping_sub(LO) & !w & HI
}

/// Does this word contain a byte that is special inside a JSON string:
/// a control character (< 0x20), `"` or `\`? Exact (no false negatives or
/// positives on the "any" question).
#[inline(always)]
pub(crate) fn has_special(w: u64) -> bool {
    let ctrl = w.wrapping_sub(LO * 0x20) & !w & HI;
    (ctrl | has_zero(w ^ (LO * b'"' as u64)) | has_zero(w ^ (LO * b'\\' as u64))) != 0
}

#[inline(always)]
pub(crate) fn load(s: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(s[at..at + 8].try_into().unwrap())
}

/// Index of the first byte in `s[from..]` that is `"`, `\`, or < 0x20.
#[inline]
pub(crate) fn find_special(s: &[u8], mut i: usize) -> Option<usize> {
    while i + 8 <= s.len() {
        if has_special(load(s, i)) {
            break;
        }
        i += 8;
    }
    while i < s.len() {
        let b = s[i];
        if b < 0x20 || b == b'"' || b == b'\\' {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline]
pub(crate) fn is_ascii(s: &[u8]) -> bool {
    let mut i = 0;
    while i + 8 <= s.len() {
        if load(s, i) & HI != 0 {
            return false;
        }
        i += 8;
    }
    s[i..].iter().all(|&b| b < 0x80)
}

/// Are all 8 bytes ASCII digits? (fast_float's `is_8digits`)
#[inline(always)]
pub(crate) fn is_8digits(v: u64) -> bool {
    let a = v.wrapping_add(0x4646_4646_4646_4646);
    let b = v.wrapping_sub(0x3030_3030_3030_3030);
    (a | b) & HI == 0
}

/// Value of 8 ASCII digits loaded little-endian (fast_float's `parse_8digits`).
#[inline(always)]
pub(crate) fn parse_8digits(mut v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064;
    const MUL2: u64 = 0x0000_2710_0000_0001;
    v -= 0x3030_3030_3030_3030;
    v = (v * 10) + (v >> 8);
    let v1 = (v & MASK).wrapping_mul(MUL1);
    let v2 = ((v >> 16) & MASK).wrapping_mul(MUL2);
    ((v1.wrapping_add(v2) >> 32) as u32) as u64
}

#[inline(always)]
pub(crate) fn load4(s: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(s[at..at + 4].try_into().unwrap())
}

#[inline(always)]
pub(crate) fn is_4digits(v: u32) -> bool {
    let a = v.wrapping_add(0x4646_4646);
    let b = v.wrapping_sub(0x3030_3030);
    (a | b) & 0x8080_8080 == 0
}

/// Value of 4 ASCII digits loaded little-endian.
#[inline(always)]
pub(crate) fn parse_4digits(mut v: u32) -> u64 {
    v -= 0x3030_3030;
    v = v.wrapping_mul(10).wrapping_add(v >> 8); // bytes 0 and 2 hold d0d1, d2d3
    (((v & 0x00FF_00FF).wrapping_mul(0x0064_0001) >> 16) & 0xFFFF) as u64
}
