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
