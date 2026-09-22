//! Float formatting.
//!
//! Output matches orjson byte-for-byte: shortest round-trip digits via zmij
//! (the same algorithm orjson uses), fixed notation for decimal exponents in
//! [-5, 16) and scientific otherwise (`1e+16`, `1e-6`, `0.00001`). Non-finite
//! values are `null`.

#[inline]
pub(crate) fn write_f64(out: &mut Vec<u8>, v: f64) {
    if !v.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    let mut buf = zmij::Buffer::new();
    small_copy(out, buf.format_finite(v).as_bytes());
}

/// Append up to 32 bytes without a `memcpy` call: two possibly-overlapping
/// fixed-size copies cover every length in a size class, and fixed-size
/// copies compile to plain loads/stores. Never reads outside `src`.
#[inline(always)]
pub(crate) fn small_copy(out: &mut Vec<u8>, src: &[u8]) {
    let n = src.len();
    debug_assert!(n <= 32);
    out.reserve(n);
    unsafe {
        let dst = out.as_mut_ptr().add(out.len());
        let s = src.as_ptr();
        if n >= 16 {
            core::ptr::copy_nonoverlapping(s, dst, 16);
            core::ptr::copy_nonoverlapping(s.add(n - 16), dst.add(n - 16), 16);
        } else if n >= 8 {
            core::ptr::copy_nonoverlapping(s, dst, 8);
            core::ptr::copy_nonoverlapping(s.add(n - 8), dst.add(n - 8), 8);
        } else if n >= 4 {
            core::ptr::copy_nonoverlapping(s, dst, 4);
            core::ptr::copy_nonoverlapping(s.add(n - 4), dst.add(n - 4), 4);
        } else {
            for i in 0..n {
                *dst.add(i) = *s.add(i);
            }
        }
        out.set_len(out.len() + n);
    }
}
