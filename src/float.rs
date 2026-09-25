//! Float formatting.
//!
//! Output matches orjson byte-for-byte: shortest round-trip digits via zmij
//! (the same algorithm orjson uses), fixed notation for decimal exponents in
//! [-5, 16) and scientific otherwise (`1e+16`, `1e-6`, `0.00001`). Non-finite
//! values are `null`.

#[inline]
pub(crate) fn write_f64(out: &mut crate::out::Out, v: f64) {
    if !v.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    write_finite(out, v);
}

/// float32, and float16 widened with `f16_to_f32`: the shortest digits that
/// round-trip through f32, with zmij's f32 notation switch (as orjson).
#[inline]
pub(crate) fn write_f32(out: &mut crate::out::Out, v: f32) {
    if !v.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    write_finite(out, v);
}

/// `v` must be finite. zmij's `Float` trait is sealed, so the finite check
/// can't be written generically and lives in the typed writers.
///
/// The digits are formatted in place at the end of `out` (as orjson does):
/// `zmij::Buffer` is a byte array (alignment 1) and `format_finite` always
/// writes from its start. Formatting into a stack buffer and copying it out
/// reloads bytes just written by smaller stores, which x86 can't forward:
/// measured on x86_64, 25–36 ns per float for values in [0, 1), normals and
/// integral floats, against 16–18 ns in place.
#[inline]
fn write_finite<F: zmij::Float>(out: &mut crate::out::Out, v: F) {
    out.reserve(core::mem::size_of::<zmij::Buffer>());
    unsafe {
        let buf = &mut *out.as_mut_ptr().add(out.len()).cast::<zmij::Buffer>();
        let n = buf.format_finite(v).len();
        out.set_len(out.len() + n);
    }
}

// SPDX-License-Identifier: (Apache-2.0 OR MIT)
// Copyright half-rs Contributors (2016-2026)
// https://github.com/VoidStarKat/half-rs
//
// `f16_to_f32_fallback` as shipped in orjson (`src/serialize/writer/half.rs`),
// with `as` casts in place of `cast_signed`/`cast_unsigned` and
// `f32::from_bits` without `unsafe` (clippy rejects the redundant block). Exact: every f16 is an f32. Proven over all 65,536 inputs
// by E2E-12.
pub(crate) const fn f16_to_f32(i: u16) -> f32 {
    if i & 0x7FFFu16 == 0 {
        return f32::from_bits((i as u32) << 16);
    }
    let half_sign = (i & 0x8000u16) as u32;
    let half_exp = (i & 0x7C00u16) as u32;
    let half_man = (i & 0x03FFu16) as u32;
    if half_exp == 0x7C00u32 {
        if half_man == 0 {
            return f32::from_bits((half_sign << 16) | 0x7F80_0000u32);
        } else {
            return f32::from_bits((half_sign << 16) | 0x7FC0_0000u32 | (half_man << 13));
        }
    }
    let sign = half_sign << 16;
    let unbiased_exp = ((half_exp as i32) >> 10) - 15;
    if half_exp == 0 {
        let e = (half_man as u16).leading_zeros() - 6;
        let exp = (127 - 15 - e) << 23;
        let man = (half_man << (14 + e)) & 0x7F_FF_FFu32;
        return f32::from_bits(sign | exp | man);
    }
    let exp = ((unbiased_exp + 127) as u32) << 23;
    let man = (half_man & 0x03FFu32) << 13;
    f32::from_bits(sign | exp | man)
}

/// Append up to 32 bytes without a `memcpy` call: two possibly-overlapping
/// fixed-size copies cover every length in a size class, and fixed-size
/// copies compile to plain loads/stores. Never reads outside `src`.
#[inline(always)]
pub(crate) fn small_copy(out: &mut crate::out::Out, src: &[u8]) {
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
