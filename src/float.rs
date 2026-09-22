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
    out.extend_from_slice(buf.format_finite(v).as_bytes());
}
