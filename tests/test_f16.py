"""E2E-12: every float16, the proof of `f16_to_f32` (design C5).

Each finite element's text, read back as float32, must equal the element
widened by numpy, bit for bit (so `-0.0` is caught); inf and NaN are `null`.
"""

import json

import numpy as np

import isojson


def test_all_65536_float16_values():
    a = np.arange(65536, dtype=np.uint16).view(np.float16)
    texts = isojson.dumps(a, option=isojson.OPT_SERIALIZE_NUMPY)[1:-1].split(b",")
    assert len(texts) == 65536
    with np.errstate(invalid="ignore"):
        wide = a.astype(np.float32).view(np.uint32)
    finite = np.isfinite(a)
    bad = []
    for i, t in enumerate(texts):
        if not finite[i]:
            if t != b"null":
                bad.append((i, t))
        elif np.float32(t.decode()).view(np.uint32) != wide[i]:
            bad.append((i, t))
    assert not bad, bad[:10]
    assert int((~finite).sum()) == 2 * 1024 + 2 - 2  # 2 infinities + 2046 NaNs
    json.loads(b"[" + b",".join(texts) + b"]")


def test_float16_scalars_match_the_array_path():
    a = np.arange(0, 65536, 257, dtype=np.uint16).view(np.float16)
    opt = isojson.OPT_SERIALIZE_NUMPY
    assert isojson.dumps(list(a), option=opt) == isojson.dumps(a, option=opt)
