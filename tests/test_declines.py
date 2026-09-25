"""E2E-13: the decline rule (design FR-7), and unaligned reads (FR-6).

Every case (a)–(f): with a `default` it receives the whole original object;
without one, the exact FR-7 message; with a raising `default`, orjson's
message with `default`'s exception as the cause. In both raising cases the
note is `reason_note`'s text, whose exact form is pinned by the Rust tests
(C6); here it must name the evidence.
"""

import json
import warnings

import numpy as np
import pytest

import isojson
from oracle.python_api import expected_text, float_matches

NUMPY = isojson.OPT_SERIALIZE_NUMPY
INDENT = isojson.OPT_INDENT_2
pytestmark = pytest.mark.filterwarnings("ignore:The 'generic' unit:DeprecationWarning")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    GENERIC_NAT_5 = np.array([np.iinfo("i8").min, 5], dtype="i8").view("M8")

PREFIX = "isojson can't write this numpy object natively: "

# (id, object, message, evidence the note must contain)
CASES = [
    ("a-not-contiguous", np.arange(6).reshape(2, 3).T,
     "numpy array is not C contiguous; use ndarray.tolist() in default", "shape=(3, 2)"),
    ("b-not-native", np.array([1, 2], dtype=">i4"), "numpy array is not native-endianness", "dtype.str='>i4'"),
    ("c-zero-dim", np.array(5.0), "unsupported datatype in numpy array", "0-d array, dtype.str='<f8'"),
    ("d-dtype-str", np.array(["a"]), "unsupported datatype in numpy array", "dtype.str='<U1'"),
    ("d-dtype-complex", np.array([1j]), "unsupported datatype in numpy array", "dtype.str='<c16'"),
    ("d-dtype-object", np.array([None]), "unsupported datatype in numpy array", "dtype.str='|O'"),
    ("e-generic-value", GENERIC_NAT_5, "unsupported numpy.datetime64 unit: generic", "value=5"),
    ("e-generic-scalar", GENERIC_NAT_5[1], "unsupported numpy.datetime64 unit: generic", "value=5"),
    ("f-overflow", np.array([307445734561825861], dtype="M8[m]"),
     "unrepresentable numpy.datetime64: 307445734561825861 minutes", "outside 0000-01-01"),
    ("f-year", np.array([0, 8030], dtype="M8[Y]"), "unrepresentable numpy.datetime64: 8030 years", "outside"),
    ("f-multiplied", np.array([10**17], dtype="M8[10s]"),
     "unrepresentable numpy.datetime64: 100000000000000000 seconds × 10", "outside"),
    ("f-scalar", np.datetime64(-2000, "Y"), "unrepresentable numpy.datetime64: -2000 years", "outside"),
]


@pytest.mark.parametrize(("x", "message", "evidence"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_without_default_raises_the_reason(x, message, evidence):
    with pytest.raises(TypeError) as e:
        isojson.dumps(x, option=NUMPY)
    assert str(e.value) == message
    assert e.value.__notes__ == [e.value.__notes__[0]]
    note = e.value.__notes__[0]
    assert note.startswith(PREFIX + message) and evidence in note, note


@pytest.mark.parametrize(("x", "message", "evidence"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_default_receives_the_whole_object(x, message, evidence):
    seen = []
    out = isojson.dumps([1, x, 2], option=NUMPY, default=lambda o: seen.append(o) or "D")
    assert out == b'[1,"D",2]'
    assert len(seen) == 1 and seen[0] is x


@pytest.mark.parametrize(("x", "message", "evidence"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_raising_default_keeps_orjsons_error_and_adds_the_note(x, message, evidence):
    def boom(_):
        raise ValueError("nope")

    with pytest.raises(TypeError) as e:
        isojson.dumps(x, option=NUMPY, default=boom)
    assert str(e.value) == f"Type is not JSON serializable: numpy.{type(x).__qualname__}"
    assert type(e.value.__cause__) is ValueError
    assert len(e.value.__notes__) == 1
    note = e.value.__notes__[0]
    assert note.startswith(PREFIX + message) and evidence in note, note


def test_default_depth_limit_gets_no_note():
    x = np.array(["a"])
    with pytest.raises(TypeError) as e:
        isojson.dumps(x, option=NUMPY, default=lambda o: o)  # returns itself forever
    assert str(e.value) == "default serializer exceeds recursion limit"
    assert not getattr(e.value, "__notes__", None)


def test_generic_element_less_array():
    """(e) is decided per element: no element, nothing to decline (as orjson)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        x = np.zeros((2, 0), "M8")
    assert isojson.dumps(x, option=NUMPY) == b"[[],[]]"


@pytest.mark.parametrize("x", [GENERIC_NAT_5, GENERIC_NAT_5.reshape(1, 2)], ids=["1-D", "2-D"])
def test_generic_nat_then_value_rolls_back(x):
    """The NaT is written `null` first, then the value declines: the output
    is rolled back and `default` gets the whole array."""
    assert isojson.dumps({"k": x}, option=NUMPY, default=lambda a: a.shape) == (
        b'{"k":[' + b",".join(str(n).encode() for n in x.shape) + b"]}"
    )


def test_unrecognized_scalars_take_the_plain_default_path():
    for x in (np.complex128(1), np.longdouble(1)):
        assert isojson.dumps(x, option=NUMPY, default=lambda o: "D") == b'"D"'
        with pytest.raises(TypeError) as e:
            isojson.dumps(x, option=NUMPY)
        assert str(e.value) == f"Type is not JSON serializable: numpy.{type(x).__qualname__}"
        assert not getattr(e.value, "__notes__", None)


@pytest.mark.parametrize("opts", [NUMPY, NUMPY | INDENT], ids=["compact", "indent"])
def test_rollback_equals_the_document_with_defaults_result(opts):
    """(f) in row 2 of a 2-D array inside a document: the bytes equal the
    same document with the array replaced by `default(array)`."""
    arr = np.array([[0, 1], [2, 307445734561825861]], dtype="M8[m]")

    def default(a):
        return {"replaced": [str(v) for v in a.shape]}

    got = isojson.dumps({"a": 1, "x": [0, arr]}, option=opts, default=default)
    want = isojson.dumps({"a": 1, "x": [0, default(arr)]}, option=opts)
    assert got == want


UNALIGNED = ["f2", "f4", "f8", "i2", "i8", "M8[ns]"]


@pytest.mark.parametrize("dtype", UNALIGNED)
@pytest.mark.parametrize("shape", [(5,), (2, 3)], ids=["1-D", "2-D"])
def test_unaligned_arrays_read_correctly(dtype, shape):
    """FR-6: `np.frombuffer(…, offset=1)` is C-contiguous but unaligned; a
    typed-slice read is UB (and fails the debug build's alignment check)."""
    src = (np.arange(int(np.prod(shape))) * 7 - 5).astype(dtype.replace("M8[ns]", "i8"))
    if dtype.startswith("M8"):
        src = (src * 10**15).view(dtype)
    raw = b"\0" + src.tobytes()
    x = np.frombuffer(raw, dtype=src.dtype, offset=1).reshape(shape)
    assert x.ctypes.data % max(2, src.itemsize) != 0 and x.flags.c_contiguous
    out = isojson.dumps(x, option=NUMPY)
    got = [v for row in (json.loads(out) if len(shape) > 1 else [json.loads(out)]) for v in row]
    flat = x.reshape(-1)
    for text, e in zip(got, flat, strict=True):
        if dtype.startswith("f"):
            assert float_matches(json.dumps(text).encode(), e)
        else:
            assert json.dumps(text).encode() == expected_text(e)
