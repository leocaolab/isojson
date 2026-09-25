"""E2E-11: `sys.modules["numpy"]` replaced after the group is loaded (FR-3).

Each case runs in a fresh child process. A cache hit is trusted; a miss
re-reads the group when `sys.modules["numpy"]` is no longer the module it
was read from. isojson trusts `sys.modules` as orjson does: what is
registered there is the caller's choice (design §13).
"""

import subprocess
import sys
import textwrap

import pytest


PRELUDE = """
import sys, types, numpy as np, isojson
N = isojson.OPT_SERIALIZE_NUMPY
assert isojson.dumps(np.float64(1.0), option=N) == b"1.0"   # loads the group
"""


def run(body):
    code = PRELUDE + textwrap.dedent(body)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stderr[-3000:])
    return r.stdout.strip()




def test_reread_after_replacement():
    """(1) A module whose `float64` is a same-layout subclass `F`: after a
    miss, `F(2.5)` is written natively, so the group was re-read."""
    out = run(
        """
        F = type("float64", (np.float64,), {})
        fake = types.ModuleType("numpy")
        fake.__dict__.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
        fake.float64 = F
        sys.modules["numpy"] = fake
        # before a miss, F is not in the cached group: it goes to default
        # (and that miss is what triggers the re-read)
        print(isojson.dumps(F(2.5), option=N, default=lambda o: "D").decode())
        print(isojson.dumps(F(2.5), option=N, default=lambda o: "D").decode())
        print(isojson.dumps(np.int64(7), option=N).decode())
        """
    )
    assert out.splitlines() == ["2.5", "2.5", "7"]


def test_hit_is_trusted_until_a_miss():
    """A hit is trusted (strong refs, no address reuse): replacing the module
    alone changes nothing for the cached types."""
    out = run(
        """
        sys.modules["numpy"] = types.ModuleType("numpy")
        print(isojson.dumps(np.float64(1.5), option=N, default=lambda o: "D").decode())
        """
    )
    assert out == "1.5"


def test_stub_missing_attributes_goes_to_default():
    """(2) A stub module missing names: after a miss the group is Absent, and
    numpy objects go to `default`."""
    out = run(
        """
        stub = types.ModuleType("numpy")
        stub.ndarray = np.ndarray
        sys.modules["numpy"] = stub
        isojson.dumps(object(), option=N, default=repr)   # a miss: the re-read
        print(isojson.dumps([np.float64(1.5), np.arange(2)], option=N, default=lambda o: type(o).__name__).decode())
        """
    )
    assert out == '["float64","ndarray"]'


@pytest.mark.parametrize("value", ["None", "42", "object()"])
def test_non_module_is_absent_without_error(value):
    """(3) A non-module object in `sys.modules["numpy"]`: after a miss the
    group is Absent, with no exception."""
    out = run(
        f"""
        sys.modules["numpy"] = {value}
        isojson.dumps(object(), option=N, default=repr)   # a miss: the re-read
        print(isojson.dumps([np.float64(1.5), 1], option=N, default=lambda o: "D").decode())
        """
    )
    assert out == '["D",1]'


def test_recheck_is_once_per_call():
    """FR-3 re-checks `sys.modules["numpy"]` at a call's first miss only: a
    numpy replaced by `default` mid-call is seen from the next call."""
    out = run(
        """
        F = type("float64", (np.float64,), {})
        fake = types.ModuleType("numpy")
        fake.__dict__.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
        fake.float64 = F
        def default(o):
            if isinstance(o, F):
                return "D"
            sys.modules["numpy"] = fake     # swapped after this call's first miss
            return "x"
        print(isojson.dumps([object(), F(2.5)], option=N, default=default).decode())
        print(isojson.dumps([F(2.5)], option=N, default=default).decode())
        """
    )
    assert out.splitlines() == ['["x","D"]', "[2.5]"]
