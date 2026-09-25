"""Strict own-GIL sub-interpreters on CPython 3.12–3.14, one API.

`make()` creates an isolated interpreter (own GIL, strict extension check,
no override), `run(i, code)` executes source in it and raises `RuntimeError`
with the interpreter's error if the code raised, `destroy(i)` closes it.
Modelled on `tests/test_subinterp_312_313.py`, which is left as it is (T-4).
`AVAILABLE` is False where the private module is missing.
"""

import sys

AVAILABLE = True

if sys.version_info[:2] == (3, 12):
    try:
        import _xxsubinterpreters as _xi
    except ImportError:
        AVAILABLE = False

    def make():
        return _xi.create(isolated=True)

    def run(i, code):
        try:
            _xi.run_string(i, code)
        except _xi.RunFailedError as e:
            raise RuntimeError(str(e)) from None

    def destroy(i):
        _xi.destroy(i)

elif sys.version_info[:2] == (3, 13):
    try:
        import _interpreters as _xi
    except ImportError:
        AVAILABLE = False

    def make():
        return _xi.create("isolated")

    def run(i, code):
        err = _xi.exec(i, code)
        if err is not None:
            raise RuntimeError(f"{err.type.__name__}: {err.msg}\n{err.formatted}")

    def destroy(i):
        _xi.destroy(i)

else:
    try:
        from concurrent import interpreters as _xi
    except ImportError:
        AVAILABLE = False

    def make():
        return _xi.create()

    def run(i, code):
        try:
            i.exec(code)
        except _xi.ExecutionFailed as e:
            raise RuntimeError(str(e)) from None

    def destroy(i):
        i.close()
