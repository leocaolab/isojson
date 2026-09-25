"""Strict own-GIL sub-interpreters (`concurrent.interpreters`, CPython 3.14).

`make()` creates an isolated interpreter (own GIL, strict extension check,
no override), `run(i, code)` executes source in it and raises `RuntimeError`
with the interpreter's error if the code raised, `destroy(i)` closes it.
"""

from concurrent import interpreters as _xi


def make():
    return _xi.create()


def run(i, code):
    try:
        i.exec(code)
    except _xi.ExecutionFailed as e:
        raise RuntimeError(str(e)) from None


def destroy(i):
    i.close()
