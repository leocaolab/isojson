"""Own-GIL sub-interpreters on CPython 3.12 / 3.13, which have no
`concurrent.interpreters` (3.14+): use the private `_xxsubinterpreters`
(3.12) / `_interpreters` (3.13) modules to create isolated interpreters
(own GIL, strict extension check, no override)."""

import sys
import threading

import pytest

if sys.version_info[:2] == (3, 12):
    xi = pytest.importorskip("_xxsubinterpreters")

    def make():
        return xi.create(isolated=True)

    def run(i, code):
        xi.run_string(i, code)

elif sys.version_info[:2] == (3, 13):
    xi = pytest.importorskip("_interpreters")

    def make():
        return xi.create("isolated")

    def run(i, code):
        err = xi.exec(i, code)
        if err is not None:
            raise RuntimeError(err)

else:
    pytest.skip("3.14+ is covered by test_subinterp.py", allow_module_level=True)

CODE = (
    "import isojson, json\n"
    "d = {'k': [1, 'é', 2.5, None], 'n': %d}\n"
    "assert isojson.loads(isojson.dumps(d)) == d\n"
    "assert issubclass(isojson.JSONDecodeError, json.JSONDecodeError)\n"
)


def test_isolated_interpreters_concurrently():
    ids = [make() for _ in range(6)]
    errors = []

    def worker(i, n):
        try:
            for k in range(200):
                run(i, CODE % (n * 1000 + k))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e)[:300])

    threads = [threading.Thread(target=worker, args=(i, n)) for n, i in enumerate(ids)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in ids:
        xi.destroy(i)
    assert not errors, errors
