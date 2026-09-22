import json
from typing import Any, Callable

__version__: str

class JSONDecodeError(json.JSONDecodeError): ...

JSONEncodeError = TypeError

OPT_APPEND_NEWLINE: int
OPT_INDENT_2: int
OPT_NAIVE_UTC: int
OPT_NON_STR_KEYS: int
OPT_OMIT_MICROSECONDS: int
OPT_PASSTHROUGH_DATACLASS: int
OPT_PASSTHROUGH_DATETIME: int
OPT_PASSTHROUGH_SUBCLASS: int
OPT_SERIALIZE_DATACLASS: int
OPT_SERIALIZE_NUMPY: int
OPT_SERIALIZE_UUID: int
OPT_SORT_KEYS: int
OPT_STRICT_INTEGER: int
OPT_UTC_Z: int

def dumps(
    obj: Any,
    /,
    default: Callable[[Any], Any] | None = None,
    option: int | None = None,
) -> bytes: ...
def loads(obj: bytes | bytearray | memoryview | str, /) -> Any: ...
