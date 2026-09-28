"""Strict JSON records and offline schema validation."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)


@lru_cache(maxsize=2)
def validator(name):
    schema = loads((Path(__file__).parent / "schemas" / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


class JsonRecord:
    """Validated snapshot. Returned nested data cannot mutate the record."""

    __slots__ = ("_text",)
    schema_file = ""

    def __init__(self, **data):
        text = canonical_json(data)
        normalized = loads(text)
        validator(self.schema_file).validate(normalized)
        self.validate_semantics(normalized)
        object.__setattr__(self, "_text", text)

    def __setattr__(self, name, value):
        raise AttributeError("Contract records are immutable")

    def __getattr__(self, name):
        data = self.to_dict()
        if name not in data:
            raise AttributeError(name)
        return data[name]

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return loads(self._text)

    @staticmethod
    def validate_semantics(data):
        pass

