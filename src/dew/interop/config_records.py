"""Constructor-checked records that keep only the fields a translator states."""

from collections.abc import Callable, Mapping

from dew.registry import Configured, configured, from_record


class NativeFields[Value](dict[str, Configured]):
    """A native value's supplied fields, without its unmentioned defaults.

    The record keeps checkpoint serialization independent of constructor
    defaults. Its typed value is rebuilt from the current record, so edits
    to a field are read against the same native declarations. A copy made
    as a plain dict (`dict(record)`, `{**record}`) has no `.value`: edit the
    record itself, or build its value first.
    """

    def __init__(self, owner: type[Value], fields: Mapping[str, Configured]):
        super().__init__(fields)
        self._owner = owner

    @property
    def value(self) -> Value:
        """Build the native value from the fields this record currently holds."""
        return from_record(self._owner, self)


def native_fields[**Fields, Value](constructor: Callable[Fields, Value]
                                 ) -> Callable[Fields, NativeFields[Value]]:
    """Use the native constructor's signature to check a translated record."""
    if not isinstance(constructor, type):
        raise TypeError("a native record names its value class")

    def recorded(*args: Fields.args, **kwargs: Fields.kwargs) -> NativeFields[Value]:
        if args:
            raise TypeError("a constructor record states its fields by name")
        return NativeFields(constructor, {name: configured(value) for name, value in kwargs.items()})

    return recorded
