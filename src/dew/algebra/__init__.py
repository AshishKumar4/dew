"""Dew's algebraic derivation of its own forms (docs/design/algebra.md): a template's walk forms derived from
its declaration (`derive`: a `Reduction` to its `Derived` init, step, merge, finish and mapped outputs) and a
recurrence's chunk forms and adjoint from its token update (`scan`), each shared piece in `forms`."""

from dew.algebra import derive, forms, scan
from dew.algebra.derive import Reduction
from dew.algebra.forms import Derived, Refused
from dew.algebra.scan import Scan

__all__ = ["Derived", "Reduction", "Refused", "Scan", "derive", "forms", "scan"]
