"""A run's and a block's kernel choices (`dew.kernels.Kernels`): one field an op, each holding exactly the
kernels that op's selectors serve and a site's own kernels, so a command line offers every kernel and no
other (api03), and a block naming them runs them."""

import dataclasses
import typing

import pytest

from dew import kernels
from dew.nn.kernels import generation


def test_the_kernels_type_names_every_choice_with_exactly_its_kernels():
    """`Kernels`' fields are the kernel choices (`generation.CHOICES`), each a choice among exactly the
    kernels its op's selectors hold (`tools/kernel_choices.served`, the production modules' every
    `selection.Op`, the generic path among them) or a site decides by name (`generation.SITE_KERNELS`), and
    its block is the one `tools/kernel_choices.py` writes from them (row 12776): a kernel added to an op, or
    a choice added, without its field fails here."""
    from tools import kernel_choices

    served = kernel_choices.served()
    text = kernel_choices.MODULE.read_text()
    first, last = kernel_choices.block(text)
    wanted = kernel_choices.rendered(served, kernel_choices.keyword_only_now())
    assert text.splitlines()[first:last + 1] == wanted
    hints = typing.get_type_hints(kernels.Kernels)
    offered = {field.name: frozenset(value for member in typing.get_args(hints[field.name])
                                     for value in typing.get_args(member))
               for field in dataclasses.fields(kernels.Kernels)}
    assert set(offered) == set(map(str, generation.CHOICES)) and offered == served


def test_a_block_runs_the_kernels_its_choices_name_and_its_names_over_them():
    """`use(Kernels(...), **names)` names each field it sets and each name over it (`generation.chosen`), the
    choices a dispatch then reads (`dew._cache.KERNEL_CHOICES`); a field left None names nothing."""
    from dew import _cache

    with kernels.use(kernels.Kernels(attention="xla", narrow_copies="on"), attention="reference"):
        assert dict(_cache.KERNEL_CHOICES.value) == {"attention": "reference", "narrow_copies": "on"}
    assert kernels.Kernels().named() == {}
    with pytest.raises(ValueError), kernels.use(narrow_copies="sideways"):
        pass
