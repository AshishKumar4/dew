"""Scratch removal of the pairwise change; diagnostic only."""
from dew.nn import precision
from test_zz_q35_cause import test_probe as probe


def test_no_pairwise(monkeypatch):
    monkeypatch.setattr(precision,'projection_product',lambda *args,**kwargs:None)
    probe(users=True)


def test_no_pairwise_contractions(monkeypatch):
    monkeypatch.setattr(precision,'_summed_pairwise',lambda product,preferred,*operands:product(*operands,preferred_element_type=preferred))
    probe(users=True)
