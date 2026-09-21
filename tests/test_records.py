"""The config readers every source translation shares refuse what they cannot read."""

import math

import pytest

from dew import records


def test_integer_reads_a_width_and_refuses_the_flag_python_calls_an_int():
    assert records.integer(4096, 'hidden_size') == 4096
    with pytest.raises(ValueError, match=r"tie_word_embeddings=True: this field is an integer"):
        records.integer(True, 'tie_word_embeddings')


def test_number_reads_an_integer_as_a_real_and_refuses_a_flag():
    assert records.number(1, 'rope_theta') == 1.0
    assert records.number(1e-5, 'norm_eps') == pytest.approx(1e-5)
    with pytest.raises(ValueError, match=r"norm_eps=False: this field is a finite number"):
        records.number(False, 'norm_eps')


def test_number_reads_the_float_record_transformers_writes_a_non_finite_bound_as():
    assert records.number({'__float__': 'Infinity'}, 'time_step_limit') == math.inf
    assert records.number({'__float__': '-Infinity'}, 'time_step_limit') == -math.inf
    # A bare infinity is not a spelling a config file uses, so it stays refused.
    with pytest.raises(ValueError, match='this field is a finite number'):
        records.number(math.inf, 'time_step_limit')


def test_boolean_and_text_take_only_their_own_type():
    assert records.boolean(False, 'use_cache') is False
    assert records.text('silu', 'hidden_act') == 'silu'
    with pytest.raises(ValueError, match=r"use_cache=1: this field is a boolean"):
        records.boolean(1, 'use_cache')
    with pytest.raises(ValueError, match=r"hidden_act=None: this field is a string"):
        records.text(None, 'hidden_act')


def test_record_takes_a_string_keyed_section_and_names_what_it_refused():
    section = {'hidden_size': 8}
    assert records.record(section, 'text_config') is section
    with pytest.raises(ValueError, match='this field is a record of named fields'):
        records.record({1: 'one'}, 'text_config')
    with pytest.raises(ValueError, match=r"text_config=\[\]: this field is a record of named fields"):
        records.record([], 'text_config')


def test_lists_read_element_by_element_under_the_key_that_carried_them():
    assert records.strings(['LlamaForCausalLM'], 'architectures') == ('LlamaForCausalLM',)
    assert records.integers([0, 2], 'mlp_only_layers') == (0, 2)
    # A string is a sequence of strings to Python and one name to a config.
    with pytest.raises(ValueError, match='this field is a list of strings'):
        records.strings('LlamaForCausalLM', 'architectures')
    with pytest.raises(ValueError, match=r"mlp_only_layers=True: this field is an integer"):
        records.integers([True], 'mlp_only_layers')


def test_json_value_carries_scalars_lists_and_sections_and_names_an_unknown_type():
    assert records.json_value({'a': [1, None, 'x'], 'b': {'c': 2.5}}, 'generation_config') == {
        'a': [1, None, 'x'], 'b': {'c': 2.5}}
    with pytest.raises(ValueError, match='is not a value a JSON config carries'):
        records.json_value(object(), 'generation_config')


@pytest.mark.parametrize('reader', [records.integer, records.number, records.boolean,
                                    records.text, records.record, records.strings,
                                    records.integers, records.json_value])
def test_every_reader_names_the_key_and_shows_the_value_it_refused(reader):
    with pytest.raises(ValueError) as raised:
        reader(bytearray(b'\x00'), 'quantization_config')
    assert str(raised.value).startswith("quantization_config=bytearray(b'\\x00'): ")
