"""The config readers every source translation shares refuse what they cannot read."""

import math

import pytest

from dew import records


def test_integer_reads_a_width_and_refuses_the_flag_python_calls_an_int():
    assert records.integer(4096, 'hidden_size') == 4096
    with pytest.raises(ValueError):
        records.integer(value=True, key='tie_word_embeddings')


def test_number_reads_an_integer_as_a_real_and_refuses_a_flag():
    assert records.number(1, 'rope_theta') == 1.0
    assert records.number(1e-5, 'norm_eps') == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        records.number(value=False, key='norm_eps')


def test_number_reads_the_float_record_transformers_writes_a_non_finite_bound_as():
    assert records.number({'__float__': 'Infinity'}, 'time_step_limit') == math.inf
    assert records.number({'__float__': '-Infinity'}, 'time_step_limit') == -math.inf
    # A bare infinity is not a spelling a config file uses, so it stays refused, and so is the NaN
    # transformers records, which no number field holds, and any other mapping.
    for refused in (math.inf, {'__float__': 'NaN'}, {'__float__': 'nan'}, {'__float__': 'inf'},
                    {'__float__': 'Infinity', 'note': 1}, {'value': 1.0}):
        with pytest.raises(ValueError):
            records.number(refused, 'time_step_limit')


def test_boolean_and_text_take_only_their_own_type():
    assert records.boolean(value=False, key='use_cache') is False
    assert records.text('silu', 'hidden_act') == 'silu'
    with pytest.raises(ValueError):
        records.boolean(1, 'use_cache')
    with pytest.raises(ValueError):
        records.text(None, 'hidden_act')


def test_record_takes_a_string_keyed_section_and_names_what_it_refused():
    section = {'hidden_size': 8}
    assert records.record(section, 'text_config') is section
    with pytest.raises(ValueError):
        records.record({1: 'one'}, 'text_config')
    with pytest.raises(ValueError):
        records.record([], 'text_config')


def test_lists_read_element_by_element_under_the_key_that_carried_them():
    assert records.strings(['LlamaForCausalLM'], 'architectures') == ('LlamaForCausalLM',)
    assert records.integers([0, 2], 'mlp_only_layers') == (0, 2)
    # A string is a sequence of strings to Python and one name to a config.
    with pytest.raises(ValueError):
        records.strings('LlamaForCausalLM', 'architectures')
    with pytest.raises(ValueError):
        records.integers([True], 'mlp_only_layers')


def test_json_value_carries_scalars_lists_and_sections_and_names_an_unknown_type():
    assert records.json_value({'a': [1, None, 'x'], 'b': {'c': 2.5}}, 'generation_config') == {
        'a': [1, None, 'x'], 'b': {'c': 2.5}}
    with pytest.raises(ValueError):
        records.json_value(object(), 'generation_config')

