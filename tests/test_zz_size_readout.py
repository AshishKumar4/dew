"""Sizing the readout sites' declared seeds (`seeds_for`, paired): scratch, never landed."""

import jax.numpy as jnp
import reference_error
import test_decision_packing as packing
import test_decision_readout as readout


def sizing(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
    count = reference_error.seeds_for(reference_fn, draw, label, truth_fn=truth_fn, dew_fn=dew_fn)
    print(f"SIZED {label}: {count} (declared {seeds})", flush=True)


readout.assert_as_exact = sizing
packing.assert_as_exact = sizing


def test_size_tied():
    readout.test_the_tied_readout_is_the_backbones_own_logit_at_each_label()


def test_size_untied():
    readout.test_an_untied_readout_at_its_draw_is_the_tied_one_a_label_of_two_variants_among_them()


def test_size_slots():
    readout.test_a_slot_heads_scores_are_the_releases_law_with_its_query_and_key_trained()


def test_size_packed():
    objective = packing.objective_for("readout")
    packing.assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                                  objective.model.head), 8)
