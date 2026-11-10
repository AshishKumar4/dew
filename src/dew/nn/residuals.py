"""The names a block gives the values it runs (`jax.ad_checkpoint.checkpoint_name`), which a remat
policy saves, offloads or recomputes by (`dew.nn.backbones.decoder_block.RematPolicy`): each declared
once, here, and named by its member where it is marked."""

import enum


class Residual(enum.StrEnum):
    """A value a block names as it runs (`checkpoint_name`), each after the projection that
    produced it: `kv_proj` is latent attention's fused `kv_b_proj`, `output_gate` a gated attention's
    gate (`gate_proj`) before it scales the context, `attention_output` the
    attention call's result (`scaled_dot_product_attention`'s, local
    attention's), which a remat policy saves rather than replay the kernel
    (`dew.nn.dit`), `context`
    a mixer's output before `o_proj`, the attention kernel's or a recurrence's,
    `scan` a recurrence's (an SSM's or a delta rule's) alone, which a backward
    that drops it recomputes by running the scan again, `conv` a
    linear-attention or SSM mixer's short conv output (`causal_conv1d`,
    its queries, keys and values for a delta rule), `delta_decay`,
    `delta_inverse` and `delta_memory` the rule's cumulated decay, chunk inverse
    and memory (with `w` and the corrected values) the port saves for its
    backward (`dew.nn.kernels.fla_gdn`), `memory_carry` an in-context memory's
    network and momentum as each chunk of its scan hands them on
    (`dew.nn.memory.CARRY`), and the MLP names cover the dense MLP
    and the routed experts alike. A remat policy picks from these; a
    name off the list is a typo that would otherwise recompute silently."""

    Q_PROJ = 'q_proj'
    K_PROJ = 'k_proj'
    V_PROJ = 'v_proj'
    KV_PROJ = 'kv_proj'
    O_PROJ = 'o_proj'
    OUTPUT_GATE = 'output_gate'
    ATTENTION_OUTPUT = 'attention_output'
    CONTEXT = 'context'
    SCAN = 'scan'
    CONV = 'conv'
    DELTA_DECAY = 'delta_decay'
    DELTA_INVERSE = 'delta_inverse'
    DELTA_MEMORY = 'delta_memory'
    MEMORY_CARRY = 'memory_carry'
    GATE_PROJ = 'gate_proj'
    UP_PROJ = 'up_proj'
    DOWN_PROJ = 'down_proj'


QKV_RESIDUALS = (Residual.Q_PROJ, Residual.K_PROJ, Residual.V_PROJ, Residual.KV_PROJ)
"""An attention's input projections' names."""
ATTENTION_RESIDUALS = (*QKV_RESIDUALS, Residual.O_PROJ)
"""An attention's projections' names."""
MLP_RESIDUALS = (Residual.GATE_PROJ, Residual.UP_PROJ, Residual.DOWN_PROJ)
"""A feed-forward's names, the dense MLP's and the routed experts' alike."""
DELTA_RESIDUALS = (Residual.DELTA_DECAY, Residual.DELTA_INVERSE, Residual.DELTA_MEMORY)
"""A delta rule's saved names (`dew.nn.kernels.fla_gdn`)."""
MEMORY_RESIDUALS = (Residual.MEMORY_CARRY,)
"""An in-context memory's carry (`dew.nn.memory.CARRY`)."""
RESIDUALS = tuple(Residual)
"""Every name a block marks (`Residual`): a remat policy picks from these, a name off them a typo that would
otherwise recompute silently."""
ATTENTION_CONTEXT_RESIDUALS = (*ATTENTION_RESIDUALS, Residual.CONTEXT)
"""An attention mixer's names, its projections and its context."""
DELTA_RULE_RESIDUALS = (Residual.CONTEXT, Residual.SCAN, Residual.CONV, *DELTA_RESIDUALS)
"""A delta-rule mixer's names: its scan's output as its context and as a scan, its convolution's output,
the delta rule's three."""
