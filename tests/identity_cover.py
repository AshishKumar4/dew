"""The recurrence template identity test's cover (tests/test_recurrence_mosaic.py's
`test_a_kernels_name_is_its_body_at_every_config_its_gate_group_takes`): the gate groups it walks on each
target, held to their reach by `test_the_identity_tests_cover_is_its_targets_reach`, whose message
regenerates them."""

COVER_PRECISION = "highest"
"""The matmul precision the cover is derived and its shards lowered at (`jax.default_matmul_precision`): the
suite's lane's (tests/lane_environment.py), which an fp32 call's rounding reads (`precision.rounds_products`:
at the default, lightning attention's fp32 forward rounds as its bf16 one, and sm100's cover is a group
fewer), held whatever a run's environment sets."""

IDENTITY_COVER = {
    'sm90': ('qwen3.5-27b-gdn-bfloat16-forward', 'qwen3.5-27b-gdn-bfloat16-forward-variant-0',
             'qwen3.5-27b-gdn-bfloat16-training', 'qwen3.5-27b-gdn-bfloat16-training-variant-two roles',
             'qwen3.5-27b-gdn-bfloat16-forward-packed', 'qwen3.5-27b-gdn-float32-forward',
             'kimi-linear-kda-bfloat16-forward', 'kimi-linear-kda-bfloat16-training',
             'nemotron-h-4b-ssd-bfloat16-forward', 'minimax-text-01-lightning-bfloat16-forward',
             'minimax-text-01-lightning-float32-forward', 'a write gate-bfloat16-forward'),
    'sm100': ('qwen3.5-27b-gdn-bfloat16-forward', 'qwen3.5-27b-gdn-bfloat16-forward-variant-0',
              'qwen3.5-27b-gdn-bfloat16-training', 'qwen3.5-27b-gdn-bfloat16-training-variant-two roles',
              'qwen3.5-27b-gdn-bfloat16-forward-packed', 'qwen3.5-27b-gdn-float32-forward',
              'kimi-linear-kda-bfloat16-forward', 'kimi-linear-kda-bfloat16-training',
              'nemotron-h-4b-ssd-bfloat16-forward', 'minimax-text-01-lightning-bfloat16-forward',
              'minimax-text-01-lightning-float32-forward', 'a write gate-bfloat16-forward'),
    'sm80': ('qwen3.5-27b-gdn-bfloat16-forward', 'qwen3.5-27b-gdn-bfloat16-forward-variant-0',
             'qwen3.5-27b-gdn-bfloat16-training', 'qwen3.5-27b-gdn-bfloat16-forward-packed',
             'kimi-linear-kda-bfloat16-forward', 'kimi-linear-kda-bfloat16-training',
             'nemotron-h-4b-ssd-bfloat16-forward', 'nemotron-h-4b-ssd-float32-forward',
             'minimax-text-01-lightning-bfloat16-forward', 'a write gate-bfloat16-forward'),
    **dict.fromkeys(('sm86', 'sm89'), (
        'qwen3.5-27b-gdn-bfloat16-forward', 'qwen3.5-27b-gdn-bfloat16-forward-variant-0',
        'qwen3.5-27b-gdn-bfloat16-training', 'qwen3.5-27b-gdn-bfloat16-forward-packed',
        'kimi-linear-kda-bfloat16-forward', 'nemotron-h-4b-ssd-bfloat16-forward',
        'nemotron-h-4b-ssd-float32-forward', 'minimax-text-01-lightning-bfloat16-forward',
        'a write gate-bfloat16-forward')),
    'sm120': ('qwen3.5-27b-gdn-bfloat16-forward', 'qwen3.5-27b-gdn-bfloat16-forward-variant-0',
              'qwen3.5-27b-gdn-bfloat16-forward-packed', 'kimi-linear-kda-bfloat16-forward',
              'nemotron-h-4b-ssd-float32-forward', 'minimax-text-01-lightning-bfloat16-forward',
              'minimax-text-01-lightning-bfloat16-training')}
"""The gate groups (test_recurrence_mosaic's `group_name`) the identity test walks on each target (its
`_cover`'s): read here, each shard derives these groups' cases alone, where deriving every group's reach to
find them took 7 to 12 s of every shard (row 9697: 11.5 s of a sm90 shard's 17 to 40 s, its 102 sm90 and
sm100 shards repeating it, jsf-idcost3, jsf-idgroups)."""
