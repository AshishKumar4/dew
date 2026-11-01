"""The releases whose own modeling code is a Dew family's reference, by family: the Hub repo, the commit
Dew is held to it at, and the `auto_map` its config names that code by. A family exports with its
release's code (`DecoderFamily.release_code`); tools/remote_code_reference.py records the family's
fixture from it, reading this module by its path in the release's own environment, so it holds constants
alone and imports nothing."""

LFM3 = ("LiquidAI/LFM3-1.7B", "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        {"AutoConfig": "configuration_lfm3.Lfm3Config", "AutoModel": "modeling_lfm3.Lfm3Model",
         "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM"})
