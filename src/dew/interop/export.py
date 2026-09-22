"""Write a trained run to the Hugging Face layout its model family publishes.

`export_run` loads the run's `run.json` and checkpoint, rebuilds the model,
and hands it to that family's own writer. An exported run and an exported
source therefore leave the same files behind. The run's tokenizer name is
written beside the weights, so the export loads without the run directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

DECODER_FAMILIES = "a decoder of a registered family (CausalTransformer) or a DiffusionGemma"
"""Names the model kinds that have a published layout to export to."""


def export_run(run_dir: str, destination: str | Path, *, ema: bool = True,
               step: int | None = None) -> None:
    """Write the run in `run_dir` to `destination` in its family's layout.

    The run loads the way `dew.pipeline` loads it: `run.json` for the model
    record, the latest checkpoint or `step` for the weights, `ema` for the
    averaged copy where the run kept one. The model that comes back decides
    the layout, and a model with no published layout is refused by name.

    The result is a Hugging Face directory: `load_pretrained` reads it back,
    and so does transformers for a family it knows.
    """
    from dew.inference.pipeline import pipeline
    from dew.inference.tasks import run_record
    from dew.sampling.pipelines import TextToImage

    record = run_record(str(run_dir))
    task = pipeline(str(run_dir), ema=ema, step=step)
    # An image task keeps the objective's whole tree under `params`, which is
    # the same weights under the name that task gives them.
    variables = task.params if isinstance(task, TextToImage) else task.variables
    _export_model(task.model, variables, Path(destination), record)


def _export_model(model, variables, destination: Path, record: Mapping[str, object]) -> None:
    """Write `model` and `variables` to `destination` through its family's writer.

    The family is chosen before any field is read out of `record`. A model with
    no published layout is therefore refused by type, not by a missing field.
    """
    from dew.interop.hf_decoders import save_export_assets, save_pretrained_decoder
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.records import record as named_fields, text as named

    if isinstance(model, DiffusionGemma):
        from dew.interop import diffusion_gemma
        from dew.interop.safetensors_io import save_hf_layout

        # The block-diffusion export is the decoder export under the
        # reference's own encoder/decoder names, and it reads the published
        # config the run recorded rather than deriving one, because the
        # canvas and the sampler settings are the file's, not the model's.
        config = named_fields(named_fields(record["model"], "model")["config"], "model config")
        save_hf_layout(diffusion_gemma.export_weights(model, variables, config),
                       dict(config), destination)
        save_export_assets(destination, tokenizer=named(record["tokenizer"], "tokenizer"))
        return
    if isinstance(model, CausalTransformer):
        save_pretrained_decoder(model, variables, destination,
                                tokenizer=named(record["tokenizer"], "tokenizer"))
        return
    raise ValueError(
        f"{type(model).__name__} has no published layout to export to; a run exports "
        f"when its model is {DECODER_FAMILIES}. A latent diffusion run's denoiser is "
        f"bound to a published file only by the load that read one, and a run records "
        f"no such bindings, so publish it with `Pretrained.save` on the source it was "
        f"trained from")
