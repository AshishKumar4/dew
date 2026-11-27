"""A run config's record reads back as the config that wrote it.

`to_dict` writes every recorded field, `from_dict` builds the same value
back, a field the record lacks takes its declared default, and a field the
class does not declare is refused.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dew.config import ObjectiveConfig, RunConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.earlier import earlier_run
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.decision.config import DecisionRunConfig
from dew.kernels import Kernels
from dew.quantization import Quantization

ROOT = Path(__file__).resolve().parents[1]


def written(config) -> dict:
    """The record `save` writes, as the JSON file holds it."""
    return json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize("config", ["run", "diffusion", "lm", "jepa", "decision"])
def test_a_run_config_reads_back_as_its_class_from_its_own_record(config):
    """`run.json` names the run's class, so a reader that knows only
    `RunConfig` rebuilds a run of any kind as the kind it is."""
    value = {"run": RunConfig(), "diffusion": DiffusionRunConfig(), "lm": LMRunConfig(),
             "jepa": JepaRunConfig(),
             "decision": DecisionRunConfig()}[config]
    assert RunConfig.read(json.loads(json.dumps(value.record()))) == value


def test_a_record_keeps_what_it_states_and_defaults_what_it_lacks():
    run = DiffusionRunConfig()
    quantized = dataclasses.replace(run, trainer=dataclasses.replace(
        run.trainer, quantization=Quantization(dtype="fp8", patterns=(".*mlp.*",), weight_only=True)))
    assert DiffusionRunConfig.from_dict(written(quantized)) == quantized
    stated = LMRunConfig(objective=ObjectiveConfig("lm", {"ema_decay": 0.99}))
    assert LMRunConfig.from_dict({"objective": {"name": "lm", "fields": {"ema_decay": 0.99}}}) == stated
    assert LMRunConfig.from_dict({}).objective == ObjectiveConfig("lm")


PUBLISHED = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "run.json"
LIVE_PIN = ROOT / "site" / "live" / "container" / "text-to-image"


def changed(written, held, at: str = "") -> list[str]:
    """Where `written` does not keep what `held` states: a key it lacks, a
    list of another length, or another value. What `held` does not state is
    no change."""
    if isinstance(held, dict) and isinstance(written, dict):
        return [place for key, value in held.items() for place in
                (changed(written[key], value, f"{at}/{key}") if key in written else [f"{at}/{key}"])]
    if isinstance(held, list) and isinstance(written, list) and len(written) == len(held):
        return [place for index, pair in enumerate(zip(written, held, strict=True))
                for place in changed(*pair, f"{at}/{index}")]
    return [] if written == held else [at]


@pytest.mark.parametrize("live", [False, pytest.param(True, marks=pytest.mark.network)])
def test_the_published_run_reads_back_as_it_was_written(live):
    """dewml/hybrid-dit-176m's run.json, as the revision the live image pins
    publishes it, is a current record: every field it states loads and writes
    back as it states it. A field added since is written beside them; a
    rename, a removal or a rewrite fails here, and the fix is to re-export the
    published run in the same change, since the site, the quick start and the
    live sampler all load it."""
    held = json.loads(PUBLISHED.read_text())
    if live:
        from huggingface_hub import hf_hub_download

        repo, revision = LIVE_PIN.read_text().strip().split("@")
        published = Path(hf_hub_download(repo, "run.json", revision=revision))
        assert json.loads(published.read_text()) == held
    assert changed(json.loads(json.dumps(RunConfig.read(held).record())), held) == []
    stated = held["fields"]["objective"]["fields"]
    built = RunConfig.read(held).objective.arguments
    assert (built["guidance"], built["steps"]) == (stated["guidance"], stated["steps"])


def test_a_run_saved_with_a_field_its_class_lost_since_reads_as_the_run_it_trained():
    """The published run as an earlier Dew saved it, its source options stating the layer scan they lost
    since (row 4074: a decoder's structure decides it), reads as the run it trained and writes back as
    today's record, the field left out (`dew.config.earlier.earlier_removed`), as `RunConfig.load` reads a
    saved run."""
    held = json.loads(PUBLISHED.read_text())
    source = {**held["fields"]["source"], "scan_layers": False}
    earlier = {**held, "fields": {**held["fields"], "source": source}}
    assert changed(json.loads(json.dumps(RunConfig.read(earlier_run(earlier)).record())), held) == []


@pytest.mark.parametrize("kernels", [{}, {"attention": "xla"}])
def test_a_run_whose_model_named_its_attention_kernel_runs_it_as_the_runs_own(kernels):
    """The published run as an earlier Dew saved it, its model naming its attention kernel (row 11938: the
    run's since, `Diagnostics.kernels`), reads as a run choosing that kernel for every attention it
    traces, its model without the field, unless the run named one of its own, which stands, as
    `RunConfig.load` reads a saved run (`earlier_run`)."""
    held = json.loads(PUBLISHED.read_text())
    model = held["fields"]["model"]
    diagnostics = {**held["fields"]["diagnostics"], "kernels": kernels}
    earlier = {**held, "fields": {**held["fields"], "diagnostics": diagnostics, "model": {
        **model, "fields": {**model["fields"], "attention_impl": "reference"}}}}
    run = RunConfig.read(earlier_run(earlier))
    assert run.diagnostics.kernels == Kernels(**{"attention": "reference", **kernels})
    assert "attention_impl" not in json.loads(json.dumps(run.record()))["fields"]["model"]["fields"]


@dataclasses.dataclass(frozen=True)
class _TaggedKernels(Kernels):
    """Kernel choices of a subclass, which a run's record tags with its class."""


@pytest.mark.parametrize("tagged", [False, True], ids=["plain", "tagged"])
@pytest.mark.parametrize("generated", [None, "generated", "written"])
def test_a_run_saved_with_a_retired_kernel_choice_reads_as_the_run_it_trained(tmp_path, generated, tagged):
    """The published run as a Dew of rows 3272/6109's opt-in seam saved it, its kernel choices stating the
    retired `generated` (written None where unset, as a record of its kernels writes each), in a plain
    record or a subclass's tagged one, loads as today's run of the written math, the choice left out
    (`_records.saved_fields`), its other choices and its class kept."""
    held = json.loads(PUBLISHED.read_text())
    choices = {**held["fields"]["diagnostics"]["kernels"], "generated": generated, "attention": "xla"}
    name = f"{_TaggedKernels.__module__}:{_TaggedKernels.__qualname__}"
    kernels = {"class": name, "fields": choices} if tagged else choices
    diagnostics = {**held["fields"]["diagnostics"], "kernels": kernels}
    (tmp_path / "run.json").write_text(json.dumps({**held, "fields": {**held["fields"],
                                                                      "diagnostics": diagnostics}}))
    run = RunConfig.load(str(tmp_path))
    assert run.diagnostics.kernels == (_TaggedKernels if tagged else Kernels)(attention="xla")
    assert type(run.diagnostics.kernels) is (_TaggedKernels if tagged else Kernels)


@dataclasses.dataclass(frozen=True)
class _ChildRun(RunConfig):
    """A run of a kind of its own, which its parent's record tags with its class."""


@dataclasses.dataclass(frozen=True)
class _ParentRun(RunConfig):
    """A run of a kind that holds other runs as fields, and data of its own."""

    child: RunConfig = dataclasses.field(default_factory=RunConfig)
    children: tuple[RunConfig, ...] = ()
    named: dict[str, RunConfig] = dataclasses.field(default_factory=dict)
    either: RunConfig | str = ""
    payload: dict[str, object] = dataclasses.field(default_factory=dict)


def _with_generated(record: dict) -> dict:
    """`record`'s diagnostics' kernel choices, untagged or tagged, also naming the retired `generated`."""
    tagged = "class" in record and "fields" in record
    fields = record["fields"] if tagged else record
    diagnostics = fields["diagnostics"]
    kernels = diagnostics["kernels"]
    named = "class" in kernels
    choices = {**(kernels["fields"] if named else kernels), "generated": "generated"}
    retired = {**fields, "diagnostics": {**diagnostics,
                                         "kernels": {**kernels, "fields": choices} if named else choices}}
    return {**record, "fields": retired} if tagged else retired


@pytest.mark.parametrize("tagged", [False, True], ids=["plain", "tagged"])
def test_a_nested_run_saved_with_a_retired_kernel_choice_loads(tmp_path, tagged):
    """A saved run of a kind holding runs, its own, its child's (of the field's class, untagged, or a
    subclass, tagged), a tuple's, a typed mapping's and a mixed union's kernel choices naming the retired
    `generated`, each kernels record plain or a subclass's tagged one, loads (`RunConfig.load`), each run as
    it saved it; a payload of data that looks like a kernels record stays as written."""
    kernels = (_TaggedKernels if tagged else Kernels)(attention="xla")
    def run(kind):
        return kind(diagnostics=dataclasses.replace(RunConfig().diagnostics, kernels=kernels))

    payload = {"class": "dew.kernels:Kernels", "fields": {"generated": "keep"}}
    parent = dataclasses.replace(run(_ParentRun), child=run(_ChildRun if tagged else RunConfig),
                                 children=(run(RunConfig), run(_ChildRun)), named={"one": run(RunConfig)},
                                 either=run(_ChildRun), payload=payload)
    saved = json.loads(Path(parent.save(str(tmp_path))).read_text())
    fields = _with_generated(saved["fields"])
    fields = {**fields, "child": _with_generated(fields["child"]),
              "children": [_with_generated(each) for each in fields["children"]],
              "named": {"one": _with_generated(fields["named"]["one"])},
              "either": _with_generated(fields["either"])}
    (tmp_path / "run.json").write_text(json.dumps({**saved, "fields": fields}))
    loaded = _ParentRun.load(str(tmp_path))
    assert loaded == parent
    assert loaded.payload == payload


@dataclasses.dataclass(frozen=True)
class _GeneratedKernels(Kernels):
    """Kernel choices of a subclass that declares a `generated` field of its own."""

    generated: str = "its own"


def test_a_kernels_subclass_that_declares_the_retired_name_keeps_it(tmp_path):
    """A run whose kernel choices are a subclass declaring `generated` as its own field saves and loads it
    as it is: only a name its class does not declare is retired."""
    run = RunConfig(diagnostics=dataclasses.replace(RunConfig().diagnostics,
                                                    kernels=_GeneratedKernels(generated="kept")))
    assert RunConfig.load(run.save(str(tmp_path))) == run


def test_a_kernels_subclass_field_of_the_retired_name_runs_no_choice():
    """A run whose kernel choices are a subclass declaring `generated` as its own field traces with the
    choices `Kernels` declares alone (`Kernels.named`): its own field is no kernel choice."""
    from dew.nn.backbones.decoder_stages import scanned

    run = RunConfig(diagnostics=dataclasses.replace(
        RunConfig().diagnostics, kernels=_GeneratedKernels(layer_scan="plain", generated="kept")))
    with run.traced():
        assert not scanned()


def test_a_run_of_before_its_diagnostics_loses_the_retired_choice_it_moves(tmp_path):
    """A saved run's `kernels` of before its diagnostics held them, naming the retired choice, loads as
    today's run's diagnostics' kernel choices without it (`earlier_run`)."""
    held = json.loads(PUBLISHED.read_text())
    fields = {key: value for key, value in held["fields"].items() if key != "diagnostics"}
    record = {**held, "fields": {**fields, "kernels": {"attention": "xla", "generated": "generated"}}}
    (tmp_path / "run.json").write_text(json.dumps(record))
    assert RunConfig.load(str(tmp_path)).diagnostics.kernels == Kernels(attention="xla")


def test_a_live_run_record_naming_the_retired_choice_is_refused():
    """A run record read live (`RunConfig.read`: a recipe's, as `dew train recipe.py` rebuilds it) holds
    each of its fields: one naming the retired choice is refused, as only a saved run's load reads it
    without (`RunConfig.load`)."""
    from dew import _records
    from dew.registry import from_record

    record = json.loads(json.dumps(RunConfig().record()))
    assert RunConfig.read(record) == RunConfig()
    kernels = record["fields"]["diagnostics"]["kernels"]
    record["fields"]["diagnostics"]["kernels"] = {**kernels, "generated": "generated"}
    with pytest.raises(ValueError):
        RunConfig.read(record)
    # Within a saved run's reading too (a saved run's decoder reading a live record), and the saved
    # reading's mode stands past it.
    with _records.reading_saved():
        with pytest.raises(ValueError):
            RunConfig.read(record)
        saved = {**kernels, "generated": "generated"}
        assert from_record(Kernels, saved) == from_record(Kernels, kernels)


@dataclasses.dataclass(frozen=True)
class _UnrecordedKernels(Kernels):
    """Kernel choices of a subclass holding a `generated` attribute its record does not carry."""

    generated: str = dataclasses.field(default="its own", metadata={"record": False})


class _ForwardingKernels(Kernels):
    """Kernel choices of a subclass whose constructor forwards its keywords to the base's."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.mark.parametrize("kind", [_UnrecordedKernels, _ForwardingKernels], ids=["unrecorded", "forwarding"])
def test_a_kernels_subclass_its_decoder_reads_without_the_retired_name_loses_it(tmp_path, kind):
    """A saved run whose kernel choices are a subclass that records no `generated` (an attribute its record
    leaves out, or a constructor forwarding to the base's) and whose record names the retired choice loads
    without it: only a name its decoder accepts is kept."""
    run = RunConfig(diagnostics=dataclasses.replace(RunConfig().diagnostics, kernels=kind(attention="xla")))
    saved = json.loads(Path(run.save(str(tmp_path))).read_text())
    kernels = saved["fields"]["diagnostics"]["kernels"]
    kernels["fields"] = {**kernels["fields"], "generated": "generated"}
    (tmp_path / "run.json").write_text(json.dumps(saved))
    loaded = RunConfig.load(str(tmp_path)).diagnostics.kernels
    assert type(loaded) is kind and loaded.attention == "xla"


def test_a_run_resumed_from_a_record_pinning_the_retired_choice_resumes(tmp_path):
    """A run resuming one whose saved record pins the retired choice (`check_resume`, `resumed`'s reading
    of the saved run) reads the record as a saved one: the same run, resumable."""
    pinned = Kernels(attention="xla")
    run = RunConfig(diagnostics=dataclasses.replace(RunConfig().diagnostics, kernels=pinned))
    saved = json.loads(Path(run.save(str(tmp_path))).read_text())
    saved["fields"]["diagnostics"]["kernels"] = {"attention": "xla", "generated": "generated"}
    run.check_resume(earlier_run(saved))


def test_a_saved_decision_runs_untagged_segment_reads_as_the_one_class_it_names_as_written():
    """A saved decision run's untagged row segment `{"stream": "context"}` reads as the schema it is, the one
    class of the union whose fields it names as written: a class's retired field never makes another
    class read it too (`registry._unnamed`)."""
    from dew.decision.layout import Schema

    saved = {"class": "decision", "fields": {"rows": {"segments": [{"stream": "context"}]}}}
    run = RunConfig.read(earlier_run(saved))
    assert isinstance(run, DecisionRunConfig) and run.rows.segments == (Schema(stream="context"),)


def test_a_live_configuration_naming_the_retired_choice_is_refused():
    """Outside a saved record's reading (`_records.reading_saved`) the retired choice is no field: a record
    read live naming it is refused, one without it read."""
    from dew.registry import from_record

    with pytest.raises(ValueError):
        from_record(Kernels, {"attention": "xla", "generated": "written"})
    assert from_record(Kernels, {"attention": "xla"}) == Kernels(attention="xla")


def test_a_saved_runs_kernel_pins_and_a_releases_leave_the_retired_choice_out():
    """A saved run's step's kernel pins (`SavedRun.kernels`) and a release's (`ReleaseTask.read`, its
    `dew_release.json`'s task) naming the retired choice read as today's, the choice left out."""
    from dew.inference.tasks import SavedRun
    from dew.interop.release import ReleaseTask

    pins = {"attention": "reference", "generated": "written"}
    assert SavedRun("", {"kernels": pins}, 0).kernels == (("attention", "reference"),)
    assert ReleaseTask.read({"kernels": pins}).kernels == {"attention": "reference"}


def test_a_run_saved_scanning_its_stack_resumes_scanned():
    """A run an earlier Dew saved with its stack scanned (`scan_layers=True`, its source options' or its
    model's until row 4074) reads as a run whose kernel choices scan it (`Diagnostics.kernels`'
    `layer_scan`), every program it traces within `RunConfig.traced` scanning its stacks, a host-streamed
    resume's among them, where it resumed a layer at a time and its host placement raised (the API02 review on
    API02a); its own choice, where it names one, stands."""
    from dew.nn.backbones.decoder_stages import scanned
    from dew.nn.kernels.generation import chosen

    held = json.loads(PUBLISHED.read_text())
    source = {**held["fields"]["source"], "scan_layers": True}
    run = RunConfig.read(earlier_run({**held, "fields": {**held["fields"], "source": source}}))
    assert run.diagnostics.kernels == Kernels(layer_scan="scanned")
    # Within a block that runs its stacks unrolled, the run's own choice scans them, and the block's stands
    # again past it: the scanned default would hide either.
    with chosen(layer_scan="plain"):
        with run.traced():
            assert scanned()
        assert not scanned()
    diagnostics = {**held["fields"]["diagnostics"], "kernels": {"layer_scan": "plain"}}
    named = {**held, "fields": {**held["fields"], "source": source, "diagnostics": diagnostics}}
    assert RunConfig.read(earlier_run(named)).diagnostics.kernels == Kernels(layer_scan="plain")


WRAPPERS = [("diffusion_gemma", "text"), ("dew.nn.multimodal:MultimodalTransformer", "language_model")]
"""Models that wrap a decoder in a field they declare, by alias and by import path."""


PINS = {"scan_layers": (True, "layer_scan", "scanned"),
        "attention_impl": ("reference", "attention", "reference")}
"""Each field an earlier decoder pinned its execution by (rows 4074, 11938): its pinning value, the kernel
choice it reads as now and the kernel it names there."""


@pytest.mark.parametrize("recorded", ["by its fields", "by its class"])
@pytest.mark.parametrize(("wrapper", "field"), WRAPPERS)
@pytest.mark.parametrize("pin", sorted(PINS))
def test_a_run_saved_pinning_a_wrapped_decoders_execution_resumes_so(wrapper, field, recorded, pin):
    """A run an earlier Dew saved whose model wraps its decoder (`DiffusionGemma.text`,
    `MultimodalTransformer.language_model`), the decoder pinning its execution (`scan_layers=True` until row
    4074, `attention_impl` until row 11938), recorded by its fields alone or by its class, reads as a run
    whose kernel choices run it so, where the root model alone was read and the wrapped decoder's field left
    out unread (the API02 review); the run's own choice, where it names one, stands."""
    from dew.config.earlier import earlier_attention_kernel, earlier_layer_scan

    value, choice, kernel = PINS[pin]
    read = earlier_layer_scan if pin == "scan_layers" else earlier_attention_kernel

    def saved(decoder: dict, **run) -> dict:
        held = decoder if recorded == "by its fields" else {"class": "causal_transformer", "fields": decoder}
        model = {"name": wrapper, "fields": {field: held}}
        return {"class": "dew.config:RunConfig", "fields": {"model": model, **run}}

    assert read(saved({"vocab_size": 16, pin: value}))["fields"]["kernels"] == {choice: kernel}
    own = {choice: "xla" if pin == "attention_impl" else "plain"}
    assert read(saved({"vocab_size": 16, pin: value}, kernels=own))["fields"]["kernels"] == own
    plain = saved({"vocab_size": 16})
    assert read(plain) == plain

@pytest.mark.parametrize("pinned", ["mosaic", "xla", "cutlass"])
def test_a_run_saved_pinning_the_grouped_products_resumes_syrk_pinned_and_its_fp32_loss_chunked(pinned):
    """The published run as an earlier Dew saved it (before rows 12457 and 12485: no `kernels_split` beside
    its fields), its grouped products' kernel pinned, reads as a run pinning SYRK's alike where the pin was
    `mosaic` or `xla`, which chose SYRK's kernel then, and none for another (`cutlass`); traced within the
    run (`RunConfig.traced`), its fp32 tiled loss, which the pin left on the chunked program, still takes it
    rather than refusing a pin of the fused loss it never had. A run's own SYRK pin stands."""
    import jax.numpy as jnp

    from dew.nn.kernels import generation
    from dew.nn.protocols import OutputTable
    from dew.objectives.lm import chunked

    held = json.loads(PUBLISHED.read_text())
    assert "kernels_split" not in held

    def saved(kernels: dict) -> RunConfig:
        diagnostics = {**held["fields"]["diagnostics"], "kernels": kernels}
        return RunConfig.read(earlier_run({**held, "fields": {**held["fields"], "diagnostics": diagnostics}}))

    run = saved({"grouped_matmul": pinned})
    carried = {"syrk": pinned} if pinned in ("mosaic", "xla") else {}
    assert run.diagnostics.kernels == Kernels(grouped_matmul=pinned, **carried)
    hidden = jnp.ones((64, 64), jnp.float32)
    head = OutputTable(jnp.ones((128, 64), jnp.float32), vocab_major=True)
    with run.traced(), generation.assumed("sm90"), generation.routes() as routes:
        chunked.chunked_cross_entropy(hidden, head, jnp.zeros((64,), jnp.int32), chunks=2, tile=(64, 64),
                                      predict=False)
    assert not any(route.choice == "linear_ce" and route.kernel == "mosaic" for route in routes), routes
    assert saved({"grouped_matmul": "mosaic", "syrk": "xla"}).diagnostics.kernels == Kernels(
        grouped_matmul="mosaic", syrk="xla")


LAYOUT = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "layout.json"


def published_layout() -> tuple[dict, dict]:
    """The published run's checkpoint record and its variables as abstract
    arrays, nested as the checkpoint stores them (`layout.json`)."""
    import jax
    from flax.traverse_util import unflatten_dict

    from dew.inference.tasks import earlier_shapes

    fixture = json.loads(LAYOUT.read_text())
    variables = unflatten_dict({path: jax.ShapeDtypeStruct(tuple(shape), dtype)
                                for path, (shape, dtype) in fixture["variables"].items()}, sep="/")
    # The published run's record, as a run loader reads it (`SavedRun.pinned`).
    return earlier_shapes(fixture["artifact"]), variables


def test_the_published_runs_weights_are_the_ones_its_models_read():
    """dewml/hybrid-dit-176m's checkpoint layout (layout.json: each array's
    path, shape and dtype, and the model record beside them) is what the
    run's denoiser and autoencoder read: both trace over arrays of that
    layout. A parameter renamed, moved or reshaped in either fails here, and
    the fix is to republish the run in the same change, since the site, the
    quick start and the live sampler all load it."""
    import jax
    import jax.numpy as jnp

    from dew.inference.tasks import saved_model
    from dew.nn.autoencoders import AutoEncoder
    from dew.nn.dit import TextContext

    artifact, variables = published_layout()
    model = saved_model(artifact, None).build()
    autoencoder = AutoEncoder.bound(artifact["autoencoder"], params=variables["autoencoder"])
    sample = tuple(artifact["inputs"]["sample"]["shape"])
    latents = jax.ShapeDtypeStruct((1, *autoencoder.latent_shape(sample)), jnp.float32)
    # CLIP ViT-L/14's 77 tokens of 768 features, the published run's text condition.
    text = TextContext(jax.ShapeDtypeStruct((1, 77, 768), jnp.float32),
                       jax.ShapeDtypeStruct((1, 77), jnp.float32))
    denoiser = {name: tree for name, tree in variables.items() if name not in ("encoders", "autoencoder")}
    predicted = jax.eval_shape(lambda held, x, t, text: model.apply(held, x, t, textcontext=text),
                               denoiser, latents, jax.ShapeDtypeStruct((1,), jnp.float32), text)
    assert predicted.shape == latents.shape
    assert jax.eval_shape(autoencoder.decode, variables["autoencoder"], latents).shape == (1, *sample)


@pytest.mark.network
def test_the_live_run_holds_the_committed_layout_and_samples_an_image():
    """The revision the live image pins holds exactly the committed layout,
    and loads and samples an image, its text tower included, on one device
    of the lane's backend: the CPU unless the run sets JAX_PLATFORMS, as the
    live sampler runs it. Loaded over the lane's eight host devices, the
    weights would be held once per device, 10.9 GiB against 3.3."""
    import jax
    import numpy as np
    from flax.core import unfreeze

    import dew
    from dew.inference import TextToImage
    from dew.training import MeshSpec

    repo, revision = LIVE_PIN.read_text().strip().split("@")
    pipe = dew.pipeline(repo, revision=revision, mesh=MeshSpec().build(jax.devices()[:1]))
    assert isinstance(pipe, TextToImage)
    artifact, expected = published_layout()
    held = unfreeze(pipe.variables)
    assert jax.tree.structure(held) == jax.tree.structure(expected)
    def same(held, wanted) -> bool:
        return (held.shape, held.dtype) == (wanted.shape, wanted.dtype)

    assert jax.tree.all(jax.tree.map(same, held, expected))
    images = pipe(["a watercolor painting of a mountain lake at sunrise"], steps=1, key=0).host().images
    assert images.shape == (1, *artifact["inputs"]["sample"]["shape"])
    assert np.all(np.isfinite(np.asarray(images, np.float32)))
