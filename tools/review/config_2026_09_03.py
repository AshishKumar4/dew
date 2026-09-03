"""Reproductions for config/loading findings 2, 14, 16."""
import os, warnings, io, contextlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")

def verdict(tag, ok, detail):
    print(f"[{tag}] {'CONFIRMED' if ok else 'NOT REPRODUCED'}: {detail}")

model_cfg = {"patch_size": 4, "emb_features": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 1}

# 16. build_model drops unknown keys with a print instead of raising.
from dew.registry import build_model
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m = build_model("simple_dit", {**model_cfg, "num_layerss": 99, "emb_feature": 4096})
verdict("16 dropped-keys", m.num_layers == 1 and "Dropping" in buf.getvalue(),
        f"built num_layers={m.num_layers} emb_features={m.emb_features}; stdout: {buf.getvalue().strip()!r}")

# 2. parse_config rebuilds the flow preset without the training shift.
from dew.sampling.loading import parse_config
from dew.diffusion.schedules import FlowMatchingScheduler
conf = {
    "model": model_cfg, "architecture": "simple_dit", "noise_schedule": "flow",
    "arguments": {"noise_schedule": "flow", "flow_shift": 3.0, "image_size": 16},
    "run_config": {"flow_shift": 3.0},
    "input_config": {"sample_data_key": "image", "sample_data_shape": (16, 16, 3), "conditions": []},
}
with contextlib.redirect_stdout(io.StringIO()):
    result = parse_config(conf)
sched = result["noise_schedule"]
verdict("2 flow-shift", isinstance(sched, FlowMatchingScheduler) and sched.shift == 1.0,
        f"config says flow_shift=3.0, rebuilt sampling schedule has shift={sched.shift}")

# 14. parse_config silences warnings process-wide.
first = warnings.filters[0]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("default")
    # re-run parse_config inside the catch block: it installs an 'ignore' filter on top
    with contextlib.redirect_stdout(io.StringIO()):
        parse_config(conf)
    warnings.warn("this should be visible", UserWarning)
verdict("14 warnings-ignored", first[0] == "ignore" and len(caught) == 0,
        f"warnings.filters[0]={first[:2]} after parse_config; a UserWarning raised afterwards was recorded {len(caught)} times")
