"""torchaudio's MelSpectrogram at MiMo's settings (24 kHz, a 960-sample window and FFT, a 240-sample hop,
128 mels, magnitudes), its log floored at 1e-7, as MiMo-Audio-Tokenizer's `mel_spectrogram` computes it,
run on drawn waveforms in float32 and float64: the reference arm of tools/venv_reference.py's `mimo-mel`
site (its protocol tools/reference_worker.py), run in an environment of torch 2.11 and torchaudio 2.11
(torchaudio stops at torch 2.11), on the CPU.

    python tools/torchaudio_mel_worker.py mimo-mel    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.waveform` `[samples]`. OUT gets `<draw>.reference.mel` and `<draw>.truth.mel`,
its log mel frames `[frames, 128]` in float32 and float64, the transform's filters and window built in the
dtype it computes in.
"""

import sys

import numpy as np
import torch
from reference_worker import serve
from torch_precision import default_dtype


def mel(waveform: np.ndarray, dtype) -> np.ndarray:
    """The log mel frames of `waveform` at `dtype`."""
    from torchaudio.transforms import MelSpectrogram

    with default_dtype(dtype):
        transform = MelSpectrogram(sample_rate=24000, n_fft=960, hop_length=240, win_length=960, f_min=0,
                                   f_max=None, n_mels=128, power=1.0, center=True)
    spec = transform(torch.from_numpy(waveform).to(dtype)[None, :])
    return torch.log(torch.clip(spec, min=1e-7)).squeeze(0).T.numpy()


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's log mel frames, in float32 and float64."""
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        out[f"{name}.reference.mel"] = mel(cases[f"{name}.waveform"], torch.float32)
        out[f"{name}.truth.mel"] = mel(cases[f"{name}.waveform"], torch.float64)
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["mimo-mel"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
