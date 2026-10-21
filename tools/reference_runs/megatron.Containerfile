# The environment megatron_lm.py runs in: Megatron-Core with Transformer Engine on NVIDIA's PyTorch
# container, and the flash-linear-attention kernels Megatron-Core's gated delta net calls.
#   podman build -t dew-megatron -f tools/reference_runs/megatron.Containerfile .
FROM nvcr.io/nvidia/pytorch:26.09-py3
COPY tools/pip.txt /tmp/pip.txt
RUN pip install --no-cache-dir -r /tmp/pip.txt && pip install --no-cache-dir megatron-core==0.19.2 "flash-linear-attention~=0.4.0" einops \
    && pip freeze > /opt/megatron-freeze.txt
