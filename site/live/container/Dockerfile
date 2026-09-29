# The live kernel: Dew at one pinned commit, JAX on the CPU, and the relay in
# server.py. Everything is installed here, at build time; the running container
# has no network access.
FROM python:3.13-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/live/
RUN pip install --no-cache-dir -r /opt/live/requirements.txt

# The landing page's text-to-image model, a Dew run on the Hugging Face Hub,
# pinned in text-to-image as <repo>@<revision>. Its own layer, before Dew, so a
# new Dew commit does not download it again.
ENV HF_HOME=/opt/hf
COPY text-to-image /opt/live/
RUN python -c "from huggingface_hub import snapshot_download; repo, rev = open('/opt/live/text-to-image').read().strip().split('@'); snapshot_download(repo_id=repo, revision=rev, local_dir='/opt/models/text-to-image')"

# dew-commit holds the full SHA of the Dew commit to install; deploy.mjs writes it.
COPY dew-commit /opt/live/
RUN grep -Eq '^[0-9a-f]{40}$' /opt/live/dew-commit \
	&& pip install --no-cache-dir "dewml @ git+https://github.com/AshishKumar4/dew@$(cat /opt/live/dew-commit)"

# Cells run as this user, which cannot signal or trace the server.
RUN useradd --create-home --uid 1001 --shell /usr/sbin/nologin kernel

# The page's sampling cells, run once here so the tokenizer and VAE configs are
# in HF_HOME and the default cell's compiled programs in the XLA cache. The
# cache's programs target AVX2, which every host this runs on has.
ENV JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/opt/xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
	JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 XLA_FLAGS=--xla_cpu_max_isa=AVX2
COPY warm.py sampler_setup.py sampler.py /opt/live/
RUN python /opt/live/warm.py && chown -R kernel:kernel /opt/xla /opt/hf

COPY server.py /opt/live/server.py
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
EXPOSE 8888
CMD ["python", "/opt/live/server.py"]
