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
# pinned in text-to-image as <repo>@<revision>, in the hub cache with `main`
# naming that revision, so the offline kernel's from_pretrained(repo) reads it.
# Its own layer, before Dew, so a new Dew commit does not download it again.
ENV HF_HOME=/opt/hf
COPY text-to-image /opt/live/
RUN python -c "from pathlib import Path; from huggingface_hub import snapshot_download; repo, rev = open('/opt/live/text-to-image').read().strip().split('@'); refs = Path(snapshot_download(repo_id=repo, revision=rev)).parent.parent / 'refs'; refs.mkdir(exist_ok=True); (refs / 'main').write_text(rev)"

# The landing page's text models, one <repo>@<revision> a line in text-models,
# each at /opt/models/<repo>: their weights, config and tokenizer.
COPY text-models /opt/live/
RUN python -c "from huggingface_hub import snapshot_download; [snapshot_download(repo_id=repo, revision=rev, local_dir=f'/opt/models/{repo}', allow_patterns=['*.json', '*.safetensors', '*.txt']) for repo, rev in (line.split('@') for line in open('/opt/live/text-models').read().split())]"

# dew-commit holds the full SHA of the Dew commit to install; deploy.mjs writes it.
COPY dew-commit /opt/live/
RUN grep -Eq '^[0-9a-f]{40}$' /opt/live/dew-commit \
	&& pip install --no-cache-dir "dewml @ git+https://github.com/AshishKumar4/dew@$(cat /opt/live/dew-commit)"

# Cells run as this user, which cannot signal or trace the server.
RUN useradd --create-home --uid 1001 --shell /usr/sbin/nologin kernel

# The page's setup cell, run once here so the tokenizer and VAE configs are in
# HF_HOME. The kernel then keeps compiled programs in /opt/xla, which a kernel
# restarted in the same container reads; programs compiled here would never be
# read (see warm.py).
ENV JAX_PLATFORMS=cpu
COPY warm.py sampler_setup.py /opt/live/
RUN python /opt/live/warm.py && mkdir /opt/xla && chown -R kernel:kernel /opt/xla /opt/hf
ENV JAX_COMPILATION_CACHE_DIR=/opt/xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
COPY sampler.py text.py progress.py preload.py /opt/live/

COPY server.py /opt/live/server.py
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
EXPOSE 8888
CMD ["python", "/opt/live/server.py"]
