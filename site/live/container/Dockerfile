# The live kernel: Dew at one pinned commit, JAX on the CPU, and the relay in
# server.py. Everything is installed here, at build time; the running container
# has no network access.
FROM python:3.13-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/live/
RUN pip install --no-cache-dir -r /opt/live/requirements.txt

# dew-commit holds the full SHA of the Dew commit to install; deploy.mjs writes it.
COPY dew-commit /opt/live/
RUN grep -Eq '^[0-9a-f]{40}$' /opt/live/dew-commit \
	&& pip install --no-cache-dir "dew-ml @ git+https://github.com/AshishKumar4/dew@$(cat /opt/live/dew-commit)"

# Cells run as this user, which cannot signal or trace the server.
RUN useradd --create-home --uid 1001 --shell /usr/sbin/nologin kernel

COPY server.py /opt/live/server.py
ENV PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
EXPOSE 8888
CMD ["python", "/opt/live/server.py"]
