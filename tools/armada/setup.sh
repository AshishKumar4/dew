#!/bin/sh
# As root, once per armada environment (.armada.json): uv, which install.sh
# builds each Python's environment with, on armada's own runner layer (git,
# a compiler, tini).
set -eu
# BEGIN uv bootstrap (site/scripts/gen_install.py)
uv_least=0.13.4
# uv at uv_least or newer (pyproject.toml's [tool.uv] required-version). bootstrap_uv [NAME=VALUE...]
# installs that release from astral's installer, NAME=VALUE in the installer's environment, where uv is
# missing or older; uv_version is the release of the uv on PATH (none where there is none), and
# uv_older VERSION says whether VERSION is older: a pre-release of uv_least is, a build of it is not.
uv_version() { (uv --version 2>/dev/null || true) | awk '{print $2}'; }
uv_older() {
	awk -v have="${1%%+*}" -v least="$uv_least" 'BEGIN {
		split(have, h, "."); split(least, l, ".")
		for (i = 1; i <= 3; i++) if (h[i] + 0 != l[i] + 0) exit !(h[i] + 0 < l[i] + 0)
		exit have !~ /[^0-9.]/
	}'
}
bootstrap_uv() {
	have=$(uv_version)
	if [ -z "$have" ] || uv_older "$have"; then
		curl -LsSf "https://astral.sh/uv/$uv_least/install.sh" | env "$@" sh
	fi
}
# END uv bootstrap
# The system trust store (curl, git, Python's ssl) from trixie-security: 20250419+deb13u1 leaves out
# the roots that eMudhra's eight compromised intermediate CAs chain to (Mozilla, 2026-10-20).
apt-get update -qq
apt-get install -y -qq --only-upgrade ca-certificates
bootstrap_uv UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1
uv --version
