#!/bin/sh
# As root, once per armada environment (.armada.json): uv, which install.sh
# builds each Python's environment with, on armada's own runner layer (git,
# a compiler, tini).
set -eu
# The system trust store (curl, git, Python's ssl) from trixie-security: 20250419+deb13u1 leaves out
# the roots that eMudhra's eight compromised intermediate CAs chain to (Mozilla, 2026-10-20).
apt-get update -qq
apt-get install -y -qq --only-upgrade ca-certificates
# uv 0.13.4 is the first whose own CA store leaves out emSign's (GHSA-hq85-5cpj-mr62).
curl -LsSf https://astral.sh/uv/0.13.4/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
uv --version
