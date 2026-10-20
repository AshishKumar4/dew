#!/bin/sh
# As root, once per armada environment (.armada.json): uv, which install.sh
# builds each Python's environment with, on armada's own runner layer (git,
# a compiler, tini).
set -eu
# The system trust store (curl, git, Python's ssl) leaves out the roots that eMudhra's eight
# compromised intermediate CAs chain to, as Mozilla advised on 2026-10-20 where an intermediate
# cannot be distrusted alone: emSign Root TLS CA - G3, emSign ECC Root CA - G3 and AAA Certificate
# Services. Every host the environment fetches from verifies without them.
sed -i -E 's#^mozilla/(emSign_Root_TLS_CA_-_G3|emSign_ECC_Root_CA_-_G3|AAA_Certificate_Services)\.crt$#!&#' /etc/ca-certificates.conf
update-ca-certificates --fresh >/dev/null
# uv 0.13.4 is the first whose own CA store leaves out emSign's (GHSA-hq85-5cpj-mr62).
curl -LsSf https://astral.sh/uv/0.13.4/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
uv --version
