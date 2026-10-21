"""The CA bundle Dew's environment verifies TLS with (certifi, which requests, httpx and huggingface_hub
read): it trusts none of the roots that chain to the eMudhra intermediates whose private keys were
compromised in October 2026 (Mozilla's notice to trust stores, oss-security 2026-10-20). certifi
2026.10.19 dropped the one it held; the locks pin it or later (constraints.txt)."""

import hashlib
import re
from pathlib import Path

import certifi
from cryptography import x509
from cryptography.hazmat.primitives import serialization

COMPROMISED_CHAINS = {
    "c619e4780ed8e0abeda24ad732567de4383a7ea941338f6d1a7d1b49ac792fe9": "emSign Root TLS CA - G3",
    "8d417db2dd8bf5e3084d1e3f196d583849d81bdd4c00c70b9d39369e96b8c782": "emSign ECC Root CA - G3",
    "bd153ed7b0434f6886b17bce8bbe84ed340c7132d702a8f4fa318f756ecbd6f3": "AAA Certificate Services",
}
"""Each root Mozilla named, by the SHA-256 of its SubjectPublicKeyInfo, as the notice gives them."""


def spki_sha256(certificate: x509.Certificate) -> str:
    public = certificate.public_key().public_bytes(serialization.Encoding.DER,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(public).hexdigest()


def test_the_environments_ca_bundle_trusts_no_root_of_the_compromised_emudhra_intermediates():
    bundle = Path(certifi.where()).read_bytes()
    certificates = [x509.load_pem_x509_certificate(pem) for pem in
                    re.findall(rb"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", bundle, re.S)]
    assert len(certificates) > 100, f"{certifi.where()} holds {len(certificates)} certificates"
    trusted = sorted(COMPROMISED_CHAINS[key] for key in map(spki_sha256, certificates)
                     if key in COMPROMISED_CHAINS)
    assert not trusted, f"certifi {certifi.__version__} trusts {trusted}"
