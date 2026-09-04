# The FID feature extractor, whose modules are derived from
# https://github.com/matthias-wright/jax-fid

import hashlib
import pickle

import flax
import jax
import numpy as np


def fetch(repo: str, filename: str, revision: str, digest: str) -> str:
    """The path to `filename` of `repo` at `revision`, once its bytes hash to
    `digest`.

    The weights used to come from a consumer file-sharing link with no
    checksum, which a released package then unpickled: whoever held that link
    chose what ran. A Hub repo pins a revision, and the digest is checked here
    as well, so the bytes are what this code was written against whatever the
    transport did.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, revision=revision)
    with open(path, 'rb') as handle:
        found = hashlib.file_digest(handle, 'sha256').hexdigest()
    if found != digest:
        raise ValueError(
            f"{repo}/{filename} at {revision} hashes to {found}, not the {digest} "
            "this code was written against")
    return path


# What a pickle of numpy arrays needs to rebuild them, and nothing else. The
# FID weights were written under numpy 1, where these lived under
# `numpy.core`; numpy 2 keeps that path alive only as a shim that warns on
# every attribute read, so a legacy name is resolved at its current home
# instead. Anything outside this set is refused rather than imported, so a
# downloaded pickle cannot run code.
_ARRAY_GLOBALS = {
    ('numpy', 'dtype'),
    ('numpy', 'ndarray'),
    ('numpy._core.multiarray', '_reconstruct'),
    ('numpy._core.numeric', '_frombuffer'),
}


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        current = ('numpy._core.' + module[len('numpy.core.'):]
                   if module.startswith('numpy.core.') else module)
        if (current, name) not in _ARRAY_GLOBALS:
            raise pickle.UnpicklingError(
                f"the weights file asks for {module}.{name}, which is not one of the "
                "numpy array constructors this loader allows")
        return super().find_class(current, name)


def load_arrays(path):
    """The nested dict of arrays in a numpy-only pickle at `path`."""
    with open(path, 'rb') as handle:
        return _ArrayUnpickler(handle).load()


def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]
