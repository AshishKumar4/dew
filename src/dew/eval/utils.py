# The FID feature extractor, whose modules are derived from
# https://github.com/matthias-wright/jax-fid

import hashlib
import pickle


def _check_digest(path: str, repo: str, filename: str, revision: str, digest: str) -> str:
    """Separate verification lets tests check the pin against local bytes
    without downloading the weights."""
    with open(path, 'rb') as handle:
        found = hashlib.file_digest(handle, 'sha256').hexdigest()
    if found != digest:
        raise ValueError(
            f"{repo}/{filename} at {revision} hashes to {found}, not the {digest} "
            "this code was written against")
    return path


def fetch(repo: str, filename: str, revision: str, digest: str) -> str:
    """The digest check rejects altered weight bytes before unpickling, even
    when the download resolves to the pinned revision."""
    from huggingface_hub import hf_hub_download

    return _check_digest(hf_hub_download(repo, filename, revision=revision),
                         repo, filename, revision, digest)


# Only what a pickle of numpy arrays needs to rebuild them. The
# FID weights name these under `numpy.core`, the numpy 1 home; numpy 2 keeps
# that path as a shim that warns on every attribute read, so a legacy name
# resolves at its current home instead. Anything outside this set is refused,
# so a downloaded pickle cannot run code.
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
