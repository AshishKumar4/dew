"""Convert the upstream jax-fid FID weights into a Flax variables tree.

The published extractor is a pickle of numpy arrays, so this module is the one
place in Dew that unpickles a file, behind a pinned digest and an unpickler
that can build nothing but numpy arrays. Everything downstream reads the
safetensors written here and hands it to `InceptionV3.apply`, so the runtime
never opens a pickle it did not just convert.
"""

import hashlib
import pickle
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from dew.interop.safetensors_io import SEPARATOR, _flatten, _unflatten, read_file, write_file
from dew.nn.text_encoders import ParamTree
from dew.objectives.base import Variables

# The FID feature extractor's weights, the jax-fid pickle mirrored on the Hub
# with a pinned revision and digest. The bytes are byte-identical to the
# jax-fid file.
FID_WEIGHTS_REPO = 'hayden-donnelly/inception-v3-fid'
FID_WEIGHTS_FILE = 'inception_v3_fid.pickle'
FID_WEIGHTS_REVISION = 'ccb3ff416ff491ae7fd964c5e7c01d12ab7c48bf'
FID_WEIGHTS_DIGEST = '4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77'
# The converted tree, written beside the pickle it was converted from.
CONVERTED_FILE = 'inception_v3_fid.safetensors'


def _check_digest(path: str, repo: str, filename: str, revision: str, digest: str) -> str:
    """Return `path` if its SHA-256 matches `digest`, and raise otherwise.

    This is separate from `fetch` so a test can check the pin against local
    bytes without downloading the weights.
    """
    with open(path, 'rb') as handle:
        found = hashlib.file_digest(handle, 'sha256').hexdigest()
    if found != digest:
        raise ValueError(
            f"{repo}/{filename} at {revision} hashes to {found}, not the {digest} "
            "this code was written against")
    return path


def fetch(repo: str, filename: str, revision: str, digest: str) -> str:
    """Download `filename` from `repo` at `revision` and return its local path.

    The digest is checked before the caller unpickles the file, so altered bytes
    are rejected even when the revision resolves correctly.
    """
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


def load_arrays(path) -> ParamTree:
    """Read the nested dict of arrays in a numpy-only pickle at `path`."""
    with open(path, 'rb') as handle:
        return _ArrayUnpickler(handle).load()


# InceptionV3 names its children after their class and the order it built them
# in; upstream named the same modules after the torchvision layers they were
# ported from. These two tables are that vocabulary, and nothing else knows it.
_MODULES = {
    'BasicConv2d_0': 'Conv2d_1a_3x3',
    'BasicConv2d_1': 'Conv2d_2a_3x3',
    'BasicConv2d_2': 'Conv2d_2b_3x3',
    'BasicConv2d_3': 'Conv2d_3b_1x1',
    'BasicConv2d_4': 'Conv2d_4a_3x3',
    'InceptionA_0': 'Mixed_5b',
    'InceptionA_1': 'Mixed_5c',
    'InceptionA_2': 'Mixed_5d',
    'InceptionB_0': 'Mixed_6a',
    'InceptionC_0': 'Mixed_6b',
    'InceptionC_1': 'Mixed_6c',
    'InceptionC_2': 'Mixed_6d',
    'InceptionC_3': 'Mixed_6e',
    'InceptionD_0': 'Mixed_7a',
    'InceptionE_0': 'Mixed_7b',
    'InceptionE_1': 'Mixed_7c',
}

# The branches each mixed block builds, in the order it builds them, so
# BasicConv2d_2 inside an InceptionA is upstream's branch5x5_2.
_BRANCHES = {
    'InceptionA': ('branch1x1', 'branch5x5_1', 'branch5x5_2', 'branch3x3dbl_1',
                   'branch3x3dbl_2', 'branch3x3dbl_3', 'branch_pool'),
    'InceptionB': ('branch3x3', 'branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3'),
    'InceptionC': ('branch1x1', 'branch7x7_1', 'branch7x7_2', 'branch7x7_3',
                   'branch7x7dbl_1', 'branch7x7dbl_2', 'branch7x7dbl_3',
                   'branch7x7dbl_4', 'branch7x7dbl_5', 'branch_pool'),
    'InceptionD': ('branch3x3_1', 'branch3x3_2', 'branch7x7x3_1', 'branch7x7x3_2',
                   'branch7x7x3_3', 'branch7x7x3_4'),
    'InceptionE': ('branch1x1', 'branch3x3_1', 'branch3x3_2a', 'branch3x3_2b',
                   'branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3a',
                   'branch3x3dbl_3b', 'branch_pool'),
}

# A convolution and its norm, the two modules a BasicConv2d holds. The running
# statistics are the batch_stats collection, the rest are params; upstream kept
# all four norm arrays together under `bn`.
_LEAVES = {
    ('Conv_0', 'kernel'): ('conv', 'kernel'),
    ('Conv_0', 'bias'): ('conv', 'bias'),
    ('BatchNorm_0', 'scale'): ('bn', 'scale'),
    ('BatchNorm_0', 'bias'): ('bn', 'bias'),
    ('BatchNorm_0', 'mean'): ('bn', 'mean'),
    ('BatchNorm_0', 'var'): ('bn', 'var'),
}

# The upstream artifact is a classifier. The extractor stops at pool3 and
# builds no head, so these have no module to land on.
_HEAD = ('fc', 'AuxLogits')


def _upstream(path: tuple[str, ...]) -> tuple[str, ...]:
    """Return the upstream name of a module leaf.

    `path` is a variables path without its collection, such as
    `BasicConv2d_0/Conv_0/kernel`.
    """
    module, *rest = path
    if module not in _MODULES:
        raise ValueError(
            f"the extractor has a {module} the jax-fid weights never named, at "
            f"{SEPARATOR.join(path)}")
    names = [_MODULES[module]]
    if len(rest) > 2:
        branch, *rest = rest
        block = module.rsplit('_', 1)[0]
        order = _BRANCHES.get(block, ())
        index = int(branch.rsplit('_', 1)[1])
        if index >= len(order):
            raise ValueError(
                f"{SEPARATOR.join(path)}: {block} builds more convolutions than the "
                f"{len(order)} branches jax-fid named for it")
        names.append(order[index])
    leaf = _LEAVES.get((rest[0], rest[1])) if len(rest) == 2 else None
    if leaf is None:
        raise ValueError(
            f"{SEPARATOR.join(path)}: jax-fid stored no {SEPARATOR.join(rest)}")
    return (*names, *leaf)


def _module_leaves() -> list[tuple[str, ...]]:
    """Return every leaf the extractor initialises, collection first."""
    from dew.eval.inception import InceptionV3

    variables = jax.eval_shape(
        lambda: InceptionV3().init(jax.random.PRNGKey(0), jnp.zeros((1, 299, 299, 3))))
    paths, _ = jax.tree_util.tree_flatten_with_path(variables)
    return [tuple(entry.key for entry in path) for path, _ in paths]


def upstream_names() -> dict[str, tuple[str, ...]]:
    """Return every leaf the extractor initialises, against its jax-fid name.

    The keys are the '/'-joined variables paths, collection first, and they are
    what a converted file stores; this mapping is the whole of what the two
    layouts have to agree on.
    """
    return {SEPARATOR.join(path): _upstream(tuple(path[1:])) for path in _module_leaves()}


def convert(pickle_path) -> Variables:
    """Read the jax-fid pickle into the variables tree `InceptionV3` initialises.

    The extractor's own tree says which arrays are wanted and what they are
    called; every leaf of it is looked up under its upstream name, and a pickle
    that does not answer for one of them, or that carries an array with no
    module to land on, is refused by name.
    """
    source = load_arrays(pickle_path)
    converted: dict[str, np.ndarray] = {}
    taken: set[tuple[str, ...]] = set()
    for name, upstream in upstream_names().items():
        node = source
        for step in upstream:
            if not isinstance(node, Mapping) or step not in node:
                raise ValueError(
                    f"{pickle_path} has no {SEPARATOR.join(upstream)}, which is the "
                    f"extractor's {name}")
            node = node[step]
        converted[name] = np.asarray(node)
        taken.add(upstream)
    for leaf in flatten_dict(source):
        if leaf not in taken and leaf[0] not in _HEAD:
            raise ValueError(
                f"{pickle_path} carries {SEPARATOR.join(leaf)}, which the extractor "
                "has no module for")
    return _unflatten(converted)


def save(variables: Variables, path, divisor: int = 1) -> None:
    """Write a variables tree, one tensor per leaf under its '/'-joined path.

    The header records the width the tree was written at, because a Flax
    parameter has to be the shape its module declares: whoever applies the file
    has to build the extractor the same way, and the file is what says so.
    """
    write_file(_flatten(variables), path,
               {"format": "flax", "module": "dew.eval.inception.InceptionV3",
                "channel_divisor": str(divisor)})


def load(path) -> Variables:
    """Read a converted file back into the tree `InceptionV3.apply` takes."""
    tensors, _ = read_file(path)
    return _unflatten(tensors)


def channel_divisor(path) -> int:
    """Return the `InceptionV3(channel_divisor=)` a file was written at.

    It is read from the file's header. The published weights are the whole
    network and say 1.
    """
    _, metadata = read_file(path)
    if "channel_divisor" not in metadata:
        raise ValueError(
            f"{path} does not say what width it holds: an InceptionV3 file's header "
            "carries channel_divisor, and this one has "
            f"{', '.join(sorted(metadata)) or 'no metadata at all'}")
    return int(metadata["channel_divisor"])


def cached_weights() -> Path:
    """Return the converted weights, beside the pickle in the Hub cache.

    The download and the conversion happen on the first call; every call after
    it finds the file already there.
    """
    pickled = Path(fetch(FID_WEIGHTS_REPO, FID_WEIGHTS_FILE,
                         FID_WEIGHTS_REVISION, FID_WEIGHTS_DIGEST))
    converted = pickled.parent / CONVERTED_FILE
    if not converted.exists():
        save(convert(pickled), converted)
    return converted
