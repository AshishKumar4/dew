"""What every published autoencoder's loader shares: its release's directory and config, resolved once, its
class checked before any weight is read (`check_class`, `published_source`)."""

from __future__ import annotations

from collections.abc import Mapping


def check_class(config: Mapping[str, object], names: tuple[str, ...], where, *,
                nameless: bool = False) -> None:
    """Refuse a published autoencoder's `config` whose `_class_name` is none of `names`, naming the class it
    holds (a Wan VAE's given to the KL loader); one that names none is taken as `names`' own where
    `nameless` (an older AutoencoderKL's config)."""
    named = config.get("_class_name")
    if named not in names and not (named is None and nameless):
        raise ValueError(f"{where} holds a {named}, not an {names[0]}")


def published_source(name_or_dir, revision, subfolder: str, names: tuple[str, ...], *, weights: bool,
                     nameless: bool = False, read: Mapping[str, object] | None = None):
    """A published autoencoder's directory and config, resolved once: its release's metadata at the commit
    `revision` resolves to (`dew.interop.sources.snapshot`), its config read there, or `read` where the
    caller read it there already (`load_autoencoder`'s dispatch), checked first (`check_class`), and its
    weights fetched only after, at that commit, so a foreign checkpoint is refused by name before any of
    its weights is read and the weights read are the checked config's."""
    import json
    import os

    from dew.interop.sources import snapshot

    release = snapshot(str(name_or_dir), revision, weights=False)
    directory = release / subfolder
    config: dict = dict(read) if read is not None else json.loads((directory / "config.json").read_text())
    check_class(config, names, directory, nameless=nameless)
    if weights and not os.path.isdir(name_or_dir):
        # A Hub release's snapshot is named by the commit it holds.
        fetched = snapshot(str(name_or_dir), release.name, weights=(subfolder,) if subfolder else True)
        directory = fetched / subfolder
    return directory, config
