"""What every published autoencoder's loader shares: its release's directory and config, its class checked
before any weight is read (`check_class`, `published_source`)."""

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
                     nameless: bool = False):
    """A published autoencoder's directory and config (`dew.interop.components.component_source`), its
    config checked first (`check_class`) and its weights fetched only after, so a foreign checkpoint is
    refused by name before any of its weights is read."""
    from dew.interop.components import component_source

    directory, config = component_source(name_or_dir, revision, subfolder, weights=False)
    check_class(config, names, directory, nameless=nameless)
    if weights:
        directory, config = component_source(name_or_dir, revision, subfolder, weights=True)
    return directory, config

