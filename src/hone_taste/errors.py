"""Typed exceptions raised at the public API of hone-taste."""


class HoneTasteError(Exception):
    """Base class for every error hone-taste raises."""


class ConfigError(HoneTasteError):
    """A scorer, profile or combination was configured wrongly (bad name, weight, path ...)."""


class MissingExtra(HoneTasteError):
    """An optional dependency is not installed; the message names the extra to install."""


class LicenseNotAccepted(MissingExtra):
    """A model with unclear or restrictive license terms was requested without `accept_license=True`."""


class ModelUnavailable(HoneTasteError, RuntimeError):
    """A model could not be loaded or reached (weights not downloadable, server down, unknown model ...).

    Also a `RuntimeError`, as the TextClient port asks of transport errors raised by a `TextClient`.
    """
