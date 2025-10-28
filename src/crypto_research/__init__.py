from importlib import metadata

try:
    __version__ = metadata.version("crypto-research")
except metadata.PackageNotFoundError:  # pragma: no cover - local dev
    __version__ = "0.0.0"

__all__ = ["__version__"]

