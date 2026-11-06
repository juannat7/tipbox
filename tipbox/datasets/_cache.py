"""Atomic caching for small, immutable source files pinned by SHA256."""

import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


def _validate(path, size, checksum):
    if path.stat().st_size != size:
        raise ValueError(f"Source file has an unexpected size: {path}")
    if sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(
            f"Source file checksum does not match the pinned version: {path}"
        )


def pinned_file(
    url,
    relative_path,
    *,
    size,
    checksum,
    cache_dir=None,
    local_path=None,
    download=True,
):
    """Return a verified local file; never replace a corrupt cached copy."""
    if local_path is not None:
        path = Path(local_path).expanduser()
    else:
        if cache_dir is None:
            default = Path.home() / ".cache"
            xdg = Path(os.environ.get("XDG_CACHE_HOME") or default)
            cache_dir = (xdg if xdg.is_absolute() else default) / "tipbox"
        path = Path(cache_dir).expanduser() / relative_path
    if not path.exists():
        if local_path is not None or not download:
            raise FileNotFoundError(
                f"Source file is not cached at {path}. Use download=True or a local source path."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary:
            partial = Path(temporary) / path.name
            request = Request(url, headers={"User-Agent": "TipBox dataset loader"})
            with urlopen(request, timeout=60) as response:
                payload = response.read(size + 1)
            partial.write_bytes(payload)
            _validate(partial, size, checksum)
            partial.replace(path)
    _validate(path, size, checksum)
    return path
