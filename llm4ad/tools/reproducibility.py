# SPDX-License-Identifier: BSD-2-Clause
"""Small, explicit helpers for seeds, redacted metadata and run manifests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
from datetime import datetime, timezone

import numpy as np

_SECRET_NAMES = {"key", "api_key", "apikey", "authorization", "password", "secret",
                 "secret_key", "access_token", "refresh_token", "token", "credentials"}


def redact(value):
    """Redact credential fields recursively; prompt/response text is retained verbatim.

    Callers must still inspect artifacts before sharing: free text can itself contain
    private information. This function is not a secret detector for arbitrary text.
    """
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower().lstrip("_") in _SECRET_NAMES
                else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def seed_all(seed: int):
    """Seed Python and NumPy. GPU kernels and remote model sampling are separate."""
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path):
    """Hash a dataset, weight file or artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint():
    """Fingerprint installed source/configs, including uncommitted local changes."""
    import llm4ad
    root = Path(llm4ad.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.yaml', '.json'}:
            digest.update(path.relative_to(root).as_posix().encode('utf-8') + b'\0')
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def write_run_manifest(path, *, config, model, seed, datasets=(), repository=None):
    """Write a versioned manifest atomically; return the exact recorded dictionary.

    ``model`` should include the model ID, immutable revision and sampling parameters.
    ``repository`` is optional: no unrelated current-directory Git state is assumed.
    """
    from llm4ad import __version__
    commit = None
    dirty = None
    if repository is not None:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True).strip())
    distributions = sorted((d.metadata.get("Name", ""), d.version)
                           for d in importlib.metadata.distributions())
    manifest = redact({
        "schema_version": 1, "llm4ad_version": __version__, "commit": commit,
        "installed_source_sha256": source_fingerprint(),
        "working_tree_dirty": dirty, "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": dict(distributions), "seed": seed, "config": config, "model": model,
        "datasets": [{"path": str(p), "sha256": sha256_file(p)} for p in datasets],
    })
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest
