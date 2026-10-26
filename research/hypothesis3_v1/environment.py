"""Dependency inspection only; importing this module never installs packages."""
from importlib import metadata


def missing_requirements(path, installed_version=metadata.version):
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    missing, reused = [], []
    for text in path.read_text(encoding="utf-8").splitlines():
        text = text.strip()
        if not text or text.startswith("#"):
            continue
        req = Requirement(text)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            version = installed_version(req.name)
        except metadata.PackageNotFoundError:
            missing.append(text)
            continue
        if version not in req.specifier:
            missing.append(text)
        else:
            reused.append(dict(name=req.name, version=version))
    return missing, reused
