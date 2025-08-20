
import os
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def data_path(*parts) -> Path:
    p = project_root() / "data" / Path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
