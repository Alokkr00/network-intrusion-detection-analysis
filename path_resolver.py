"""
Dynamic Platform-Agnostic Directory Resolution Module (Python)
Aegis SOC Architecture - Network Intrusion Detection System

Provides centralized, normalized, cross-platform directory and path resolution using pathlib.
Supports environment variable overrides and directory traversal security validation.
"""

import os
from pathlib import Path

# Application Root Directory (default: directory containing this file)
_ENV_ROOT = os.environ.get("NIDS_ROOT_DIR")
ROOT_DIR = Path(_ENV_ROOT).resolve() if _ENV_ROOT else Path(__file__).resolve().parent

# Configured Subdirectories with Intelligent Fallbacks
_ENV_UPLOAD = os.environ.get("NIDS_UPLOAD_DIR")
UPLOAD_DIR = Path(_ENV_UPLOAD).resolve() if _ENV_UPLOAD else ROOT_DIR / "Uploaded_files"

_ENV_MODELS = os.environ.get("NIDS_MODELS_DIR")
MODELS_DIR = Path(_ENV_MODELS).resolve() if _ENV_MODELS else ROOT_DIR

_ENV_DATA = os.environ.get("NIDS_DATA_DIR")
DATA_DIR = Path(_ENV_DATA).resolve() if _ENV_DATA else ROOT_DIR

# Ensure UPLOAD_DIR exists on module import
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_root_dir() -> Path:
    """Return the normalized root Path."""
    return ROOT_DIR


def get_upload_dir() -> Path:
    """Return the normalized upload directory Path, creating it if needed."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def get_models_dir() -> Path:
    """Return the normalized models directory Path."""
    return MODELS_DIR


def get_data_dir() -> Path:
    """Return the normalized data directory Path."""
    return DATA_DIR


def resolve_path(*parts) -> Path:
    """
    Resolve path parts relative to ROOT_DIR.
    Handles cross-platform slashes and normalizes result.
    """
    return (ROOT_DIR.joinpath(*parts)).resolve()


def resolve_model(filename: str) -> Path:
    """Resolve a machine learning model artifact path."""
    return (MODELS_DIR / Path(filename).name).resolve()


def resolve_upload(filename: str) -> Path:
    """Resolve an uploaded file path safely, stripping any directory traversal."""
    safe_name = Path(filename).name
    return (UPLOAD_DIR / safe_name).resolve()


def is_safe_path(target_path: Path, base_dir: Path = None) -> bool:
    """
    Verify whether target_path is safely contained within base_dir
    to prevent directory traversal exploits.
    """
    if base_dir is None:
        base_dir = UPLOAD_DIR
    resolved_target = Path(target_path).resolve()
    resolved_base = Path(base_dir).resolve()
    try:
        resolved_target.relative_to(resolved_base)
        return True
    except ValueError:
        return False


def to_posix(path_input) -> str:
    """Convert any Path or path string into POSIX forward-slash format."""
    return Path(path_input).as_posix()
