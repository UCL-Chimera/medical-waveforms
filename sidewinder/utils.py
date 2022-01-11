from pathlib import Path


def get_root_directory() -> Path:
    return Path(__file__).parent.parent
