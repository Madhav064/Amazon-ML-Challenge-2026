import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT.parent / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT.parent / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT.parent / "output"))


def work(split: str, *parts: str) -> Path:
    p = WORK_DIR / split
    p.mkdir(parents=True, exist_ok=True)
    return p.joinpath(*parts)
