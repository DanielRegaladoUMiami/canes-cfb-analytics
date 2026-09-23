"""Repo paths, importable from any notebook regardless of its working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
NOTEBOOKS = ROOT / "notebooks"
PREDICTIONS = DATA / "predictions"
