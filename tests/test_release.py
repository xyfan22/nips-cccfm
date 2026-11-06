import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPRODUCE = load("reproduce_validation", "scripts/reproduce_validation.py")
DOWNLOAD = load("download_data", "scripts/download_data.py")
VERIFY = load("verify_release", "scripts/verify_release.py")


def test_all_released_summaries(tmp_path):
    for function in (REPRODUCE.factorial, REPRODUCE.pbmc, REPRODUCE.stratification):
        assert function(ROOT, tmp_path)["max_summary_error"] < 1e-11


def test_integrity_manifest():
    assert VERIFY.verify() > 40


def test_factorial_nonuniform_counts_rejected(monkeypatch, tmp_path):
    read = REPRODUCE.read_csv

    def altered(path):
        rows = read(path)
        if path.name == "rank256_hidden1024_seed7.csv":
            rows[0]["n_pred_cells"] = str(int(rows[0]["n_target_cells"]) + 1)
        return rows

    monkeypatch.setattr(REPRODUCE, "read_csv", altered)
    with pytest.raises(ValueError, match="non-uniform"):
        REPRODUCE.factorial(ROOT, tmp_path)


def test_pbmc_inconsistent_grid_rejected(monkeypatch, tmp_path):
    read = REPRODUCE.read_csv

    def altered(path):
        rows = read(path)
        if path.name == "per_condition_combined.csv":
            rows[0]["cell_type"] = "Unexpected cell type"
        return rows

    monkeypatch.setattr(REPRODUCE, "read_csv", altered)
    with pytest.raises(ValueError, match="six-type, 88-cytokine grid"):
        REPRODUCE.pbmc(ROOT, tmp_path)


def test_numeric_summary_mismatch_rejected():
    with pytest.raises(ValueError, match="Numeric mismatch"):
        REPRODUCE.compare([{"id": "a", "value": 2.0}], [{"id": "a", "value": "1.0"}], ("id",))


def test_duplicate_summary_key_rejected():
    row = {"id": "a", "value": 1}
    with pytest.raises(ValueError, match="Duplicate"):
        REPRODUCE.compare([row, row], [row], ("id",))


def test_nonfinite_summary_rejected():
    with pytest.raises(ValueError, match="Numeric mismatch"):
        REPRODUCE.compare([{"id": "a", "value": float("nan")}], [{"id": "a", "value": 1}], ("id",))


def test_download_checks_size_and_content(tmp_path):
    path = tmp_path / "input.h5ad"
    path.write_bytes(b"abc")
    entry = {"size_bytes": 3, "md5": "900150983cd24fb0d6963f7d28e17f72"}
    DOWNLOAD.verify(path, entry)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="MD5"):
        DOWNLOAD.verify(path, entry)
    path.write_bytes(b"a")
    with pytest.raises(ValueError, match="Size"):
        DOWNLOAD.verify(path, entry)
