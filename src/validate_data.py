from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "dataset_manifest.json"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(entry: dict[str, object], data_dir: Path, strict_hash: bool) -> dict[str, object]:
    filename = entry["filename"]
    path = data_dir / filename
    result: dict[str, object] = {
        "filename": filename,
        "required": entry.get("required", True),
        "exists": path.exists(),
        "status": "ok",
        "notes": [],
    }

    if not path.exists():
        result["status"] = "error" if entry.get("required", True) else "warning"
        result["notes"].append("file is missing")
        return result

    actual_hash = sha256sum(path)
    expected_hash = entry.get("sha256")
    if expected_hash and actual_hash != expected_hash:
        result["status"] = "error" if strict_hash else "warning"
        result["notes"].append("sha256 mismatch against tracked local reproduction snapshot")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
        expected_rows = entry.get("expected_rows")
        expected_columns = entry.get("expected_columns")
        if expected_rows is not None and int(expected_rows) != len(df):
            result["status"] = "error"
            result["notes"].append(f"row count mismatch: expected {expected_rows}, found {len(df)}")
        if expected_columns is not None and int(expected_columns) != len(df.columns):
            result["status"] = "error"
            result["notes"].append(f"column count mismatch: expected {expected_columns}, found {len(df.columns)}")

        required_columns = set(entry.get("required_columns", []))
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            result["status"] = "error"
            result["notes"].append(f"missing required columns: {', '.join(missing_columns)}")

    if not result["notes"]:
        result["notes"].append("validation passed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local raw data files against the tracked dataset manifest.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the files to validate.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON to validate against. Defaults to a manifest inferred from the selected data directory.",
    )
    parser.add_argument(
        "--strict-hash",
        action="store_true",
        help="Fail when file hashes differ from the tracked local reproduction snapshot.",
    )
    return parser.parse_args()


def infer_manifest_path(data_dir: Path) -> Path:
    if data_dir.resolve() == (BASE_DIR / "data" / "sample").resolve():
        return BASE_DIR / "data" / "sample_manifest.json"
    return DEFAULT_MANIFEST_PATH


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else infer_manifest_path(data_dir)
    manifest = load_manifest(manifest_path)
    rows = [validate_dataset(entry, data_dir=data_dir, strict_hash=args.strict_hash) for entry in manifest["datasets"]]
    status_df = pd.DataFrame(rows)
    print(status_df[["filename", "required", "exists", "status"]].to_string(index=False))
    for row in rows:
        print(f"\n[{row['filename']}]")
        for note in row["notes"]:
            print(f"- {note}")

    has_error = any(row["status"] == "error" for row in rows)
    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
