from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BASE_DIR / "data" / "dataset_manifest.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ML7501 raw datasets from the tracked manifest URLs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the dataset manifest JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the downloaded raw files will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each file download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned downloads without fetching any files.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("datasets", []))


def download_file(url: str, destination: Path, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "ML7501-Project-Downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "unknown")
        temp_path = destination.with_suffix(destination.suffix + ".part")
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination)
        return content_type


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_manifest(manifest_path)
    downloadable = [row for row in datasets if str(row.get("source_url", "")).startswith("http")]

    if not downloadable:
        print("No downloadable HTTP sources were found in the manifest.")
        return 0

    for dataset in downloadable:
        filename = str(dataset["filename"])
        url = str(dataset["source_url"])
        destination = output_dir / filename

        if destination.exists() and not args.overwrite:
            print(f"Skipping existing file: {destination}")
            continue

        print(f"Downloading {filename}")
        print(f"  Source: {url}")
        print(f"  Target: {destination}")

        if args.dry_run:
            continue

        try:
            content_type = download_file(url=url, destination=destination, timeout=args.timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"  Failed: {exc}", file=sys.stderr)
            continue

        print(f"  Done (Content-Type: {content_type})")
        if "html" in content_type.lower() and destination.suffix.lower() in {".csv", ".json"}:
            print(
                "  Warning: the server returned HTML instead of a raw data file. "
                "You may need to download this dataset manually from the portal page.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
