# Data Directory

Raw datasets live in `data/raw/`.

The tracked source code expects these exact filenames:

- `Real-estate_Transactions_2026-03-27.csv`
- `rent_contracts.csv`
- `FCSA,DF_HOT_TYPE,4.3.0+...A.....csv`

The JSON file `rent_contracts_page1.json` is kept as an auxiliary raw source.

These raw files are large and stay out of version control.

Reproducibility files:

- `data/dataset_manifest.json` records source URLs, expected filenames, expected shapes, required columns, and the exact SHA-256 hashes used for the tracked local reproduction snapshot.
- `data/download_data.py` attempts to download the raw files directly from the tracked manifest URLs into `data/raw/`.
- `data/schemas/` contains tracked full schema snapshots for the three main CSV inputs.
- `data/sample/` contains a lightweight tracked public sample for turn-key smoke runs.
- `data/sample_manifest.json` records the tracked sample hashes and shapes.
- `python3 -m src.validate_data` checks that the required files exist and have the expected schema.
- `python3 -m src.validate_data --strict-hash` additionally enforces exact hash matching for bitwise reproduction of the local snapshot.

To attempt direct raw-data download from the tracked source URLs:

```bash
python3 data/download_data.py --output-dir data/raw
```

If a portal returns HTML instead of a raw `.csv`/`.json` file, the script warns and you should download that dataset manually from the listed source page.

The source code expects the raw files to exist locally under `data/raw/`.
