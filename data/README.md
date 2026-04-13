# Data Directory

Raw datasets live in `data/raw/`.

Files currently used by the pipeline:

- `Real-estate_Transactions_2026-03-27.csv`
- `rent_contracts.csv`
- `FCSA,DF_HOT_TYPE,4.3.0+...A.....csv`

The JSON file `rent_contracts_page1.json` is kept as an auxiliary raw source.

These raw files are large and should generally stay out of version control. The source code expects them to exist locally under `data/raw/`.
