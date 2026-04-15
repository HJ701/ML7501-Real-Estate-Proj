# Data Schemas

This folder contains tracked schema snapshots derived from the local raw-data reproduction used in the project.

Each schema file records:

- row count
- column count
- every column name
- the observed dtype in the tracked local snapshot
- missing-value percentage

These files reduce ambiguity about the expected input structure even though the full raw data are not versioned in the repository.
