# Archived interference-duration captures

These 12 raw CSV files were copied byte for byte from
`Myshare/ks_interference_model_scen2/` and
`Myshare/ks_interference_model_scen4/` on 2026-10-26. The source directory
structure is preserved. They supplement the six models' offline profiles in
[`../profiling/`](../profiling/).

The captures contain the original fields:

```text
allksid_sm, now_ksid_sm, duration
```

`allksid_sm` records the encoded colocated kernel-segment/SM state;
`now_ksid_sm` identifies the current segment/SM state, and `duration` is the
recorded timing value. Keep the identifier fields as strings, including empty
values. Do not infer a unique model mapping or split all concatenated states
using one fixed-width rule without the corresponding capture configuration.
The legacy hook uses nanosecond timing fields, but the original CSVs contain
no complete timer/configuration metadata. No timing values were rescaled here.

There are 228,259 records after the initial headers, including 76 repeated
header records from concatenated/restarted captures. These records have been
preserved. The remaining duration fields parse as positive integers. Some
files are byte-identical copies in different source locations; they remain
separate provenance entries rather than independent experimental evidence.

Example parsing:

```python
import csv

path = "data/interference/ks_interference_model_scen4/s1/ksid_sm_dur_summary_6.csv"
with open(path, newline="") as file:
    for row in csv.DictReader(file, skipinitialspace=True):
        if row["duration"] == "duration":
            continue  # Repeated source header
        duration = int(row["duration"])
        colocated_state = row["allksid_sm"]
```

[`manifest.csv`](manifest.csv) records source paths, byte sizes, SHA-256
hashes, and per-file record/header counts. [`summary.json`](summary.json)
records the aggregate coverage (5,719,470 source bytes). Recheck the copies:

```bash
python3 scripts/verify_profiling_data.py --root data/interference --hashes-only
```

These are archived duration captures, not a newly fitted interference model
or an assertion that the released empirical coefficients can be reproduced
from these files alone.
