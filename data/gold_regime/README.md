# WGC Gold Balance data

The runtime reads the normalized files in this directory and does not depend on a Desktop path or on the original workbook.

When a new World Gold Council report is supplied, refresh the dataset from the repository root:

```powershell
.\.venv\Scripts\python.exe gold_regime\wgc_import.py "C:\path\to\new\Gold Balance.xlsx" --output-dir data\gold_regime
```

The importer validates the required WGC rows and replaces:

- `wgc_gold_balance_annual.csv`
- `wgc_gold_balance_quarterly.csv`
- `wgc_gold_balance_metadata.json`

Annualized YTD values are calculated by the application for display only. They are never written as completed annual observations and can never become a Model 2 calibration year.

## GlobalPositiveCA sources

The primary source is the latest IMF World Economic Outlook Countries workbook discovered from `https://data.imf.org/Datasets/WEO`. Runtime normalization uses `BCA` and `NGDPD`, caches the resulting annual coverage table under `persistent/finance_cache/gold_regime`, and falls back to the stale cache if a refresh fails.

The World Bank `BN.CAB.XOKA.CD` series is downloaded separately as a historical cross-check. It does not control the Model 2 calibration gate.
