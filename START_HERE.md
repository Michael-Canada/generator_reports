# Generator Analysis Platform

## Purpose

This project analyzes the generator forecast performance across the markets (MISO, SPP, ERCOT). It identifies generators with poor forecasting accuracy based on a number of criteria, among which are:
- RMSE of the difference between the forecast and actual generation (in MW and % of Pmax)
- temporal radical differences between forecast and actual generation (Chronic over/under-forecasting patterns)
- Differences in the value of Pmax between reflow and resourceDB
- Wrong bids
- Consistent errors across different fuel types


## Key Capabilities

- **Performance Analysis**: Compares actual vs. forecasted generation
- **Anomaly Detection**: Identifies generators with forecast performance that is worse than other generations of the same kind
- **Market Coverage**: Supports MISO, SPP, and ERCOT electricity markets
- **Automated Reporting**: Generates PDF reports and alerts

## Business Value

Helps developers improve forecast accuracy by identifying generators whose models require attention

## Getting Started

Run the main analyzer: `python Auto_weekly_generator_analyzer2.py`. Make sure to choose the desired parameters at the begining of the file:
- **MIN_MW_TO_BE_ANALYZED = 100**: (default) only generators with Pmax, or max_observed_pg > 100 will be reported.
- **RUN_BID_VALIDATION**: True for comprehensive (and slow) run. Flase otherwise.

For detailed documentation, see the README.md, API_REFERENCE.md and other documentation files.
