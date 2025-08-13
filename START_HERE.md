# Generator Analysis Platform

## Purpose

This project analyzes electricity generator forecast performance across major U.S. power markets (MISO, SPP, ERCOT). It identifies generators with poor forecasting accuracy based on the difference between the forecast and actual generation, and also detects chronic over/under-forecasting patterns that could impact market operations.

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
