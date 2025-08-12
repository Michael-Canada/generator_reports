# PDF Report Generation Documentation

## Overview

The enhanced generator analysis system now includes comprehensive PDF report generation that provides detailed analysis of all performance measures and lists poorly performing generators.

## Features

The PDF report includes the following sections:

### 1. Executive Summary
- Overall statistics and key metrics
- Performance distribution charts
- Alert severity breakdown
- High-level summary table

### 2. Performance Classification System
- **Description**: Explains the 5-tier classification system (Excellent, Good, Fair, Poor, Critical)
- **Criteria**: 
  - Excellent: RMSE ≤ 2% of capacity, R² ≥ 0.95
  - Good: RMSE ≤ 5% of capacity, R² ≥ 0.85
  - Fair: RMSE ≤ 10% of capacity, R² ≥ 0.70
  - Poor: RMSE ≤ 20% of capacity, R² ≥ 0.50
  - Critical: RMSE > 20% of capacity or R² < 0.50
- **Bottom 10 Percentile**: Lists generators with worst performance scores

### 3. Chronic Forecast Error Detection
- **Description**: Explains chronic over/under-forecasting detection using sliding window approach
- **Criteria**:
  - Minimum 5 problematic days in any 7-day sliding window
  - Over-forecasting: Forecast > 2x actual generation
  - Under-forecasting: Forecast < 0.5x actual generation
  - High severity if 10+ problematic days occur in any 14-day window
  - Minimum 12 hours of data per day to qualify
  - Only considers periods with generation ≥ 5 MW to avoid noise
- **Pattern Detection**: Catches weekday-only or other non-consecutive patterns
- **Affected Generators**: Lists generators with chronic errors and pattern details (e.g., "5/7 days")

### 4. Advanced Metrics Analysis
- **Consistency Score**: Measures forecast error consistency (0-1, higher = better)
- **Volatility Score**: Rolling standard deviation of errors (lower = better)
- **Trend Analysis**: Statistical trend detection (improving/stable/deteriorating)
- **Bottom Performers**: Lists worst performers for each metric

### 5. Statistical Anomaly Detection
- **Z-Score Analysis**: RMSE and MAE compared to population
- **Thresholds**: Z-score > 2.0 flagged, > 3.0 considered critical
- **Population Outliers**: Generators performing significantly worse than peers

### 6. Bid Validation Analysis (if available)
- **First Block vs Pmin**: Validation of minimum generation blocks
- **Last Block Sufficiency**: Capacity for typical generation levels
- **Pmax Validation**: Maximum capacity validation
- **Nuclear Exclusion**: Special handling for nuclear generators
- **High-Severity Issues**: Critical bid configuration problems

### 7. Operational Characteristics
- **Capacity Utilization**: Percentage of time generators run
- **Must-Run Analysis**: Baseload vs. cycling generators
- **Performance by Fuel Type**: RMSE comparison across fuel types
- **Low Utilization Issues**: Intermittent generators with poor forecasts

### 8. Recommendations and Action Items
- **Priority Actions**: Critical and high-severity issues requiring immediate attention
- **Chronic Error Solutions**: Specific recommendations for persistent problems
- **Statistical Outliers**: Investigation guidance for anomalous performers
- **General Best Practices**: Ongoing monitoring and improvement strategies

## Usage

### Automatic Generation
PDF reports are automatically generated when running:
- `Auto_weekly_generator_analyzer2.py` (full forecast analysis)
- `bid_validation.py` (bid validation analysis)

### Manual Generation
```python
from performance_report_generator import PerformanceReportGenerator

# Initialize generator
report_generator = PerformanceReportGenerator(config)

# Generate comprehensive report
pdf_filename = report_generator.generate_comprehensive_report(
    results_df=your_results_dataframe,
    anomalies_df=your_anomalies_dataframe,
    alerts=your_alerts_list,
    bid_validation_results=your_bid_validation_dataframe,
    market="miso",
    output_filename="custom_report_name.pdf"
)
```

## Dependencies

The PDF generation requires additional packages:
```bash
pip install matplotlib seaborn
```

If these packages are not available, the system will:
- Display a warning message
- Continue with normal CSV report generation
- Skip PDF generation gracefully

## Output Files

### Generator Analysis (Auto_weekly_generator_analyzer2.py)
- `generator_performance_report_{market}_{date}.pdf` - Comprehensive performance analysis
- Standard CSV files (unchanged)

### Bid Validation (bid_validation.py)
- `bid_validation_report_{market}_{timestamp}.pdf` - Bid validation analysis
- Standard CSV files (unchanged)

## Testing

Test the PDF generation functionality:
```bash
python test_pdf_generation.py
```

This creates a sample report with test data to verify the system works correctly.

## Report Content Details

### Bottom 10 Percentile Analysis
For each performance measure, the report identifies and lists generators in the bottom 10 percentile:

1. **Performance Score**: Composite score (0-100) based on RMSE, R², consistency, and volatility
2. **RMSE**: Root Mean Square Error in MW
3. **Consistency Score**: How consistent forecast errors are over time
4. **Volatility Score**: Standard deviation of forecast errors
5. **Statistical Anomalies**: Z-score analysis compared to population

### Visual Elements
- Performance distribution charts
- Error histograms
- Severity breakdowns
- Trend analysis plots
- Statistical scatter plots

### Data Tables
- Generator identification (name, plant ID, unit ID)
- Performance metrics and classifications
- Issue descriptions and recommendations
- Duration and severity information

## Troubleshooting

### Common Issues

1. **Import Error**: Install matplotlib and seaborn
   ```bash
   pip install matplotlib seaborn
   ```

2. **Memory Issues**: For large datasets, the system automatically limits table sizes
   ```python
   # Tables limited to top 10 entries to prevent overcrowding
   worst_performers.head(10)
   ```

3. **Missing Data**: The system handles missing data gracefully
   ```python
   # Handles None values and missing columns
   if 'column_name' in dataframe.columns:
       # Process data
   ```

### Error Recovery
If PDF generation fails, the system:
- Logs the error details
- Continues with normal CSV generation
- Provides helpful error messages
- Does not interrupt the main analysis

## Future Enhancements

Planned improvements include:
- Interactive PDF elements
- Multi-page detailed analysis
- Custom filtering options
- Email distribution capabilities
- Dashboard integration
