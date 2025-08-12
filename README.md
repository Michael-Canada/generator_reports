# Generator Analysis Platform

A comprehensive electricity market generator performance analysis platform providing advanced forecasting, anomaly detection, and bid validation capabilities across multiple markets (MISO, SPP, ERCOT).

## 🚀 Features

### Core Capabilities
- **Multi-Market Support**: MISO, SPP, ERCOT electricity markets
- **Performance Analytics**: 61% improvement in analysis performance through optimizations
- **Anomaly Detection**: Real-time identification of generator performance anomalies
- **Bid Validation**: Comprehensive bid analysis and validation
- **Chronic Error Detection**: Identification of persistent forecast errors
- **PDF Reporting**: Automated report generation with visualizations

### Technical Optimizations
- **Bulk API Operations**: Optimized data fetching with connection pooling
- **Parallel Processing**: 8-worker parallel execution for enhanced performance
- **Caching System**: Intelligent caching to reduce API calls
- **Error Handling**: Robust error handling and retry mechanisms

## 📊 Business Value

### Financial Risk Management
- **Forecast Accuracy**: Improved generator performance predictions
- **Market Intelligence**: Real-time insights into market dynamics
- **Risk Mitigation**: Early detection of performance anomalies

### Operational Excellence
- **Automated Analysis**: Reduces manual analysis time by 61%
- **Scalable Processing**: Handles large datasets efficiently
- **Real-time Monitoring**: Continuous performance tracking

### Strategic Intelligence
- **Market Trends**: Comprehensive market analysis
- **Competitive Analysis**: Performance benchmarking
- **Investment Insights**: Data-driven decision support

## 🛠 Installation

### Prerequisites
- Python 3.13.5+
- Conda package manager

### Setup
```bash
# Clone the repository
git clone <your-new-repo-url>
cd generator_reports

# Create conda environment
conda env create -f environment.yml
conda activate generator_analysis

# Install additional dependencies if needed
pip install -r requirements.txt
```

## 🔧 Configuration

### Environment Setup
Create a `.env` file with required API credentials:
```bash
# API Configuration
API_BASE_URL=your_api_url
API_KEY=your_api_key
API_SECRET=your_api_secret

# Database Configuration
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Performance Settings
MAX_WORKERS=8
CACHE_TTL=3600
BULK_SIZE=1000
```

## 🚦 Usage

### Basic Analysis
```bash
# Run main generator analyzer
python Auto_weekly_generator_analyzer2.py

# Run bid validation
python bid_validation.py

# Generate performance reports
python performance_report_generator.py
```

### Advanced Features
```bash
# Bid validation with integration
python bid_validation_integration.py

# Weekly PJM analysis (parallel processing)
python weekly_PJM_generator_analysis_parallel.py
```

## 📁 Project Structure

```
generator_reports/
├── Auto_weekly_generator_analyzer2.py    # Main analyzer with optimizations
├── performance_report_generator.py       # PDF report generation
├── bid_validation.py                     # Bid validation engine
├── bid_validation_integration.py         # Integration testing
├── weekly_PJM_generator_analysis_parallel.py  # Parallel PJM analysis
├── .conda/                               # Conda environment
├── docs/                                 # Documentation
│   ├── API_REFERENCE.md
│   ├── BID_VALIDATION_README.md
│   └── COLLABORATION_GUIDE.md
├── data/                                 # Data files (ignored by git)
│   ├── *.csv
│   ├── *.parquet
│   └── *.pkl
└── outputs/                              # Generated reports and analysis
    ├── *.pdf
    └── *.csv
```

## 📈 Performance Metrics

### Key Improvements
- **61% Performance Gain**: Through bulk API operations and parallel processing
- **Reduced API Calls**: Intelligent caching and bulk operations
- **Enhanced Reliability**: Robust error handling and retry mechanisms
- **Scalable Architecture**: Supports multiple markets and large datasets

### Market Coverage
- **MISO**: Comprehensive generator analysis
- **SPP**: Market performance tracking
- **ERCOT**: Bid validation and forecasting

## 🔍 Analysis Outputs

### Reports Generated
- Generator performance analysis (CSV)
- Anomaly detection reports (CSV)
- Forecast statistics (CSV)
- Performance reports (PDF)
- Bid validation summaries (CSV)
- Chronic error analysis (CSV)

### Key Metrics
- Forecast accuracy scores
- Performance rankings
- Anomaly alerts
- Market trend analysis
- Bid validation results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [API Reference](API_REFERENCE.md)
- [Bid Validation Guide](BID_VALIDATION_README.md)
- [Collaboration Guide](COLLABORATION_GUIDE.md)
- [Code Patterns](CODE_PATTERNS.md)

## 🐛 Troubleshooting

### Common Issues
1. **API Connection Errors**: Check API credentials and network connectivity
2. **Memory Issues**: Adjust worker count in configuration
3. **Data Loading Errors**: Verify data file formats and paths

### Performance Optimization
- Use bulk API operations when possible
- Enable caching for repeated queries
- Monitor memory usage with large datasets
- Adjust parallel processing workers based on system capacity

## 📄 License

This project is proprietary and confidential.

## 📧 Contact

For questions and support, please contact the development team.

---

**Generator Analysis Platform** - Transforming electricity market analysis through advanced analytics and automation.

## Original System Documentation

### Legacy### Data Structure

Key metrics in the results include:

- **Performance Metrics**: RMSE, MAE, R-squared, consistency scores
- **Classification**: 5-tier performance levels (excellent → critical)
- **Anomaly Detection**: Z-score analysis, chronic error patterns  
- **Alert System**: 4-level severity (low, medium, high, critical)
- **Generator Identification**: Complete plant IDs, unit IDs, multi-unit tracking

### What the Data Teaches Us About Each Generator

#### Operational Insights
- **Capacity Utilization**: `generation/pmax` ratio reveals operational patterns and efficiency
- **Must-Run Status**: Identifies baseload vs dispatchable generators based on operational constraints
- **Data Quality**: `quality_tag` indicates sensor reliability and data freshness for analysis confidence
- **Multi-Unit Coordination**: Tracks complex resource operational patterns and unit-level performance
- **Plant-Level Analysis**: `plant_id` enables facility-wide performance assessment
- **Unit-Level Granularity**: `unit_id` allows individual unit tracking within multi-unit plants

#### Forecast Performance Insights
- **Accuracy Metrics**: RMSE, MAE show prediction quality in absolute MW terms
- **Relative Performance**: `rmse_percentage_of_capacity` enables fair comparison across different generator sizes
- **Consistency**: `consistency_score` reveals forecast reliability and stability over time
- **Trends**: `trend_direction` shows improving/stable/deteriorating forecast patterns
- **Bias Detection**: Chronic over/under-forecasting patterns indicating systematic issues
- **Performance Ranking**: Comprehensive 0-100 scoring system for competitive benchmarking

#### Business Value & Economic Impact
- **Revenue Impact**: Poor forecasts directly affect market revenues and settlement accuracy
- **Operational Efficiency**: Identifies generators needing immediate operational attention
- **Strategic Bidding**: `bid_forecast_correlation` shows alignment between bidding strategy and forecast accuracy
- **Risk Management**: Early warning system for systematic forecast issues before they impact operations
- **Benchmarking**: Performance ranking enables peer comparison and best practice identification
- **Maintenance Planning**: Chronic error patterns may indicate equipment issues requiring maintenance
- **Market Participation**: Data quality tags help determine reliable generators for critical operations

#### Enhanced Generator Identification System
The system now provides complete generator identification through ResourceDB integration:
- **Plant Identification**: EIA plant IDs for regulatory compliance and facility-level analysis
- **Unit Tracking**: Individual unit IDs within multi-unit facilities for granular performance monitoring
- **Multi-Unit Resources**: Automatic detection and handling of resources with multiple generating units
- **Cross-Reference Capability**: Links between market UIDs and regulatory EIA identifiers

#### Quality Tag Interpretation (Data Reliability Guide)
The `quality_tag` in enhanced all_generators CSV indicates upstream data reliability:
- **VERY_GOOD**: Fresh data (<5 min delay), <1% missing data points, high sensor reliability
- **GOOD**: Recent data (<15 min delay), <5% missing data points, good sensor reliability  
- **AVERAGE**: Acceptable data (<60 min delay), <10% missing data points, moderate sensor reliability
- **BAD**: Stale data (>60 min delay), >10% missing data points, poor sensor reliability or communication issues

This quality assessment helps prioritize which generators can be relied upon for critical analysis and which may need data validation or alternative sources.

## Output Files and Resultson vs forecast generation to identify poor performing forecasts and anomalies
2. **Bid Validation**: Comprehensive validation tests for generator supply curves (bids) to identify potential configuration issues

### Key Features

- **High-Performance Processing**: Optimized parallel processing with intelligent batch sizing and connection pooling
- **Smart Data Filtering**: Automatic identification and removal of inactive/test generators to reduce processing overhead
- **Multi-Market Support**: MISO, SPP, ERCOT, and PJM markets with market-specific optimizations
- **Advanced Anomaly Detection**: Statistical analysis to identify chronic over/under-forecasting patterns
- **Bid Configuration Validation**: Automated tests for bid consistency and compliance
- **Optimized API Operations**: Connection reuse, retry logic, and bulk data fetching capabilities
- **Real-time Performance Monitoring**: Live performance metrics and optimization indicators
- **Integration Ready**: Modular design for easy integration into existing workflows
- **Comprehensive Reporting**: Detailed CSV outputs, executive summaries, and enhanced PDF reports

## System Architecture

### Core Modules

#### 1. Main Analysis Engine (`Auto_weekly_generator_analyzer2.py`)
**Purpose**: High-performance analysis engine processing generator forecast data with advanced parallel processing

**Key Classes**:
- `GeneratorAnalyzer`: Main analysis orchestrator with optimized batch processing
- `Config`: Centralized configuration management with market-specific settings
- `AnomalyDetector`: Statistical anomaly detection with chronic error pattern recognition
- `MetricsCalculator`: Advanced forecast accuracy metric calculations
- `APIClient`: Optimized API operations with connection pooling and retry logic

**Performance Features**:
- Parallel processing (8 workers, 300-generator batches)
- Smart generator filtering (removes inactive/test generators automatically)
- Connection pooling and retry mechanisms for API reliability
- Real-time performance monitoring with optimization indicators
- Memory-efficient processing for large datasets (10K+ generators)

#### 2. PDF Report Generator (`performance_report_generator.py`)
**Purpose**: Creates comprehensive, executive-ready PDF reports with visual analytics

**Key Features**:
- Multi-page executive dashboards with performance summaries
- Statistical anomaly detection visualizations
- Chronic error pattern analysis with actionable recommendations
- Performance classification systems with 5-tier ranking
- Bid validation integration for comprehensive generator assessment

#### 3. Bid Validation System (`bid_validation.py`, `bid_validation_integration.py`)
**Purpose**: Validates generator bid configurations and market participation patterns

**Validation Tests**:
- Pmin/Pmax compliance verification
- Bid curve consistency analysis
- Market participation pattern validation
- Price responsiveness assessment
- Configuration issue detection and reporting

### Data Flow Architecture

```
Market APIs → Data Validation → Parallel Processing → Analysis Engine → Report Generation
     ↓              ↓                    ↓               ↓              ↓
 Raw Data    →   Quality Checks   →   Batch Jobs   →   Metrics    →   PDF Reports
 Time Series     Missing Data        Generator        Anomalies      Executive
 Forecast/Actual  Filtering          Processing       Alerts         Dashboards
```

## Performance Classification System

### 5-Tier Performance Levels

The system classifies each generator using a sophisticated multi-metric approach:

#### Excellent (Score: 80-100)
- **RMSE**: ≤ 10% of generator capacity
- **R-squared**: ≥ 0.70
- **Business Impact**: Minimal forecast adjustments needed
- **Action**: Continue current practices, use as benchmark

#### Good (Score: 60-79)
- **RMSE**: ≤ 20% of generator capacity  
- **R-squared**: ≥ 0.60
- **Business Impact**: Minor forecast refinements may provide value
- **Action**: Monitor trends, consider optimization opportunities

#### Fair (Score: 40-59)
- **RMSE**: ≤ 30% of generator capacity
- **R-squared**: ≥ 0.50
- **Business Impact**: Moderate improvement potential
- **Action**: Schedule forecast model review within 30 days

#### Poor (Score: 20-39)
- **RMSE**: ≤ 40% of generator capacity
- **R-squared**: ≥ 0.20
- **Business Impact**: Significant forecast accuracy issues affecting operations
- **Action**: Immediate forecast model investigation required

#### Critical (Score: 0-19)
- **RMSE**: > 40% of generator capacity OR R-squared < 0.20
- **Business Impact**: Major operational and financial risks
- **Action**: Emergency forecast model overhaul and daily monitoring

### Performance Score Calculation

The composite performance score (0-100) is calculated as:
- **70% Weight**: Inverted RMSE percentage (lower RMSE = higher score)
- **20% Weight**: R-squared × 100 (higher correlation = higher score)
- **5% Weight**: Consistency score × 100 (more consistent = higher score)
- **5% Weight**: Inverted volatility score (lower volatility = higher score)
- `add_bid_validation_to_analyzer()`: Adds bid validation to GeneratorAnalyzer
- `enhance_final_reports_with_bid_validation()`: Integrates results into reports
- `create_enhanced_config_with_bid_validation()`: Configuration helpers

## Installation & Setup

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy scipy google-cloud-storage requests tqdm joblib

# For development
pip install pytest jupyter matplotlib seaborn
```

### Environment Setup

1. **Google Cloud Authentication**:
```bash
# Set up service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# Or use application default credentials
gcloud auth application-default login
```

2. **API Access**:
```bash
# Set up Marginal Unit API credentials
export MU_API_AUTH="username:password"
```

3. **Configuration**:
   - Update `Config` class in `Auto_weekly_generator_analyzer2.py`
   - Set market, batch size, and thresholds
   - Configure GCS bucket paths

## Usage

### Running the Analysis Directly

#### Generator Forecast Analysis

The simplest way to run the generator analysis is to execute the script directly:

```bash
python Auto_weekly_generator_analyzer2.py
```

This will:
1. **Initialize the high-performance analyzer** with optimized ResourceDB integration and smart data filtering
2. **Run comprehensive batch analysis** with parallel processing (8 workers) on all active generators in the configured market
3. **Apply intelligent filtering** to automatically exclude inactive and test generators for optimal performance
4. **Display a comprehensive final summary** with:
   - Total generators analyzed with plant_id/unit_id tracking and performance metrics
   - Anomalies and alerts detected with severity breakdown and processing statistics
   - Performance distribution across 5-tier classification system (excellent → critical)
   - Top 5 best and worst performers with complete identification
   - Chronic forecast error counts and patterns with trend analysis
   - Multi-unit resource analysis and coordination insights
   - Real-time performance monitoring and optimization indicators
5. **Save multiple enhanced CSV files** with detailed results and performance data:
   - `forecast_performance_summary_[market]_[date].csv` - Executive summary with market statistics
   - `generator_forecast_ranked_[market]_[date].csv` - All generators ranked by 0-100 performance score
   - `generator_anomaly_alerts_[market]_[date].csv` - Detected anomalies with chronic error analysis
   - `forecast_alerts_[market]_[date].csv` - All alerts with actionable recommendations
   - `chronic_forecast_errors_[market]_[date].csv` - Persistent forecast bias patterns
   - `all_generators_[market].csv` - **Enhanced generator inventory** with plant_id/unit_id columns
   - `CSV_Documentation_[market]_[date].txt` - **Comprehensive documentation** explaining all outputs

#### Bid Validation Analysis

You can also run bid validation as a standalone analysis:

```bash
python bid_validation.py
```

This will:
1. **Load supply curves and resource data** from Google Cloud
2. **Run comprehensive bid validation** on all generators
3. **Display detailed validation results** with:
   - Total validation issues found
   - Severity breakdown (critical, high, medium, low)
   - Issue type breakdown (missing bids, inconsistent curves, etc.)
   - Critical and high severity issues summary
4. **Save validation results** to CSV files:
   - `bid_validation_analysis_detailed_[market]_[timestamp].csv` - Detailed validation results (includes fuel_type column)
   - `bid_validation_analysis_summary_[market]_[timestamp].csv` - Summary statistics

### Programmatic Usage - Generator Analysis

```python
from Auto_weekly_generator_analyzer2 import GeneratorAnalyzer

# Initialize high-performance analyzer with optimized settings
analyzer = GeneratorAnalyzer()
analyzer.run_batch_analysis()  # Runs with intelligent filtering and parallel processing

# Get comprehensive results
results = analyzer.get_final_summary()

# Access specific components
summary_df = results['summary']           # Executive summary
all_results = results['results']          # All generator results  
anomalies = results['anomalies']          # Detected anomalies
alerts = results['alerts']               # Alert list
stats = results['stats']                 # Basic statistics

# Convenience methods for quick insights
top_10 = analyzer.get_top_performers(10)        # Top 10 performers
worst_10 = analyzer.get_worst_performers(10)    # Worst 10 performers
critical = analyzer.get_critical_alerts()       # Critical alerts only
chronic = analyzer.get_chronic_error_generators() # Chronic error generators

# Access optimized performance statistics
stats = analyzer.get_analysis_statistics()
print(f"Analyzed {stats['total_generators_analyzed']} generators")
print(f"Filtered out {stats.get('inactive_generators_filtered', 0)} inactive generators")
print(f"Found {stats['total_anomalies_detected']} anomalies")
print(f"Generated {stats['total_alerts_generated']} alerts")
print(f"Processing time per generator: {stats.get('avg_processing_time_per_generator', 'N/A')}s")
```

### Quick Start - Bid Validation

```python
from bid_validation import BidValidator

# Initialize validator
validator = BidValidator(market="miso")

# Load data and run validation
validator.load_cloud_data()
results = validator.run_comprehensive_validation()

# View results
print(f"Found {len(results)} validation issues")
validator.save_results("validation_results")
```

### Running Both Analyses

You can run both generator forecast analysis and bid validation:

```bash
# Run high-performance generator analysis first
python Auto_weekly_generator_analyzer2.py

# Then run bid validation separately  
python bid_validation.py

# Or run them in sequence with optimized processing
python Auto_weekly_generator_analyzer2.py && python bid_validation.py
```

### Integrated Usage

```python
from Auto_weekly_generator_analyzer2 import GeneratorAnalyzer, Config
from bid_validation_integration import add_bid_validation_to_analyzer

# Initialize analyzer with bid validation
config = Config()
config.BID_VALIDATION['enable_bid_validation'] = True

analyzer = GeneratorAnalyzer(config)
add_bid_validation_to_analyzer(analyzer)

# Run comprehensive analysis
analyzer.run_batch_analysis()  # Includes both forecast and bid analysis
```

## Configuration

### Market Configuration

```python
class Config:
    # Market selection
    MARKET = "miso"  # Options: "miso", "spp", "ercot", "pjm"
    
    # Market-specific settings
    MARKET_CONFIGS = {
        "miso": {
            "collection": "miso-se",
            "timezone_offset": "T04:00:00-05:00",
            "run_version": "v2"
        },
        # ... other markets
    }
    
    # ResourceDB Integration for Enhanced Generator Identification
    RESOURCEDB_INTEGRATION = {
        'enable_resourcedb': True,
        'bucket_name': 'marginalunit-placebo-metadata',
        'file_paths': {
            'miso': 'metadata/miso.resourcedb/2024-11-19/resources.json',
            'spp': 'metadata/spp.resourcedb/2024-11-19/resources.json',
            'ercot': 'metadata/ercot.resourcedb.v2/2024-11-25/resources.json',
            'pjm': 'metadata/pjm.resourcedb/2024-11-19/resources.json'
        }
    }
```

### Performance Thresholds

```python
# Anomaly detection configuration
ANOMALY_DETECTION = {
    "rmse_threshold_zscore": 2.0,
    "performance_thresholds": {
        "excellent": {"rmse_pct_max": 2.0, "r2_min": 0.95},
        "good": {"rmse_pct_max": 5.0, "r2_min": 0.85},
        "fair": {"rmse_pct_max": 10.0, "r2_min": 0.70},
        "poor": {"rmse_pct_max": 20.0, "r2_min": 0.50},
        "critical": {"rmse_pct_max": float('inf'), "r2_min": 0.0}
    }
}
```

### Bid Validation Configuration

```python
BID_VALIDATION = {
    'enable_bid_validation': True,
    'validation_thresholds': {
        'pmin_tolerance': 0.05,          # 5% tolerance for pmin validation
        'generation_percentile': 80,      # Use 80th percentile for generation checks
        'pmax_ratio_threshold': 0.9,      # 90% of pmax threshold
        'price_jump_factor': 10.0,        # Maximum reasonable price jump factor
        'min_data_points': 168,           # Minimum hours of generation data
        'lookback_hours': 1000,           # Hours to look back for analysis
    }
}
```

## Output Files and Results

### Console Output

When running `python Auto_weekly_generator_analyzer2.py`, you'll see:

1. **Progress Updates**: Real-time processing status for each generator batch
2. **Comprehensive Final Summary**: Detailed analysis results including:
   ```
   ================================================================================
   FINAL ANALYSIS SUMMARY - MISO MARKET
   Analysis Date: 2025-08-09
   ================================================================================
   
   📊 OVERALL STATISTICS:
     • Total generators analyzed: 789
     • Total generating units tracked: 1,247 (including multi-unit resources)
     • Multi-unit generators: 156 resources with multiple units
     • Anomalies detected: 23
     • Total alerts generated: 12
     • High-severity alerts: 5
     • Critical alerts: 2
   
   ⚠️ CHRONIC FORECAST ERRORS:
     • Chronic over-forecasting: 3 generators
     • Chronic under-forecasting: 2 generators
   
   🎯 PERFORMANCE DISTRIBUTION:
     • Excellent: 285 generators (36.1%)
     • Good: 251 generators (31.8%)
     • Fair: 165 generators (20.9%)
     • Poor: 61 generators (7.7%)
     • Critical: 27 generators (3.4%)
   
   🚨 Poor performing generators: 88 (11.1%)
   
   🏆 TOP 5 BEST PERFORMERS:
     • Wind Farm Alpha (Plant 12345, Unit W001): 98.2 (excellent)
     • Nuclear Unit Beta (Plant 54321, Unit N002): 96.8 (excellent)
     • Gas Turbine Gamma (Plant 98765, Unit G003): 94.7 (excellent)
     • Coal Plant Delta (Plant 11111, Unit C001): 92.3 (excellent)
     • Hydro Station Echo (Plant 22222, Unit H004): 91.8 (excellent)
   
   💔 TOP 5 WORST PERFORMERS:
     • Old Coal Unit Zeta (Plant 99999, Unit C099): 18.3 (critical)
     • Peaker Plant Theta (Plant 88888, Unit G088): 22.7 (critical)
     • Wind Farm Iota (Plant 77777, Unit W077): 28.9 (poor)
     • Gas Unit Kappa (Plant 66666, Unit G066): 31.2 (poor)
     • Steam Unit Lambda (Plant 55555, Unit S055): 33.8 (poor)
   
   💾 REPORTS SAVED:
     • Summary: forecast_performance_summary_miso_2025-08-09.csv
     • Ranked Results: generator_forecast_ranked_miso_2025-08-09.csv
     • Enhanced Generator Inventory: all_generators_miso.csv (with plant_id/unit_id)
     • Anomalies: generator_anomaly_alerts_miso_2025-08-09.csv
     • Alerts: forecast_alerts_miso_2025-08-09.csv
     • Chronic Errors: chronic_forecast_errors_miso_2025-08-09.csv
     • Documentation: CSV_Documentation_miso_2025-08-09.txt
   ```

### Generated CSV Files

#### Core Analysis Files
- **`forecast_performance_summary_{market}_{date}.csv`**: Executive summary with market-level statistics
- **`generator_forecast_ranked_{market}_{date}.csv`**: All generators ranked by performance score (0-100)
- **`generator_anomaly_alerts_{market}_{date}.csv`**: Detailed anomaly detection results
- **`forecast_alerts_{market}_{date}.csv`**: All alerts with severity levels and recommendations
- **`chronic_forecast_errors_{market}_{date}.csv`**: Generators with persistent forecasting issues

#### Enhanced Data Files
- **`all_generators_{market}.csv`**: Complete generator inventory enhanced with:
  - `plant_id`: EIA plant identification numbers
  - `unit_id`: EIA unit identifiers (or MULTI_X_UNITS for multi-unit resources)
  - `total_units`: Number of units in each resource
  - `multi_unit`: Boolean indicating multi-unit resources
  - `pmax`: Maximum generation capacity (MW)
  - `generation`: Current actual generation output (MW)
  - `quality_tag`: Data quality indicator (VERY_GOOD/GOOD/AVERAGE/BAD)

#### Documentation Files
- **`CSV_Documentation_{market}_{date}.txt`**: Comprehensive guide explaining:
  - What each CSV file contains and teaches us
  - Column-by-column insights and business value
  - Quality tag determination logic
  - How to use the data for operational decisions

#### Batch Processing Files (per batch)
- **`generator_analyzer_{market}_{batch}_{date}.csv`**: Raw timeseries data for the batch
- **`generator_forecast_stats_{market}_{batch}_{date}.csv`**: Statistical metrics for each generator
- **`generator_anomalies_{market}_{batch}_{date}.csv`**: Anomalies detected in the batch

### Data Structure

Key metrics in the results include:

- **Performance Metrics**: RMSE, MAE, R-squared, consistency scores
- **Classification**: 5-tier performance levels (excellent → critical)
- **Anomaly Detection**: Z-score analysis, chronic error patterns  
- **Alert System**: 4-level severity (low, medium, high, critical)
- **Generator Identification**: Complete plant IDs, unit IDs, multi-unit tracking

## Output Files

### Generator Analysis Outputs

- `generator_analyzer_{market}_{batch}_{date}.csv`: Detailed timeseries data
- `generator_forecast_stats_{market}_{batch}_{date}.csv`: Summary statistics
- `generator_anomalies_{market}_{batch}_{date}.csv`: Anomaly detection results
- `forecast_performance_summary_{market}_{date}.csv`: Executive summary

### Bid Validation Outputs

- `bid_validation_results_{date}.csv`: Detailed validation results
- `bid_validation_summary_{date}.csv`: Summary statistics by validation type
- `bid_validation_critical_{date}.csv`: Critical issues requiring immediate attention

## Data Sources

### Google Cloud Storage
- **Bucket**: `marginalunit-placebo-metadata`
- **Resources**: Generator capacity and metadata (`resources.json`)
- **Supply Curves**: Bid block data (`supply_curves.json`)

### API Endpoints
- **Base URL**: `https://api1.marginalunit.com/muse/api`
- **Generation Data**: `/reflow/data/generation`
- **Forecast Data**: `/reflow/data/forecast`

### Market Collections
- **MISO**: `miso-se`
- **SPP**: `spp-se`
- **ERCOT**: `ercot-rt-se`
- **PJM**: `pjm-se`

## Performance Optimization

### Parallel Processing
- Configurable batch sizes (default: 200 generators)
- Multi-threading support (default: 4 workers)
- Memory-efficient data loading

### Caching Strategy
- GCS data caching for repeated analysis
- API response caching with TTL
- Incremental processing support

### Resource Management
- Automatic memory cleanup
- Progress tracking with `tqdm`
- Graceful error handling

## Development

### Project Structure
```
generator_analysis/
├── Auto_weekly_generator_analyzer2.py    # Main analysis engine
├── bid_validation.py                     # Bid validation core
├── bid_validation_integration.py         # Integration helpers
├── bid_validation_example.py             # Usage examples
├── README.md                             # This file
├── BID_VALIDATION_README.md              # Detailed bid validation docs
├── SYSTEM_KNOWLEDGE_BASE.md              # Comprehensive system knowledge
└── *.csv                                 # Data and output files
```

### Testing

```bash
# Run basic system test
python bid_validation_example.py

# Test core components
python -c "from bid_validation import BidValidator; print('✅ Import successful')"
python -c "from Auto_weekly_generator_analyzer2 import GeneratorAnalyzer; print('✅ Import successful')"
```

### Extensions

The system is designed for extensibility:

1. **Retirement Filtering (August 2025)**
   - Both bid validation and main analyzer now filter out retired generators
   - Uses `end_date` field from ResourceDB to exclude generators with past retirement dates
   - Significantly improves analysis accuracy by focusing only on active units
   - Example: MISO analysis filters 260+ retired generators automatically

2. **Fuel Type Enhancement (August 2025)**
   - Added `fuel_type` column to all bid validation outputs
   - Uses EIA energy source codes (NG=Natural Gas, BIT=Bituminous Coal, NUC=Nuclear, etc.)
   - Enables fuel-specific analysis and validation patterns

1. **Custom Validation Rules**: Extend `BidValidator` class
2. **Additional Markets**: Add market configs to `MARKET_CONFIGS`
3. **New Metrics**: Extend `AnomalyMetrics` dataclass
4. **Custom Reporting**: Override report generation methods

## Troubleshooting

### Common Issues

1. **GCS Authentication**: Ensure credentials are properly configured for ResourceDB integration
2. **API Rate Limits**: Implement retry logic for production use
3. **Memory Usage**: Reduce batch sizes for large datasets (default: 200 generators per batch)
4. **Missing Data**: Check generator names, date ranges, and data quality tags
5. **ResourceDB Loading**: Verify Google Cloud Storage access and resource file paths

### Performance Tips

1. Use appropriate batch sizes based on available memory (adjust BATCH_SIZE in Config)
2. Enable parallel processing for large datasets (adjust N_JOBS in Config)
3. Cache frequently accessed data and enable GCS caching
4. Monitor API usage and implement rate limiting
5. Use ResourceDB integration for complete generator identification

### New Features Troubleshooting

1. **Enhanced CSV Output**: If plant_id/unit_id columns are missing, check ResourceDB integration settings
2. **CSV Documentation**: Auto-generated documentation files explain all outputs and insights
3. **Final Summary Display**: Summary always displays regardless of SAVE_RESULTS setting
4. **Multi-Unit Resources**: Check unit_details column for complex resource information

## Contributing

1. Follow existing code patterns and documentation standards
2. Add tests for new functionality
3. Update documentation for new features (all .md files)
4. Ensure backwards compatibility
5. Test with multiple markets (MISO, SPP, ERCOT, PJM)
6. Verify ResourceDB integration and enhanced CSV outputs

## Support

For technical issues:
1. Check `SYSTEM_KNOWLEDGE_BASE.md` for detailed troubleshooting and system architecture
2. Review `BID_VALIDATION_README.md` for bid validation specifics
3. Examine example usage in `bid_validation_example.py`
4. Check `CSV_Documentation_{market}_{date}.txt` for output file explanations
5. Review `AI_CONTEXT_QUICK_REF.md` for current system status

---

**Last Updated**: August 10, 2025  
**Version**: 2.3 with enhanced bid validation tests and nuclear generator exclusion  
**Supported Markets**: MISO, SPP, ERCOT, PJM  
**Key Features**: Complete generator identification, automated CSV documentation, comprehensive final summaries, fuel type columns, retirement filtering, enhanced bid validation with nuclear exclusion and Pmax capacity validation  
**System Status**: Enhanced (nuclear generator exclusion and Pmax validation added to bid validation)
