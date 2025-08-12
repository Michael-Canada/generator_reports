# Code Patterns and Best Practices

## Generator Forecast Performance Analysis System

This document establishes comprehensive coding standards and patterns for the Generator Forecast Performance Analysis Platform, ensuring consistent, maintainable, and high-performance code across all components.

## Table of Contents

1. [Architecture Patterns](#architecture-patterns)
2. [Data Processing Patterns](#data-processing-patterns)
3. [Performance Optimization Patterns](#performance-optimization-patterns)
4. [Error Handling Patterns](#error-handling-patterns)
5. [Configuration Management](#configuration-management)
6. [Testing Patterns](#testing-patterns)
7. [Logging and Monitoring](#logging-and-monitoring)
8. [API Design Patterns](#api-design-patterns)

## High-Performance Code Implementation Patterns

### Key Optimization Implementation Patterns

### 1. Smart Generator Filtering Pattern (Performance Optimization)
```python
# High-performance filtering pattern to exclude inactive generators
def _filter_active_generators_optimized(self, all_generators):
    """Intelligent filtering to remove inactive/test generators for optimal performance"""
    active_generators = []
    filtered_count = 0
    
    for generator in all_generators:
        # Apply smart filtering logic
        if self._is_generator_active(generator):
            active_generators.append(generator)
        else:
            filtered_count += 1
    
    print(f"Smart filtering: Excluded {filtered_count} inactive generators")
    print(f"Processing {len(active_generators)} active generators for optimal performance")
    return active_generators

def _is_generator_active(self, generator_name):
    """Enhanced logic to determine if generator should be processed"""
    # Implementation includes activity patterns, data quality checks, etc.
    return True  # Simplified for pattern example
```

### 2. Enhanced Parallel Processing Pattern (Optimized Batch Operations)
```python
# Optimized parallel processing with intelligent batch sizing
def _analyze_batch_optimized(self, batch_generators):
    """High-performance batch processing with enhanced API operations"""
    try:
        # Bulk data fetching with connection reuse
        bulk_data = self.api_client.get_batch_generators_data(batch_generators)
        
        # Parallel processing with optimal worker configuration
        with ProcessPoolExecutor(max_workers=self.config.N_JOBS) as executor:
            futures = []
            
            for generator in batch_generators:
                future = executor.submit(
                    self._process_generator_with_prefetched_data,
                    generator,
                    bulk_data.get(generator, {})
                )
                futures.append(future)
            
            # Collect results with performance monitoring
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.PROCESSING_TIMEOUT)
                    results.append(result)
                except Exception as e:
                    self._log_processing_error(e)
            
            return results
            
    except Exception as e:
        print(f"Batch processing error: {e}")
        return []
```

### 3. Enhanced API Client Pattern (Connection Pooling & Retry Logic)
```python
# Optimized API client with connection pooling and intelligent retry
class EnhancedAPIClient:
    def __init__(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def get_batch_generators_data(self, generator_list):
        """Bulk data fetching with connection reuse"""
        results = {}
        
        # Optimized bulk API calls
        for batch in self._create_api_batches(generator_list):
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json={'generators': batch},
                    timeout=self.config.API_TIMEOUT
                )
                response.raise_for_status()
                batch_results = response.json()
                results.update(batch_results)
                
            except Exception as e:
                self._handle_api_error(e, batch)
                
        return results
```

### 4. Enhanced Data Loading Pattern (GCS + API + ResourceDB + Performance Monitoring)
```python
# High-performance data loading with smart caching and monitoring
def load_cloud_data_optimized(self):
    start_time = time.time()
    try:
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        
        # Parallel loading of resources and supply curves
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Resources.json (main resource database)
            resources_future = executor.submit(self._load_resources_data, bucket)
            
            # Supply_curves.json
            supply_future = executor.submit(self._load_supply_curves_data, bucket)
            
            # Wait for completion
            self.resource_db = resources_future.result()
            self.supply_curves = supply_future.result()
        
        load_time = time.time() - start_time
        print(f"High-performance loading: {len(self.resource_db)} resources in {load_time:.2f}s")
        
        # Performance monitoring
        self._update_performance_stats('data_loading_time', load_time)
        return True
        
    except Exception as e:
        print(f"Error in optimized data loading: {e}")
        return False

def _load_resources_data(self, bucket):
    """Optimized resource data loading with caching"""
    resources_blob = bucket.blob(f"{self.base_path}resources.json")
    resources_text = resources_blob.download_as_text()
    return {r['uid']: r for r in json.loads(resources_text)}
```

### 5. Real-time Performance Monitoring Pattern
```python
# Performance monitoring and optimization tracking
def _update_performance_stats(self, metric_name, value):
    """Real-time performance monitoring"""
    if not hasattr(self, 'performance_stats'):
        self.performance_stats = {}
    
    self.performance_stats[metric_name] = value
    
    # Calculate derived metrics
    if metric_name == 'processing_time_per_generator':
        self.performance_stats['generators_per_second'] = 1.0 / value if value > 0 else 0
    
def get_optimization_indicators(self):
    """Return real-time optimization indicators"""
    return {
        'avg_processing_time_per_generator': self.performance_stats.get('processing_time_per_generator', 0),
        'generators_per_second': self.performance_stats.get('generators_per_second', 0),
        'inactive_generators_filtered': self.performance_stats.get('filtered_generators', 0),
        'api_efficiency_ratio': self.performance_stats.get('api_efficiency', 1.0),
        'memory_usage_optimization': self.performance_stats.get('memory_efficiency', 1.0)
    }
```

### 6. Enhanced ResourceDB Integration Pattern (Optimized)
    # Add identification columns
    all_generators_df['plant_id'] = None
    all_generators_df['unit_id'] = None
    all_generators_df['total_units'] = 1
    all_generators_df['multi_unit'] = False
    all_generators_df['unit_details'] = None
    
    # Enhance with ResourceDB data
    for idx, row in all_generators_df.iterrows():
        generator_uid = row['uid']
        if generator_uid in self.resource_db:
            resource = self.resource_db[generator_uid]
            # Extract plant_id, unit_id, multi-unit information
            # ... enhancement logic
    
    return all_generators_df
```

### 2. API Request Pattern (Enhanced Generation Data)
```python
# Standard pattern for API calls with enhanced error handling
def get_generation_data(self, generator_name, start_time, end_time):
    url = f"{self.url_root}/reflow/data/generation"
    params = {
        'collection': self.reflow_collections[self.market],
        'resource_name': generator_name,
        'start_time': start_time,
        'end_time': end_time
    }
    
    try:
        response = requests.get(url, params=params, auth=self.auth)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"Error fetching generation data for {generator_name}: {e}")
        return None
```

### 3. Enhanced Validation Result Pattern
```python
# Consistent result structure with complete identification
@dataclass
class BidValidationResult:
    generator_name: str
    plant_id: Optional[str]
    unit_id: Optional[str]
    validation_type: BidValidationType
    severity: BidValidationLevel
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

# Enhanced main() function pattern for standalone execution
def main():
    """Enhanced main function with comprehensive reporting."""
    print("=" * 80)
    print("BID VALIDATION ANALYSIS")
    print("=" * 80)
    
    validator = BidValidator()
    
    # Load data with status reporting
    if not validator.load_cloud_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Display structured results with detailed breakdown
    validator._display_validation_summary(results)
    
    # Save results with confirmation
    if validator.save_results():
        print("✅ Results saved successfully")
    
    print("=" * 80)
```

### 4. CSV Documentation Generation Pattern
```python
# Automatic documentation generation pattern
def generate_csv_documentation(self) -> str:
    """Generate comprehensive documentation about all CSV files."""
    doc = """
=== CSV OUTPUT FILES DOCUMENTATION ===

The generator analysis system produces multiple CSV files, each providing specific insights:

1. ENHANCED all_generators_{market}.csv
   Purpose: Complete generator inventory with identification and current status
   Key Columns & Insights:
   - plant_id: EIA plant identification number
   - unit_id: EIA unit identifier
   - pmax/generation: Capacity vs actual output analysis
   - quality_tag: Data reliability indicator
   
   What it teaches us:
   - Generator capacity utilization patterns
   - Data reliability for analysis purposes
   - Multi-unit operational coordination
   
2. generator_forecast_ranked_{market}_{date}.csv
   Purpose: Comprehensive performance ranking
   ...
   """
   return doc

def save_csv_documentation(self) -> None:
    """Save documentation to timestamped file."""
    doc = self.generate_csv_documentation()
    filename = f"CSV_Documentation_{self.config.MARKET}_{self.today_date_str}.txt"
    with open(filename, 'w') as f:
        f.write(doc)
    print(f"CSV documentation saved to: {filename}")
```

### 5. Enhanced Configuration Integration Pattern
```python
# How ResourceDB and documentation integrate with existing Config
class Config:
    # Enhanced ResourceDB Integration
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
    
    BID_VALIDATION = {
        'enable_bid_validation': True,
        'validation_thresholds': {
            'pmin_tolerance': 0.05,
            'generation_percentile': 80,
            'pmax_ratio_threshold': 0.9,
            'price_jump_factor': 10.0,
            'min_data_points': 168,
            'lookback_hours': 1000,
        },
        'gcs_config': {
            'bucket_name': 'marginalunit-placebo-metadata',
            'base_paths': {...}
        }
    }
```

### 5. Enhanced Error Handling Pattern
```python
# Graceful degradation pattern used throughout with enhanced error context
def validate_something(self, generator_name):
    try:
        # Main validation logic with ResourceDB integration
        if generator_name in self.resource_db:
            resource = self.resource_db[generator_name]
            # Enhanced validation with complete context
        else:
            # Fallback to basic validation
        
    except KeyError as e:
        print(f"Missing data for {generator_name}: {e}")
        return None
    except Exception as e:
        print(f"Validation error for {generator_name}: {e}")
        return None

# Enhanced Final Summary Display Pattern
def _display_final_summary(self, ranked_results, anomalies, alerts, summary):
    """Always-display comprehensive final summary regardless of save settings."""
    print(f"\n" + "=" * 80)
    print(f"FINAL ANALYSIS SUMMARY - {self.config.MARKET.upper()} MARKET")
    print(f"Analysis Date: {self.today_date_str}")
    print(f"=" * 80)
    
    # Overall statistics with multi-unit tracking
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  • Total generators analyzed: {len(ranked_results)}")
    if 'total_units' in ranked_results.columns:
        total_units = ranked_results['total_units'].sum()
        multi_unit_count = len(ranked_results[ranked_results['total_units'] > 1])
        print(f"  • Total generating units tracked: {total_units} (including multi-unit resources)")
        print(f"  • Multi-unit generators: {multi_unit_count} resources with multiple units")
    
    # Performance distribution with percentages
    perf_counts = ranked_results['performance_classification'].value_counts()
    total = len(ranked_results)
    print(f"\n🎯 PERFORMANCE DISTRIBUTION:")
    for perf in ['excellent', 'good', 'fair', 'poor', 'critical']:
        count = perf_counts.get(perf, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  • {perf.capitalize()}: {count} generators ({percentage:.1f}%)")
    
    # Top and bottom performers with complete identification
    print(f"\n🏆 TOP 5 BEST PERFORMERS:")
    top_5 = ranked_results.head(5)
    for idx, row in top_5.iterrows():
        plant_info = f" (Plant {row['plant_id']}, Unit {row['unit_id']})" if 'plant_id' in row else ""
        print(f"  • {row['name']}{plant_info}: {row['performance_score']:.1f} ({row['performance_classification']})")
```
        if generator_name not in self.data_source:
            return None  # Graceful skip
            
        result = perform_validation()
        return result
        
    except DataNotFoundError:
        print(f"Warning: Data not found for {generator_name}")
        return None
    except Exception as e:
        print(f"Error validating {generator_name}: {e}")
        return None
```

### 6. Integration Helper Pattern
```python
# Clean integration functions
def add_bid_validation_to_analyzer(analyzer):
    """Add bid validation capability to existing GeneratorAnalyzer."""
    if not hasattr(analyzer, 'config'):
        raise ValueError("Analyzer must have config attribute")
    
    # Initialize validator with analyzer's config
    config = analyzer.config.get_bid_validation_config()
    analyzer.bid_validator = BidValidator(
        market=analyzer.config.MARKET,
        config=config
    )
    
    # Add validation method
    def run_bid_validation(self):
        print("Running comprehensive bid validation...")
        if not self.bid_validator.load_cloud_data():
            print("Failed to load bid validation data")
            return
        
        results = self.bid_validator.run_comprehensive_validation()
        self.bid_validation_results = results
        print(f"Bid validation complete: {len(results)} issues found")
    
    analyzer.run_bid_validation = run_bid_validation.__get__(analyzer)
```

### 7. Data Structure Access Patterns
```python
# How to safely access nested data
def _get_plant_id(self, generator_name):
    if generator_name not in self.resource_db:
        return None
    
    resource = self.resource_db[generator_name]
    generators = resource.get('generators', [])
    
    if generators:
        eia_uid = generators[0].get('eia_uid', {})
        return str(eia_uid.get('eia_id', ''))
    
    return None

# Resource-level capacity access
def _get_resource_capacity(self, generator_name):
    if generator_name not in self.resource_db:
        return None, None
    
    resource = self.resource_db[generator_name]
    physical_props = resource.get('physical_properties', {})
    
    return physical_props.get('pmin', 0), physical_props.get('pmax', 0)

# Supply curve access
def _get_bid_blocks(self, generator_name):
    if generator_name not in self.supply_curves:
        return []
    
    supply_curve = self.supply_curves[generator_name]
    offer_curve = supply_curve.get('offer_curve', {})
    return offer_curve.get('blocks', [])
```

### 8. Multi-Unit Detection Pattern (Information Only)
```python
# How to identify and log multi-unit resources
def log_multi_unit_resource(self, generator_name):
    if generator_name not in self.resource_db:
        return
    
    resource = self.resource_db[generator_name]
    generators = resource.get('generators', [])
    
    if len(generators) <= 1:
        return  # Single unit
    
    # Log multi-unit resource details
    physical_props = resource.get('physical_properties', {})
    resource_pmax = physical_props.get('pmax', 0)
    
    print(f"INFO: Multi-unit resource found: {generator_name}")
    print(f"      Units: {len(generators)}, Resource Pmax: {resource_pmax:.1f} MW")
    
    for i, gen in enumerate(generators, 1):
        eia_uid = gen.get('eia_uid', {})
        plant_id = eia_uid.get('eia_id', 'N/A')
        unit_id = eia_uid.get('unit_id', 'N/A')
        print(f"        Unit {i}: Plant {plant_id}, Unit {unit_id}")
```

### 9. Validation Test Template
```python
# Template for implementing new validation tests
def validate_new_rule(self, generator_name) -> Optional[BidValidationResult]:
    """
    Template for implementing new validation rules.
    
    Args:
        generator_name: Name of generator to validate
        
    Returns:
        BidValidationResult if issue found, None otherwise
    """
    try:
        # 1. Check data availability
        if generator_name not in self.required_data_source:
            return None
        
        # 2. Extract relevant data
        data = self._get_relevant_data(generator_name)
        
        # 3. Apply validation logic
        if self._validation_condition(data):
            # 4. Create result if issue found
            return BidValidationResult(
                generator_name=generator_name,
                plant_id=self._get_plant_id(generator_name),
                unit_id=self._get_unit_id(generator_name),
                validation_type=BidValidationType.NEW_RULE,
                severity=self._determine_severity(data),
                message=f"Validation message: {data}",
                details={'relevant_data': data},
                recommendations=["Fix suggestion 1", "Fix suggestion 2"],
                timestamp=datetime.now().isoformat()
            )
        
        return None  # No issue found
        
    except Exception as e:
        print(f"Error in new validation rule for {generator_name}: {e}")
        return None
```

## Testing Patterns

### Unit Test Template
```python
def test_validation_rule():
    validator = BidValidator(market="miso")
    
    # Mock data setup
    validator.resource_db = {"TEST_GEN": test_resource_data}
    validator.supply_curves = {"TEST_GEN": test_supply_data}
    
    # Test positive case
    result = validator.validate_new_rule("TEST_GEN")
    assert result is not None
    assert result.validation_type == BidValidationType.NEW_RULE
    
    # Test negative case
    result = validator.validate_new_rule("GOOD_GEN")
    assert result is None
```

### Integration Test Pattern
```python
def test_full_integration():
    # Test with real data sources
    validator = BidValidator(market="miso")
    assert validator.load_cloud_data()
    
    # Test on known generators
    results = validator.run_comprehensive_validation(["KNOWN_GENERATOR"])
    assert isinstance(results, pd.DataFrame)
    
    # Verify output structure
    expected_columns = ['generator_name', 'validation_type', 'severity', 'message']
    assert all(col in results.columns for col in expected_columns)
```

---

## Key Architectural Decisions

1. **Modular Design**: Separate validation engine from integration helpers
2. **Graceful Degradation**: System continues if some data is missing
3. **Configuration-Driven**: All thresholds and settings in Config class
4. **Dataclass Results**: Structured, type-safe validation results
5. **Market Agnostic**: Same code works across MISO, SPP, ERCOT, PJM
6. **API-First**: Uses existing API patterns and authentication
7. **Cloud-Native**: Direct GCS integration for metadata

These patterns can be extended for future enhancements while maintaining consistency with the existing codebase.
