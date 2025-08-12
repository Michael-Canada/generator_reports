"""
Performance Report Generator - Fixed Version

This module generates comprehensive PDF reports describing all performance measures
and listing poorly performing generators for forecast analysis and bid validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from textwrap import wrap
import warnings
warnings.filterwarnings('ignore')

class PerformanceReportGenerator:
    """Generates comprehensive PDF reports for generator performance analysis."""
    
    def __init__(self, config=None):
        self.config = config
        self.today_date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _get_pmax_from_resource_db(self, generator_name: str) -> float:
        """Get Pmax value from resource_db (resources.json)."""
        if not hasattr(self, 'resource_db') or not self.resource_db:
            print(f"Debug: No resource_db available for {generator_name} (resource_db: {getattr(self, 'resource_db', 'Not set')})")
            return None
            
        if generator_name not in self.resource_db:
            print(f"Debug: Generator {generator_name} not found in resource_db (available: {len(self.resource_db)} resources)")
            # Show a few example keys for debugging
            example_keys = list(self.resource_db.keys())[:3]
            print(f"Debug: Example resource_db keys: {example_keys}")
            return None
            
        try:
            resource = self.resource_db[generator_name]
            print(f"Debug: Found resource for {generator_name}: {resource.keys() if isinstance(resource, dict) else type(resource)}")
            generators = resource.get('generators', [])
            
            if generators:
                # Get pmax from the first generator (should be same for all units in the resource)
                pmax_value = generators[0].get('pmax')
                if pmax_value is not None:
                    print(f"Debug: Found Pmax {pmax_value} for {generator_name} in resource_db")
                    return float(pmax_value)
                else:
                    print(f"Debug: No 'pmax' key in first generator for {generator_name}, keys: {generators[0].keys()}")
            else:
                print(f"Debug: No generators list found for {generator_name}")
        except (KeyError, ValueError, TypeError) as e:
            print(f"Debug: Error getting Pmax for {generator_name}: {e}")
            
        return None
    
    def _get_pmax_alternative(self, generator_name: str) -> str:
        """Alternative method to get Pmax when resource_db lookup fails."""
        print(f"Debug: Attempting alternative Pmax lookup for {generator_name}")
        
        # Try multiple data sources for Pmax/capacity information
        
        # First try anomalies_df
        if hasattr(self, 'anomalies_df') and self.anomalies_df is not None and not self.anomalies_df.empty:
            print(f"Debug: Checking anomalies_df with {len(self.anomalies_df)} rows")
            print(f"Debug: anomalies_df columns: {list(self.anomalies_df.columns)}")
            matching_rows = self.anomalies_df[self.anomalies_df['generator_name'] == generator_name]
            print(f"Debug: Found {len(matching_rows)} matching rows in anomalies_df for {generator_name}")
            if not matching_rows.empty:
                # Look for capacity information in anomalies data
                for col in ['generator_capacity_mw', 'pmax', 'capacity', 'nameplate_capacity']:
                    if col in matching_rows.columns:
                        value = matching_rows[col].iloc[0]
                        if pd.notna(value) and value > 0:
                            print(f"Debug: Found Pmax {value} for {generator_name} in anomalies_df column {col}")
                            return f"{float(value):.1f}"

        # Try results_df  
        if hasattr(self, 'results_df') and self.results_df is not None and not self.results_df.empty:
            print(f"Debug: Checking results_df with {len(self.results_df)} rows")
            print(f"Debug: results_df columns: {list(self.results_df.columns)}")
            matching_rows = self.results_df[self.results_df['name'] == generator_name]
            print(f"Debug: Found {len(matching_rows)} matching rows in results_df for {generator_name}")
            if not matching_rows.empty:
                for col in ['generator_capacity_mw', 'pmax', 'capacity', 'nameplate_capacity', 'Pmax']:
                    if col in matching_rows.columns:
                        value = matching_rows[col].iloc[0]
                        if pd.notna(value) and value > 0:
                            print(f"Debug: Found Pmax {value} for {generator_name} in results_df column {col}")
                            return f"{float(value):.1f}"
        
        # Try to get from max actual generation as a rough estimate
        for df_name, df in [('anomalies_df', getattr(self, 'anomalies_df', None)), 
                           ('results_df', getattr(self, 'results_df', None))]:
            if df is not None and not df.empty:
                if df_name == 'anomalies_df':
                    matching_rows = df[df['generator_name'] == generator_name]
                    actual_col = 'actual_mw'
                else:
                    matching_rows = df[df['name'] == generator_name]
                    # Try multiple column names for actual generation
                    actual_col = None
                    for col_name in ['actual_generation_mw', 'max_actual_generation', 'max_pg', 'actual_pg']:
                        if col_name in df.columns:
                            actual_col = col_name
                            break
                
                if not matching_rows.empty and actual_col and actual_col in matching_rows.columns:
                    max_actual = matching_rows[actual_col].max()
                    if pd.notna(max_actual) and max_actual > 0:
                        # Use 1.2x max actual as rough Pmax estimate
                        estimated_pmax = max_actual * 1.2
                        print(f"Debug: Estimated Pmax {estimated_pmax:.1f} for {generator_name} from max actual {max_actual} in {df_name}")
                        return f"{estimated_pmax:.1f}*"  # * indicates estimate
        
        print(f"Debug: No Pmax found for {generator_name} in any data source")
        return 'N/A'
        
    def _filter_generators_for_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out small generators based on configuration thresholds."""
        if self.config is None or len(df) == 0:
            return df
            
        min_threshold = getattr(self.config, 'MIN_CAPACITY_MW_FOR_REPORTS', 20.0)
        
        # Helper function to safely get column values
        def get_column_values(df, col_names, default_value=float('inf')):
            for col in col_names:
                if col in df.columns:
                    return df[col]
            return pd.Series([default_value] * len(df), index=df.index)
        
        # Create mask for generators to EXCLUDE (all three conditions must be true)
        pmax_values = get_column_values(df, ['pmax', 'generator_capacity_mw'])
        max_actual_values = get_column_values(df, ['max_actual_generation', 'max_pg'])
        max_forecast_values = get_column_values(df, ['max_forecast_generation', 'max_fcst_pg'])
        
        exclude_mask = (
            (pmax_values < min_threshold) &
            (max_actual_values < min_threshold) &
            (max_forecast_values < min_threshold)
        )
        
        filtered_df = df[~exclude_mask].copy()
        
        excluded_count = len(df) - len(filtered_df)
        if excluded_count > 0:
            print(f"📊 Filtered out {excluded_count} small generators from PDF report tables "
                  f"(Pmax, max actual, and max forecast all < {min_threshold} MW)")
        
        return filtered_df
        
    def generate_comprehensive_report(self, 
                                    results_df: pd.DataFrame,
                                    anomalies_df: pd.DataFrame = None,
                                    alerts: List[dict] = None,
                                    bid_validation_results: pd.DataFrame = None,
                                    market: str = "miso",
                                    output_filename: str = None,
                                    resource_db: dict = None) -> str:
        """Generate a comprehensive PDF report with all performance measures."""
        
        # Store resource_db for use in sections that need Pmax data
        self.resource_db = resource_db or {}
        # Store dataframes for alternative Pmax lookup
        self.results_df = results_df
        self.anomalies_df = anomalies_df if anomalies_df is not None else pd.DataFrame()
        
        # Debug information about available data sources
        print(f"Debug: Report generation started with:")
        print(f"  - resource_db: {len(self.resource_db)} resources")
        print(f"  - results_df: {len(self.results_df)} rows, columns: {list(self.results_df.columns) if len(self.results_df) > 0 else 'Empty'}")
        print(f"  - anomalies_df: {len(self.anomalies_df)} rows, columns: {list(self.anomalies_df.columns) if len(self.anomalies_df) > 0 else 'Empty'}")
        
        if output_filename is None:
            output_filename = f"generator_performance_report_{market}_{self.today_date_str}.pdf"
        
        with PdfPages(output_filename) as pdf:
            # Title page
            self._create_title_page(pdf, market)
            
            # Executive summary
            self._create_executive_summary(pdf, results_df, anomalies_df, alerts, bid_validation_results)
            
            # Performance Classification System
            self._create_performance_classification_section(pdf, results_df)
            
            # Chronic Forecast Error Detection
            self._create_chronic_error_section(pdf, anomalies_df, alerts, results_df)
            
            # Advanced Metrics Analysis
            self._create_advanced_metrics_section(pdf, results_df)
            
            # Statistical Anomaly Detection
            self._create_statistical_anomaly_section(pdf, results_df, anomalies_df)
            
            # Bid Validation Results (if available)
            if bid_validation_results is not None and len(bid_validation_results) > 0:
                self._create_bid_validation_section(pdf, bid_validation_results)
            
            # Operational Characteristics
            self._create_operational_characteristics_section(pdf, results_df)
            
            # Recommendations and Action Items
            self._create_recommendations_section(pdf, results_df, anomalies_df, alerts)
            
        print(f"✅ Comprehensive performance report generated: {output_filename}")
        return output_filename
    
    def _create_advanced_metrics_section(self, pdf: PdfPages, results_df: pd.DataFrame):
        """Create advanced metrics analysis section."""
        print(f"🔍 DEBUG: Creating Advanced Metrics section with {len(results_df)} generators")
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Advanced Forecast Metrics Analysis', fontsize=16, fontweight='bold')
        
        if len(results_df) == 0:
            ax = plt.subplot(111)
            ax.text(0.5, 0.5, 'No advanced metrics data available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print("📊 Advanced Metrics section: No data available")
            return
        
        # Apply filtering to remove small generators
        filtered_results = self._filter_generators_for_report(results_df)
        print(f"🔍 DEBUG: After filtering: {len(filtered_results)} generators")
        print(f"🔍 DEBUG: Available columns: {list(filtered_results.columns)}")
        
        # Check if required advanced metrics columns are available
        required_columns = ['consistency_score', 'volatility_score']
        missing_columns = [col for col in required_columns if col not in filtered_results.columns]
        
        if missing_columns:
            print(f"⚠️  DEBUG: Missing columns for Advanced Metrics: {missing_columns}")
            # Show message about missing data
            ax = plt.subplot(111)
            ax.axis('off')
            
            message_text = f"""Advanced Forecast Metrics Analysis
            
Required metrics data is not available in this analysis run.
Missing columns: {', '.join(missing_columns)}

POSSIBLE REASONS:
- Analysis was run in bid validation mode only
- Insufficient time-series data to calculate advanced metrics
- Analysis focused on basic performance metrics only

TO ENABLE ADVANCED METRICS:
- Run full generator analysis with Auto_weekly_generator_analyzer2.py
- Ensure time-series forecast data is available
- Check that consistency_score and volatility_score calculations completed

CURRENT ANALYSIS INCLUDES:
- Basic forecast accuracy metrics (RMSE, MAE, R-squared)
- Performance classification system
- Statistical anomaly detection (Z-scores)"""
            
            ax.text(0.05, 0.95, message_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print("📊 Advanced Metrics section: Missing required columns")
            return
        
        print("✅ DEBUG: All required columns present, creating full Advanced Metrics section")
        
        # Description
        ax_text = plt.subplot2grid((4, 2), (0, 0), colspan=2)
        ax_text.axis('off')
        
        description_text = """Advanced Forecast Metrics

- CONSISTENCY SCORE (0-1): Measures how consistent forecast errors are over time (higher = better)
- VOLATILITY SCORE: Rolling standard deviation of forecast errors (lower = better)  
- TREND ANALYSIS: Statistical trend in forecast performance (improving/stable/deteriorating)
- RMSE % OF CAPACITY: RMSE normalized by generator capacity for fair comparison"""
        
        ax_text.text(0.05, 0.95, description_text, transform=ax_text.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
        
        # Consistency score distribution
        ax_consistency = plt.subplot2grid((4, 2), (1, 0))
        if 'consistency_score' in filtered_results.columns:
            ax_consistency.hist(filtered_results['consistency_score'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax_consistency.set_title('Consistency Score Distribution')
            ax_consistency.set_xlabel('Consistency Score (0-1)')
            ax_consistency.set_ylabel('Number of Generators')
            ax_consistency.axvline(x=0.3, color='red', linestyle='--', label='Poor Threshold')
            ax_consistency.legend()
        else:
            ax_consistency.text(0.5, 0.5, 'Consistency Score\nData Not Available', 
                              ha='center', va='center', fontsize=11, fontweight='bold')
            ax_consistency.set_title('Consistency Score Distribution')
        
        # Volatility score distribution
        ax_volatility = plt.subplot2grid((4, 2), (1, 1))
        if 'volatility_score' in filtered_results.columns:
            ax_volatility.hist(filtered_results['volatility_score'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax_volatility.set_title('Volatility Score Distribution')
            ax_volatility.set_xlabel('Volatility Score (MW)')
            ax_volatility.set_ylabel('Number of Generators')
        else:
            ax_volatility.text(0.5, 0.5, 'Volatility Score\nData Not Available', 
                             ha='center', va='center', fontsize=11, fontweight='bold')
            ax_volatility.set_title('Volatility Score Distribution')
        
        # Bottom performers tables
        metrics_to_analyze = [
            ('consistency_score', 'Lowest Consistency Score', True),
            ('volatility_score', 'Highest Volatility Score', False),
        ]
        
        table_count = 0
        for i, (metric, title, ascending) in enumerate(metrics_to_analyze):
            ax_table = plt.subplot2grid((4, 2), (2 + table_count, 0), colspan=2)
            ax_table.axis('off')
            
            if metric in filtered_results.columns:
                # Title ABOVE the table
                ax_table.text(0.5, 0.95, f'Bottom 10 Percentile: {title}', 
                             ha='center', va='top', transform=ax_table.transAxes,
                             fontsize=12, fontweight='bold')
                
                # Get bottom 10 percentile from filtered data
                bottom_10_pct = int(len(filtered_results) * 0.1)
                if bottom_10_pct == 0:
                    bottom_10_pct = 1
                
                if ascending:
                    worst_performers = filtered_results.nsmallest(bottom_10_pct, metric)
                else:
                    worst_performers = filtered_results.nlargest(bottom_10_pct, metric)
                
                table_data = []
                for idx, row in worst_performers.head(8).iterrows():
                    plant_id = row.get('plant_id', 'N/A')
                    unit_id = row.get('unit_id', 'N/A')
                    pmax = row.get('pmax', row.get('Pmax', 'N/A'))
                    fuel_type = row.get('fuel_type', 'Unknown')
                    fuel_type_str = str(fuel_type) if fuel_type and str(fuel_type) != 'nan' else 'Unknown'
                    table_data.append([
                        str(row['name'])[:20] + '...' if len(str(row['name'])) > 20 else str(row['name']),
                        str(plant_id),
                        str(unit_id),
                        f"{pmax:.1f}" if isinstance(pmax, (int, float)) else str(pmax),
                        f"{row[metric]:.3f}",
                        f"{row['RMSE_over_generation']:.1f}",
                        row['performance_classification'][:4],
                        fuel_type_str[:3]
                    ])
                
                if table_data:
                    table = ax_table.table(cellText=table_data,
                                         colLabels=['Generator', 'Plant ID', 'Unit ID', 'Pmax (MW)', metric.replace('_', ' ').title(), 'RMSE', 'Class', 'Fuel'],
                                         cellLoc='left',
                                         loc='center',
                                         bbox=[0, 0.1, 1, 0.8])
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.5)
                    
                    # Adjust column widths for 8 columns
                    cellDict = table.get_celld()
                    for j in range(len(table_data) + 1):
                        cellDict[(j, 0)].set_width(0.25)  # Generator name
                        cellDict[(j, 1)].set_width(0.10)  # Plant ID
                        cellDict[(j, 2)].set_width(0.10)  # Unit ID
                        cellDict[(j, 3)].set_width(0.12)  # Pmax
                        cellDict[(j, 4)].set_width(0.15)  # Metric
                        cellDict[(j, 5)].set_width(0.12)  # RMSE
                        cellDict[(j, 6)].set_width(0.08)  # Class
                        cellDict[(j, 7)].set_width(0.08)  # Fuel
            else:
                # Missing metric column - show informational message
                ax_table.text(0.5, 0.7, f'{metric.replace("_", " ").title()} Data Not Available', 
                            ha='center', va='center', transform=ax_table.transAxes,
                            fontsize=12, fontweight='bold')
                
                ax_table.text(0.5, 0.3, f'Column "{metric}" missing from analysis results.\nThis metric requires time-series forecast data.\nRun full generator analysis to populate this section.',
                            ha='center', va='center', transform=ax_table.transAxes,
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
            
            table_count += 1
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print("📊 Advanced Metrics section: Successfully completed")
    
    def _create_statistical_anomaly_section(self, pdf: PdfPages, results_df: pd.DataFrame, anomalies_df: pd.DataFrame):
        """Create statistical anomaly detection section."""
        print(f"🔍 DEBUG: Creating Statistical Anomaly section with {len(results_df)} generators and {len(anomalies_df) if anomalies_df is not None else 0} anomalies")
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Statistical Anomaly Detection', fontsize=16, fontweight='bold')
        
        # Description
        ax_text = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax_text.axis('off')
        
        # Check if this is bid validation mode vs. performance analysis mode
        if (anomalies_df is None or len(anomalies_df) == 0) and 'total_bid_issues' in results_df.columns:
            description_text = """Statistical Anomaly Detection - Bid Validation Mode

This section typically shows time-series forecast anomalies, but this report was generated 
from bid validation analysis only. 

To see statistical forecast anomalies, run the full performance analysis:
- Execute Auto_weekly_generator_analyzer2.py for complete analysis
- This will generate time-series forecast data needed for anomaly detection

BID VALIDATION CONTEXT:
- Current analysis focuses on bid configuration issues
- Statistical anomalies require historical forecast vs. actual data
- See "Bid Validation Analysis" section for configuration-related issues"""
        else:
            description_text = """Statistical Anomaly Detection

Uses population statistics to identify generators with anomalous performance:

- RMSE Z-SCORE: How many standard deviations above/below population mean (threshold: >2.0)
- MAE Z-SCORE: Mean Absolute Error compared to population (threshold: >2.0)
- POPULATION OUTLIERS: Generators performing significantly worse than peers

Generators with Z-scores > 2.0 are flagged for investigation.
Z-scores > 3.0 are considered critical and require immediate attention."""
        
        ax_text.text(0.05, 0.95, description_text, transform=ax_text.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
        
        # Check if this is bid validation mode
        if (anomalies_df is None or len(anomalies_df) == 0) and 'total_bid_issues' in results_df.columns:
            # Bid validation mode - show informational message
            ax_info = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)
            ax_info.axis('off')
            
            ax_info.text(0.5, 0.7, 'BID VALIDATION MODE', 
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        transform=ax_info.transAxes)
            
            info_text = """Statistical anomaly detection requires time-series forecast data.
This report focuses on bid configuration validation.

To enable anomaly detection:
- Run Auto_weekly_generator_analyzer2.py for full analysis
- This generates historical forecast vs. actual data needed
- Anomaly detection will then show generators with unusual patterns"""
            
            ax_info.text(0.5, 0.5, info_text,
                        ha='center', va='center', fontsize=12,
                        transform=ax_info.transAxes,
                        bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.7))
            
        elif len(results_df) > 0:
            # Normal performance analysis mode - Apply filtering first
            filtered_results = self._filter_generators_for_report(results_df)
            
            # Calculate Z-scores
            results_df_copy = filtered_results.copy()
            rmse_mean = results_df_copy['RMSE_over_generation'].mean()
            rmse_std = results_df_copy['RMSE_over_generation'].std()
            mae_mean = results_df_copy['MAE_over_generation'].mean()
            mae_std = results_df_copy['MAE_over_generation'].std()
            
            results_df_copy['rmse_zscore'] = (results_df_copy['RMSE_over_generation'] - rmse_mean) / (rmse_std + 1e-8)
            results_df_copy['mae_zscore'] = (results_df_copy['MAE_over_generation'] - mae_mean) / (mae_std + 1e-8)
            
            # RMSE Z-score histogram
            ax_zscore = plt.subplot2grid((3, 2), (1, 0), colspan=2)
            ax_zscore.hist(results_df_copy['rmse_zscore'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax_zscore.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold (+2)')
            ax_zscore.axvline(-2.0, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold (-2)')
            ax_zscore.axvline(3.0, color='darkred', linestyle='-', alpha=0.7, label='Critical Threshold (+3)')
            ax_zscore.axvline(-3.0, color='darkred', linestyle='-', alpha=0.7, label='Critical Threshold (-3)')
            ax_zscore.set_xlabel('RMSE Z-Score')
            ax_zscore.set_ylabel('Number of Generators')
            ax_zscore.set_title('RMSE Z-Score Distribution')
            ax_zscore.legend()
            ax_zscore.grid(True, alpha=0.3)
            
            # Table of statistical anomalies
            ax_table = plt.subplot2grid((3, 2), (2, 0), colspan=2)
            ax_table.axis('off')
            
            # Get generators with high Z-scores
            anomalous_generators = results_df_copy[
                (results_df_copy['rmse_zscore'] > 2.0) | (results_df_copy['mae_zscore'] > 2.0)
            ].sort_values('rmse_zscore', ascending=False)
            
            table_data = []
            for idx, row in anomalous_generators.head(15).iterrows():
                severity = 'Critical' if row['rmse_zscore'] > 3.0 or row['mae_zscore'] > 3.0 else 'High'
                plant_id = row.get('plant_id', 'N/A')
                unit_id = row.get('unit_id', 'N/A')
                pmax = row.get('generator_capacity_mw', row.get('pmax', row.get('Pmax', 'N/A')))
                performance_class = row['performance_classification']
                performance_class_str = str(performance_class) if performance_class and str(performance_class) != 'nan' else 'Unknown'
                table_data.append([
                    str(row['name'])[:22] + '...' if len(str(row['name'])) > 22 else str(row['name']),
                    str(plant_id),
                    str(unit_id),
                    f"{pmax:.1f}" if isinstance(pmax, (int, float)) else str(pmax),
                    f"{row['rmse_zscore']:.2f}",
                    severity,
                    performance_class_str[:4]
                ])
            
            if table_data:
                table = ax_table.table(cellText=table_data,
                                     colLabels=['Generator Name', 'Plant ID', 'Unit ID', 'Pmax (MW)', 'RMSE Z-Score', 'Severity', 'Class'],
                                     cellLoc='left',
                                     loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.5)
                
                # Adjust column widths for 7 columns
                cellDict = table.get_celld()
                for i in range(len(table_data) + 1):
                    cellDict[(i, 0)].set_width(0.25)  # Generator name
                    cellDict[(i, 1)].set_width(0.10)  # Plant ID
                    cellDict[(i, 2)].set_width(0.10)  # Unit ID
                    cellDict[(i, 3)].set_width(0.12)  # Pmax
                    cellDict[(i, 4)].set_width(0.13)  # RMSE Z-Score
                    cellDict[(i, 5)].set_width(0.15)  # Severity
                    cellDict[(i, 6)].set_width(0.15)  # Class
                    
                ax_table.set_title('Statistical Anomalies (Z-Score > 2.0)', fontweight='bold', y=0.98)
            else:
                ax_table.text(0.5, 0.5, 'No Statistical Anomalies Detected', 
                            ha='center', va='center', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print("📊 Statistical Anomaly section: Successfully completed")

    def _create_title_page(self, pdf: PdfPages, market: str):
        """Create the title page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'Generator Performance Analysis Report', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Subtitle
        ax.text(0.5, 0.75, f'{market.upper()} Market', 
                ha='center', va='center', fontsize=18)
        
        # Date
        ax.text(0.5, 0.7, f'Analysis Date: {self.today_date_str}', 
                ha='center', va='center', fontsize=14)
        
        # Description
        min_threshold = getattr(self.config, 'MIN_CAPACITY_MW_FOR_REPORTS', 20.0)
        description = f"""
        This report provides a comprehensive analysis of generator forecast performance,
        including performance classifications, anomaly detection, chronic error patterns,
        and bid validation results. The analysis identifies generators requiring
        attention and provides actionable recommendations for improvement.
        
        FILTERING APPLIED: Small generators are excluded from all tables if they meet
        ALL three criteria: Pmax < {min_threshold} MW, max actual generation < {min_threshold} MW,
        and max predicted generation < {min_threshold} MW.
        """
        ax.text(0.5, 0.5, description, ha='center', va='center', fontsize=12,
                wrap=True, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        # Footer
        ax.text(0.5, 0.1, 'Generated by Advanced Generator Analysis System', 
                ha='center', va='center', fontsize=10, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_executive_summary(self, pdf: PdfPages, results_df: pd.DataFrame,
                                anomalies_df: pd.DataFrame, alerts: List[dict],
                                bid_validation_results: pd.DataFrame):
        """Create executive summary page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Executive Summary', fontsize=16, fontweight='bold')
        
        # Apply filtering to remove small generators for summary statistics
        filtered_results = self._filter_generators_for_report(results_df) if len(results_df) > 0 else results_df
        
        # Handle empty data case
        if len(filtered_results) == 0:
            # Create empty plots with messages
            ax1.text(0.5, 0.5, 'No analysis data available\n(Debug mode with no valid results)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Performance Classification Distribution')
            
            ax2.text(0.5, 0.5, 'No RMSE data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('RMSE Distribution')
            
            ax3.text(0.5, 0.5, 'No R² data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('R² Distribution')
            
            ax4.text(0.5, 0.5, 'No summary statistics available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Summary Statistics')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Performance distribution pie chart
        if len(filtered_results) > 0:
            perf_counts = filtered_results['performance_classification'].value_counts()
            colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
            ax1.pie(perf_counts.values, labels=perf_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(perf_counts)])
            ax1.set_title('Performance Classification Distribution')
        
        # RMSE distribution histogram
        if len(filtered_results) > 0 and 'RMSE_over_generation' in filtered_results.columns:
            ax2.hist(filtered_results['RMSE_over_generation'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('RMSE Distribution')
            ax2.set_xlabel('RMSE (MW)')
            ax2.set_ylabel('Number of Generators')
        
        # R² distribution (replacing Alert Severity Distribution)
        if len(filtered_results) > 0 and 'R_SQUARED' in filtered_results.columns:
            ax3.hist(filtered_results['R_SQUARED'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('R² Distribution')
            ax3.set_xlabel('R² Value')
            ax3.set_ylabel('Number of Generators')
        
        # Key statistics table
        ax4.axis('off')
        if len(results_df) > 0:
            filtered_count = len(filtered_results)
            excluded_count = len(results_df) - filtered_count
            
            stats_data = [
                ['Total Generators (Raw)', len(results_df)],
                ['Generators Analyzed', filtered_count],
                ['Small Generators Excluded', excluded_count],
                ['Anomalies Detected', len(anomalies_df) if anomalies_df is not None else 0],
                ['Total Alerts', len(alerts) if alerts else 0],
                ['Poor/Critical Performers', len(filtered_results[filtered_results['performance_classification'].isin(['poor', 'critical'])]) if len(filtered_results) > 0 else 0],
                ['Average RMSE (MW)', f"{filtered_results['RMSE_over_generation'].mean():.2f}" if len(filtered_results) > 0 else "N/A"],
                ['Average R²', f"{filtered_results['R_SQUARED'].mean():.3f}" if len(filtered_results) > 0 else "N/A"]
            ]
            
            if bid_validation_results is not None and len(bid_validation_results) > 0:
                stats_data.append(['Bid Validation Issues', len(bid_validation_results)])
            
            table = ax4.table(cellText=stats_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax4.set_title('Key Statistics', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_performance_classification_section(self, pdf: PdfPages, results_df: pd.DataFrame):
        """Create detailed performance classification section."""
        # First page: Title, description and distribution chart
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Performance Classification System', fontsize=16, fontweight='bold')
        
        # Create text description
        ax_text = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax_text.axis('off')
        
        # Get thresholds from config if available
        thresholds = {}
        if self.config and hasattr(self.config, 'ANOMALY_DETECTION') and 'performance_thresholds' in self.config.ANOMALY_DETECTION:
            thresholds = self.config.ANOMALY_DETECTION['performance_thresholds']
        else:
            # Default thresholds
            thresholds = {
                "excellent": {"rmse_pct_max": 2.0, "r2_min": 0.95},
                "good": {"rmse_pct_max": 5.0, "r2_min": 0.85},
                "fair": {"rmse_pct_max": 10.0, "r2_min": 0.70},
                "poor": {"rmse_pct_max": 20.0, "r2_min": 0.50},
                "critical": {"rmse_pct_max": float('inf'), "r2_min": 0.0}
            }
        
        description = f"""
        PERFORMANCE CLASSIFICATION SYSTEM
        
        The system classifies each generator into one of 5 performance categories based on:
        - RMSE as percentage of generator capacity (Pmax)
        - R-squared correlation coefficient
        
        Classification Criteria:
        - EXCELLENT: RMSE ≤ {thresholds['excellent']['rmse_pct_max']}% of Pmax, R² ≥ {thresholds['excellent']['r2_min']} (Highly accurate forecasts)
        - GOOD: RMSE ≤ {thresholds['good']['rmse_pct_max']}% of Pmax, R² ≥ {thresholds['good']['r2_min']} (Good forecast accuracy)
        - FAIR: RMSE ≤ {thresholds['fair']['rmse_pct_max']}% of Pmax, R² ≥ {thresholds['fair']['r2_min']} (Acceptable performance)
        - POOR: RMSE ≤ {thresholds['poor']['rmse_pct_max']}% of Pmax, R² ≥ {thresholds['poor']['r2_min']} (Needs attention)
        - CRITICAL: RMSE > {thresholds['poor']['rmse_pct_max']}% of Pmax or R² < {thresholds['critical']['r2_min']} (Immediate action required)
        
        PERFORMANCE SCORE EXPLANATION:
        The "Score" column represents a composite performance score (0-100) calculated as:
        - 70% weight: Inverted RMSE percentage (lower RMSE = higher score)
        - 20% weight: R-squared × 100 (higher correlation = higher score)
        - 5% weight: Consistency score × 100 (more consistent = higher score)
        - 5% weight: Inverted volatility score (lower volatility = higher score)
        
        Higher scores (closer to 100) indicate better overall forecast performance.
        Lower scores (closer to 0) indicate generators requiring immediate attention.
        """
        
        ax_text.text(0.05, 0.95, description, transform=ax_text.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
        
        if len(results_df) > 0:
            # Performance scores histogram
            ax_dist = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            
            # Get performance scores, fallback to R² * 100 if not available
            if 'performance_score' in results_df.columns:
                scores = results_df['performance_score'].dropna()
            else:
                scores = results_df['R_SQUARED'] * 100
            
            # Create histogram of performance scores
            if len(scores) > 0:
                ax_dist.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                ax_dist.set_title('Performance Score Distribution')
                ax_dist.set_xlabel('Performance Score')
                ax_dist.set_ylabel('Number of Generators')
                ax_dist.grid(True, alpha=0.3)
                
                # Add vertical lines for performance thresholds if available
                if hasattr(self, 'config') and hasattr(self.config, 'ANOMALY_DETECTION'):
                    # Add reference lines for different performance levels
                    ax_dist.axvline(x=70, color='green', linestyle='--', alpha=0.6, label='Excellent (70+)')
                    ax_dist.axvline(x=50, color='orange', linestyle='--', alpha=0.6, label='Fair (50+)')
                    ax_dist.axvline(x=20, color='red', linestyle='--', alpha=0.6, label='Poor (20+)')
                    ax_dist.legend(fontsize=8)
            else:
                ax_dist.text(0.5, 0.5, 'No performance score data available', 
                           ha='center', va='center', transform=ax_dist.transAxes, fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Second page: Generator table with only POOR and CRITICAL classifications
        if len(results_df) > 0:
            # Apply filtering to remove small generators
            filtered_results = self._filter_generators_for_report(results_df)
            
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Generators Requiring Attention (Poor and Critical Only)', fontsize=16, fontweight='bold')
            
            # Filter to only poor and critical generators, sorted by performance score (ascending = worst first)
            poor_critical_generators = filtered_results[
                filtered_results['performance_classification'].isin(['poor', 'critical'])
            ]
            
            # Sort by performance score (lowest first), fallback to R² if performance_score not available
            if 'performance_score' in poor_critical_generators.columns:
                poor_critical_generators = poor_critical_generators.sort_values('performance_score', ascending=True)
            else:
                poor_critical_generators = poor_critical_generators.sort_values('R_SQUARED', ascending=True)
            
            # Create table data with plant_id and unit_id
            table_data = []
            for idx, row in poor_critical_generators.iterrows():
                plant_id = row.get('plant_id', 'N/A')
                unit_id = row.get('unit_id', 'N/A')
                # Get capacity from the correct column name
                pmax = row.get('generator_capacity_mw', row.get('pmax', row.get('Pmax', 'N/A')))
                table_data.append([
                    str(row['name'])[:25] + '...' if len(str(row['name'])) > 25 else str(row['name']),  # Generator name
                    str(plant_id),     # Plant ID
                    str(unit_id),      # Unit ID
                    f"{pmax:.1f}" if isinstance(pmax, (int, float)) else str(pmax),  # Pmax
                    row['performance_classification'],
                    f"{row.get('performance_score', row['R_SQUARED']*100):.1f}",
                    f"{row['RMSE_over_generation']:.1f}"
                ])
            
            # Create table covering most of the page
            ax_table = plt.subplot(111)
            ax_table.axis('off')
            
            if table_data:
                table = ax_table.table(cellText=table_data,
                                     colLabels=['Generator Name', 'Plant ID', 'Unit ID', 'Pmax (MW)', 'Classification', 'Score', 'RMSE'],
                                     cellLoc='left',
                                     loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.8)  # Increase row height for better readability
                
                # Adjust column widths
                cellDict = table.get_celld()
                for i in range(len(table_data) + 1):  # +1 for header
                    cellDict[(i, 0)].set_width(0.30)  # Generator name (increased from 0.25)
                    cellDict[(i, 1)].set_width(0.10)  # Plant ID (increased from 0.08)
                    cellDict[(i, 2)].set_width(0.10)  # Unit ID (increased from 0.08)
                    cellDict[(i, 3)].set_width(0.15)  # Pmax (increased from 0.12)
                    cellDict[(i, 4)].set_width(0.20)  # Classification (increased from 0.17)
                    cellDict[(i, 5)].set_width(0.08)  # Score (decreased from 0.10)
                    cellDict[(i, 6)].set_width(0.07)  # RMSE (decreased from 0.10)
                
                # Add subtitle with count (show both filtered and original counts)
                plt.figtext(0.5, 0.95, f'Generators requiring attention: {len(poor_critical_generators)} of {len(filtered_results)} analyzed '
                           f'({len(results_df)} total before filtering)', 
                           ha='center', fontsize=12, style='italic')
            else:
                # No poor/critical generators found
                ax_table.text(0.5, 0.5, '✅ No Generators Requiring Attention\nAll generators have FAIR or better performance', 
                            ha='center', va='center', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _create_chronic_error_section(self, pdf: PdfPages, anomalies_df: pd.DataFrame, alerts: List[dict], results_df: pd.DataFrame):
        """Create chronic forecast error detection section."""
        # Create first page with description
        fig1 = plt.figure(figsize=(11, 8.5))
        fig1.suptitle('Chronic Forecast Error Detection', fontsize=16, fontweight='bold')
        
        # Description takes the full page
        ax_text = plt.subplot(1, 1, 1)
        ax_text.axis('off')
        
        description = """
        CHRONIC FORECAST ERROR DETECTION
        
        Identifies generators with persistent forecasting problems over extended periods:
        
        - CHRONIC OVER-FORECASTING: Forecast consistently > 2x actual generation for 3+ days in any 5-day window
        - CHRONIC UNDER-FORECASTING: Forecast consistently < 0.5x actual generation for 3+ days in any 5-day window
        
        Detection Criteria:
        - Minimum 3 problematic days in any 5-day sliding window
        - Minimum 2 hours of data per day to qualify (adjusted for 3x daily sampling)
        - Only considers periods with generation ≥ 5 MW to avoid noise
        - High severity if 6+ problematic days occur in any 8-day window
        
        Impact: Chronic errors indicate systematic model issues requiring immediate attention.
        This approach detects sustained chronic patterns while reducing sensitivity to short-term market volatility.
        Regular 5-day window monitoring provides balanced detection of forecast degradation.
        
        
        METHODOLOGY:
        
        The sliding window approach analyzes forecast accuracy over time:
        
        1. Daily Statistics: Calculate daily average forecast-to-actual ratios for each generator
        2. Sliding Windows: Apply 5-day and 8-day sliding windows across the analysis period
        3. Pattern Detection: Identify periods where forecast ratios exceed thresholds:
           - Over-forecasting: Forecast/Actual ≥ 2.0 (forecast is at least 200% of actual)
           - Under-forecasting: Forecast/Actual ≤ 0.5 (forecast is 50% or less of actual)
        4. Severity Classification:
           - Medium Severity: 3+ problematic days in any 5-day window
           - High Severity: 6+ problematic days in any 8-day window
        
        This methodology ensures robust detection of persistent forecasting issues while minimizing
        false positives from temporary market disruptions or operational anomalies.
        """
        
        ax_text.text(0.05, 0.95, description, transform=ax_text.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Create second page with table
        fig2 = plt.figure(figsize=(11, 8.5))
        fig2.suptitle('Chronic Error Generators - Analysis Results', fontsize=16, fontweight='bold')
        
        # Chronic error analysis
        chronic_alerts = [alert for alert in alerts if 'CHRONIC' in alert.get('alert_type', '')] if alerts else []
        
        # For bid validation context, also show persistent high-severity issues
        bid_validation_alerts = [alert for alert in alerts if 'BID_VALIDATION' in alert.get('alert_type', '')] if alerts else []
        
        if chronic_alerts or bid_validation_alerts:
            # Show chronic errors if they exist, otherwise show bid validation issues
            alerts_to_show = chronic_alerts if chronic_alerts else bid_validation_alerts[:10]
            
            # Create table that takes most of the page
            ax_table = plt.subplot(1, 1, 1)
            ax_table.axis('off')
            
            # Process chronic alerts to remove duplicates and ensure all generators are included
            if chronic_alerts:
                # Group alerts by generator to remove duplicates and aggregate information
                generator_alerts = {}
                for alert in chronic_alerts:
                    generator_name = alert.get('generator', 'Unknown')
                    if generator_name not in generator_alerts:
                        generator_alerts[generator_name] = {
                            'alert': alert,
                            'error_types': [],
                            'severities': [],
                            'patterns': [],
                            'avg_actual_values': []
                        }
                    
                    # Collect error types and other info for this generator
                    error_type = alert['alert_type'].replace('CHRONIC_', '').replace('FORECASTING', 'FORECAST')
                    generator_alerts[generator_name]['error_types'].append(error_type)
                    generator_alerts[generator_name]['severities'].append(alert.get('severity', 'Unknown'))
                    
                    details = alert.get('details', {})
                    if 'problematic_days' in details and 'window_days' in details:
                        # Show the actual pattern: problematic_days/window_days
                        pattern = f"{details['problematic_days']}/{details['window_days']}"
                    elif 'duration_days' in details:
                        pattern = f"{details['duration_days']} days"
                    else:
                        pattern = 'N/A'
                    generator_alerts[generator_name]['patterns'].append(pattern)
                    generator_alerts[generator_name]['avg_actual_values'].append(details.get('avg_actual_mw', 0))

                # Sort generators by highest severity and then by max avg_actual_mw
                def get_severity_score(severities):
                    severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
                    return max(severity_order.get(sev.lower(), 0) for sev in severities)
                
                generators_sorted = sorted(generator_alerts.items(), 
                                         key=lambda x: (get_severity_score(x[1]['severities']), 
                                                       max(x[1]['avg_actual_values']) if x[1]['avg_actual_values'] else 0), 
                                         reverse=True)
                
                table_data = []
                for generator_name, info in generators_sorted:
                    alert = info['alert']
                    plant_id = alert.get('plant_id', alert.get('details', {}).get('plant_id', 'N/A'))
                    unit_id = alert.get('unit_id', alert.get('details', {}).get('unit_id', 'N/A'))
                    
                    # Combine error types (show unique types)
                    unique_error_types = list(set(info['error_types']))
                    error_type_display = ', '.join([et[:6] for et in unique_error_types[:2]])  # Show up to 2 types, abbreviated
                    if len(unique_error_types) > 2:
                        error_type_display += '+...'
                    
                    # Use the most recent/relevant pattern
                    pattern = info['patterns'][0] if info['patterns'] else 'N/A'
                    
                    # Get highest severity
                    highest_severity = max(info['severities'], key=lambda s: get_severity_score([s]))
                    
                    # Get Pmax using the same method as successful tables - look up in results DataFrame
                    pmax_display = 'N/A'
                    pmax_final = 'N/A'
                    
                    # First try to find the generator in results_df (same approach as working tables)
                    if results_df is not None and not results_df.empty:
                        # Try matching by orig_name first (full generator name with unit)
                        matching_rows = results_df[results_df['orig_name'] == generator_name]
                        if matching_rows.empty:
                            # Fallback to matching by name (base generator name)
                            matching_rows = results_df[results_df['name'] == generator_name]
                        
                        if not matching_rows.empty:
                            row = matching_rows.iloc[0]
                            # Use the EXACT same method as successful tables
                            pmax = row.get('generator_capacity_mw', row.get('pmax', row.get('Pmax', 'N/A')))
                            if isinstance(pmax, (int, float)) and pmax > 0:
                                pmax_display = f"{pmax:.1f}"
                                pmax_final = f"{pmax_display} MW"
                                print(f"Debug: Found Pmax {pmax} for {generator_name} in results_df using successful table method")
                            else:
                                print(f"Debug: Pmax from results_df for {generator_name} was: {pmax} (type: {type(pmax)})")
                        else:
                            print(f"Debug: Generator {generator_name} not found in results_df (tried orig_name and name columns)")
                    else:
                        print(f"Debug: results_df is None or empty: {results_df is None}, {len(results_df) if results_df is not None else 'N/A'}")
                    
                    # If results_df lookup failed, try alert details as fallback
                    if pmax_display == 'N/A':
                        details = alert.get('details', {})
                        for pmax_key in ['pmax', 'generator_capacity_mw', 'capacity', 'nameplate_capacity']:
                            if pmax_key in details and details[pmax_key] and details[pmax_key] != 'N/A':
                                try:
                                    pmax_value = float(details[pmax_key])
                                    if pmax_value > 0:
                                        pmax_display = f"{pmax_value:.1f}"
                                        pmax_final = f"{pmax_display} MW"
                                        print(f"Debug: Found Pmax {pmax_value} for {generator_name} in alert details key {pmax_key}")
                                        break
                                except (ValueError, TypeError):
                                    continue
                    
                    # Final fallback 
                    if pmax_display == 'N/A':
                        pmax_final = 'N/A'
                        print(f"Debug: No Pmax found for {generator_name} in any source")
                    
                    table_data.append([
                        str(generator_name)[:20] + '...' if len(str(generator_name)) > 20 else str(generator_name),
                        str(plant_id),
                        str(unit_id),
                        error_type_display[:12],  # Error types
                        pattern,  # Pattern
                        pmax_final,  # Pmax from resources.json with proper units
                        highest_severity  # Highest severity
                    ])
                
                # Sort bid validation alerts by severity (critical first) for non-chronic case
            else:
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                alerts_sorted = sorted(alerts_to_show, 
                                     key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
                table_data = []
                for alert in alerts_sorted[:10]:  # Show top 10
                    plant_id = alert.get('plant_id', alert.get('details', {}).get('plant_id', 'N/A'))
                    unit_id = alert.get('unit_id', alert.get('details', {}).get('unit_id', 'N/A'))
                    
                    # Bid validation format
                    issue_type = alert['alert_type'].replace('BID_VALIDATION_', '').replace('_', ' ').title()
                    generator = alert.get('generator', 'Unknown')
                    severity = alert.get('severity', 'Unknown')
                    fuel_type = alert.get('details', {}).get('fuel_type', 'Unknown')
                    
                    fuel_type_safe = str(fuel_type) if fuel_type and str(fuel_type) != 'nan' else 'Unknown'
                    table_data.append([
                        str(generator)[:18] + '...' if len(str(generator)) > 18 else str(generator),
                        str(plant_id),
                        str(unit_id),
                        issue_type[:12],  # Issue type
                        severity.upper(),
                        fuel_type_safe[:6]  # Fuel type
                    ])
            
            if table_data:
                if chronic_alerts:
                    col_labels = ['Generator', 'Plant ID', 'Unit ID', 'Error Type', 'Pattern', 'Pmax', 'Severity']
                    title = 'Chronic Error Generators (Unique)'
                else:
                    col_labels = ['Generator', 'Plant ID', 'Unit ID', 'Issue Type', 'Severity', 'Fuel']
                    title = 'Generators with Validation Issues'
                
                table = ax_table.table(cellText=table_data,
                                     colLabels=col_labels,
                                     cellLoc='left',
                                     loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.8)  # Increased row height for better readability
                
                # Adjust column widths 
                cellDict = table.get_celld()
                for i in range(len(table_data) + 1):  # +1 for header
                    if chronic_alerts:
                        cellDict[(i, 0)].set_width(0.25)  # Generator name (increased from 0.20)
                        cellDict[(i, 1)].set_width(0.10)  # Plant ID
                        cellDict[(i, 2)].set_width(0.10)  # Unit ID
                        cellDict[(i, 3)].set_width(0.15)  # Error type (increased from 0.12)
                        cellDict[(i, 4)].set_width(0.15)  # Pattern (increased from 0.12)
                        cellDict[(i, 5)].set_width(0.15)  # Pmax (increased from 0.12)
                        cellDict[(i, 6)].set_width(0.10)  # Severity (reduced from 0.12)
                    else:
                        cellDict[(i, 0)].set_width(0.25)  # Generator name
                        cellDict[(i, 1)].set_width(0.10)  # Plant ID
                        cellDict[(i, 2)].set_width(0.10)  # Unit ID
                        cellDict[(i, 3)].set_width(0.20)  # Issue type
                        cellDict[(i, 4)].set_width(0.15)  # Severity
                        cellDict[(i, 5)].set_width(0.20)  # Fuel
                
                ax_table.set_title(title, fontweight='bold')
        else:
            # No chronic errors found
            ax_none = plt.subplot(1, 1, 1)
            ax_none.axis('off')
            ax_none.text(0.5, 0.5, '✅ No Chronic Forecast Errors Detected', 
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
    
    def _create_bid_validation_section(self, pdf: PdfPages, bid_validation_results: pd.DataFrame):
        """Create bid validation section."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Bid Validation Analysis', fontsize=16, fontweight='bold')
        
        # Description
        ax_text = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax_text.axis('off')
        
        description = """
        BID VALIDATION ANALYSIS
        
        Identifies generators with configuration issues in their bid parameters:
        
        - PMIN VIOLATIONS: Generator operates below declared Pmin too frequently
        - PMAX VIOLATIONS: Generator operates above declared Pmax
        - CAPACITY FACTOR ISSUES: Inconsistent generation patterns vs capacity
        - MARKET PARTICIPATION: Unusual bid behavior or missing data
        
        These issues suggest potential bid configuration problems requiring operational review.
        """
        
        ax_text.text(0.05, 0.95, description, transform=ax_text.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
        
        if bid_validation_results is not None and len(bid_validation_results) > 0:
            # Apply filtering to remove small generators
            filtered_results = self._filter_generators_for_report(bid_validation_results)
            
            # Issue type distribution
            ax_issues = plt.subplot2grid((3, 2), (1, 0))
            if 'issue_type' in filtered_results.columns:
                issue_counts = filtered_results['issue_type'].value_counts()
                colors = ['red', 'orange', 'yellow', 'lightblue']
                ax_issues.pie(issue_counts.values, labels=issue_counts.index, autopct='%1.1f%%', 
                           colors=colors[:len(issue_counts)])
                ax_issues.set_title('Bid Validation Issue Types')
            
            # Severity distribution
            ax_severity = plt.subplot2grid((3, 2), (1, 1))
            if 'severity' in filtered_results.columns:
                severity_counts = filtered_results['severity'].value_counts()
                severity_colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'lightblue'}
                bar_colors = [severity_colors.get(sev, 'gray') for sev in severity_counts.index]
                ax_severity.bar(range(len(severity_counts)), severity_counts.values, color=bar_colors)
                ax_severity.set_xticks(range(len(severity_counts)))
                ax_severity.set_xticklabels(severity_counts.index, rotation=45)
                ax_severity.set_title('Issue Severity Distribution')
                ax_severity.set_ylabel('Number of Issues')
            
            # Table of top issues
            ax_table = plt.subplot2grid((3, 2), (2, 0), colspan=2)
            ax_table.axis('off')
            
            # Sort by severity and issue type
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_results = filtered_results.sort_values([
                filtered_results['severity'].map(severity_order),
                'issue_type'
            ])
            
            table_data = []
            for idx, row in sorted_results.head(10).iterrows():
                plant_id = row.get('plant_id', 'N/A')
                unit_id = row.get('unit_id', 'N/A')
                fuel_type = row.get('fuel_type', 'Unknown')
                fuel_type_safe = str(fuel_type) if fuel_type and str(fuel_type) != 'nan' else 'Unknown'
                table_data.append([
                    str(row.get('generator_name', row.get('name', 'Unknown')))[:20] + '...' if len(str(row.get('generator_name', row.get('name', 'Unknown')))) > 20 else str(row.get('generator_name', row.get('name', 'Unknown'))),
                    str(plant_id),
                    str(unit_id),
                    row.get('issue_type', 'Unknown')[:15],
                    row.get('severity', 'Unknown'),
                    f"{row.get('pmax', 0):.1f}" if 'pmax' in row else 'N/A',
                    fuel_type_safe[:6]
                ])
            
            if table_data:
                table = ax_table.table(cellText=table_data,
                                     colLabels=['Generator', 'Plant ID', 'Unit ID', 'Issue Type', 'Severity', 'Pmax', 'Fuel'],
                                     cellLoc='left',
                                     loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.5)
                
                # Adjust column widths
                cellDict = table.get_celld()
                for i in range(len(table_data) + 1):
                    cellDict[(i, 0)].set_width(0.25)  # Generator name
                    cellDict[(i, 1)].set_width(0.10)  # Plant ID
                    cellDict[(i, 2)].set_width(0.10)  # Unit ID
                    cellDict[(i, 3)].set_width(0.20)  # Issue type
                    cellDict[(i, 4)].set_width(0.15)  # Severity
                    cellDict[(i, 5)].set_width(0.10)  # Pmax
                    cellDict[(i, 6)].set_width(0.10)  # Fuel
                
                ax_table.set_title('Top Bid Validation Issues', fontweight='bold')
        else:
            # No bid validation data
            ax_none = plt.subplot2grid((3, 2), (1, 0), colspan=2)
            ax_none.axis('off')
            ax_none.text(0.5, 0.5, 'No bid validation data available\n(Bid validation disabled or no issues found)', 
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_operational_characteristics_section(self, pdf: PdfPages, results_df: pd.DataFrame):
        """Create operational characteristics section."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Operational Characteristics Analysis', fontsize=16, fontweight='bold')
        
        # Description
        ax_text = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax_text.axis('off')
        
        description = """
        OPERATIONAL CHARACTERISTICS
        
        Analysis of generator operational patterns and their impact on forecast accuracy:
        
        - CAPACITY UTILIZATION: Percentage of time generator is running
        - MUST-RUN STATUS: Whether generator runs consistently (baseload characteristics)
        - GENERATION PATTERNS: Zero vs non-zero generation frequencies
        - CAPACITY FACTORS: Relationship between Pmax and actual generation patterns
        
        These characteristics help identify if poor forecasts are due to operational complexity.
        """
        
        ax_text.text(0.05, 0.95, description, transform=ax_text.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
        
        if len(results_df) > 0:
            # Apply filtering to remove small generators
            filtered_results = self._filter_generators_for_report(results_df)
            
            # Performance by fuel type (moved to take up more space since capacity utilization chart removed)
            if 'fuel_type' in filtered_results.columns:
                ax_fuel = plt.subplot2grid((3, 2), (1, 0), colspan=2)
                fuel_rmse = filtered_results.groupby('fuel_type')['RMSE_over_generation'].mean().sort_values(ascending=False)
                colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen', 'purple', 'pink']
                ax_fuel.bar(range(len(fuel_rmse)), fuel_rmse.values, color=colors[:len(fuel_rmse)])
                ax_fuel.set_xticks(range(len(fuel_rmse)))
                ax_fuel.set_xticklabels(fuel_rmse.index, rotation=45)
                ax_fuel.set_title('Average RMSE by Fuel Type')
                ax_fuel.set_ylabel('Average RMSE (MW)')
                ax_fuel.grid(True, alpha=0.3)
            
            # Low utilization generators with poor performance (with plant_id and unit_id)
            ax_table = plt.subplot2grid((3, 2), (2, 0), colspan=2)
            ax_table.axis('off')
            
            # Find generators with low utilization and poor performance
            if '%_running' in filtered_results.columns:
                low_util_poor_perf = filtered_results[
                    (filtered_results['%_running'] < 20) & 
                    (filtered_results['performance_classification'].isin(['poor', 'critical']))
                ].sort_values('RMSE_over_generation', ascending=False)
                
                table_data = []
                for idx, row in low_util_poor_perf.head(10).iterrows():  # Show more entries
                    plant_id = row.get('plant_id', 'N/A')
                    unit_id = row.get('unit_id', 'N/A')
                    fuel_type = row.get('fuel_type', 'Unknown')
                    fuel_type_safe = str(fuel_type) if fuel_type and str(fuel_type) != 'nan' else 'Unknown'
                    table_data.append([
                        row['name'][:18] + '...' if len(str(row['name'])) > 18 else str(row['name']),
                        str(plant_id),
                        str(unit_id),
                        f"{row['%_running']:.1f}%",
                        f"{row['RMSE_over_generation']:.1f}",
                        row['performance_classification'],
                        fuel_type_safe[:6],
                        'Intermittent' if row['%_running'] < 10 else 'Low Util'
                    ])
                
                if table_data:
                    table = ax_table.table(cellText=table_data,
                                         colLabels=['Generator', 'Plant ID', 'Unit ID', '% Running', 'RMSE', 'Class', 'Fuel', 'Pattern'],
                                         cellLoc='left',
                                         loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1.2, 1.5)
                    
                    # Adjust column widths for plant_id and unit_id
                    cellDict = table.get_celld()
                    for i in range(len(table_data) + 1):  # +1 for header
                        cellDict[(i, 0)].set_width(0.22)  # Generator name
                        cellDict[(i, 1)].set_width(0.10)  # Plant ID
                        cellDict[(i, 2)].set_width(0.10)  # Unit ID
                        cellDict[(i, 3)].set_width(0.12)  # % Running
                        cellDict[(i, 4)].set_width(0.10)  # RMSE
                        cellDict[(i, 5)].set_width(0.12)  # Class
                        cellDict[(i, 6)].set_width(0.12)  # Fuel
                        cellDict[(i, 7)].set_width(0.12)  # Pattern
                    
                    ax_table.set_title('Low Utilization Generators with Poor Performance', fontweight='bold')
                else:
                    ax_table.text(0.5, 0.5, '✅ No Low-Utilization Performance Issues', 
                                ha='center', va='center', fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_recommendations_section(self, pdf: PdfPages, results_df: pd.DataFrame, 
                                      anomalies_df: pd.DataFrame, alerts: List[dict]):
        """Create recommendations and action items section."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Recommendations and Action Items', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        recommendations = []
        
        # Performance-based recommendations
        if len(results_df) > 0:
            critical_count = len(results_df[results_df['performance_classification'] == 'critical'])
            poor_count = len(results_df[results_df['performance_classification'] == 'poor'])
            
            if critical_count > 0:
                recommendations.append(f"🚨 CRITICAL: {critical_count} generators with critical performance require immediate model review")
            
            if poor_count > 0:
                recommendations.append(f"⚠️  HIGH: {poor_count} generators with poor performance need attention within 1-2 weeks")
        
        # Chronic error recommendations
        chronic_alerts = [alert for alert in alerts if 'CHRONIC' in alert.get('alert_type', '')] if alerts else []
        if chronic_alerts:
            # Count unique generators with chronic errors (not total alerts)
            unique_chronic_generators = set(alert.get('generator', '') for alert in chronic_alerts if alert.get('generator'))
            chronic_generator_count = len(unique_chronic_generators)
            recommendations.append(f"🔄 CHRONIC ERRORS: {chronic_generator_count} generators with chronic forecasting patterns")
            recommendations.append("   → Review dispatch model parameters and operational constraints")
            recommendations.append("   → Analyze market conditions during chronic error periods")
        
        # Statistical anomaly recommendations
        if len(results_df) > 0:
            # Calculate Z-scores for recommendations
            rmse_mean = results_df['RMSE_over_generation'].mean()
            rmse_std = results_df['RMSE_over_generation'].std()
            high_zscore_count = len(results_df[(results_df['RMSE_over_generation'] - rmse_mean) / (rmse_std + 1e-8) > 2.0])
            
            if high_zscore_count > 0:
                recommendations.append(f"📊 STATISTICAL: {high_zscore_count} generators are statistical outliers")
                recommendations.append("   → Compare with similar generators in same zone/fuel type")
                recommendations.append("   → Investigate if these generators have unique operational characteristics")
        
        # Fuel type specific recommendations
        if len(results_df) > 0 and 'fuel_type' in results_df.columns:
            fuel_performance = results_df.groupby('fuel_type').agg({
                'RMSE_over_generation': 'mean',
                'performance_classification': lambda x: (x.isin(['poor', 'critical'])).sum()
            })
            
            # Check if fuel_performance is not empty before calling idxmax()
            if len(fuel_performance) > 0:
                worst_fuel = fuel_performance['RMSE_over_generation'].idxmax()
                worst_count = fuel_performance.loc[worst_fuel, 'performance_classification']
                
                if worst_count > 0:
                    recommendations.append(f"⛽ FUEL TYPE: {worst_fuel} generators show higher error rates ({worst_count} poor/critical)")
                    recommendations.append(f"   → Review {worst_fuel} generator modeling parameters")
        
        # General recommendations
        recommendations.extend([
            "",
            "📋 GENERAL RECOMMENDATIONS:",
            "• Prioritize generators with multiple performance issues",
            "• Review forecast models for generators with R² < 0.5",
            "• Monitor generators with increasing error trends",
            "• Update capacity constraints for generators with Pmax issues",
            "• Consider market condition correlation analysis",
            "",
            "🔄 FOLLOW-UP ACTIONS:",
            "• Schedule weekly performance monitoring",
            "• Set up automated alerts for new chronic errors",
            "• Review and update performance thresholds quarterly",
            "• Coordinate with operations team for generators needing immediate attention"
        ])
        
        # Display recommendations
        y_pos = 0.95
        for rec in recommendations:
            if rec.startswith('🚨') or rec.startswith('⚠️'):
                color = 'red' if rec.startswith('🚨') else 'orange'
                fontweight = 'bold'
            elif rec.startswith('🔄') or rec.startswith('📊') or rec.startswith('⛽'):
                color = 'blue'
                fontweight = 'bold'
            elif rec.startswith('📋') or rec.startswith('🔄 FOLLOW-UP'):
                color = 'green'
                fontweight = 'bold'
            elif rec.startswith('   →') or rec.startswith('•'):
                color = 'black'
                fontweight = 'normal'
            else:
                color = 'black'
                fontweight = 'normal'
            
            ax.text(0.05, y_pos, rec, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', color=color, fontweight=fontweight)
            y_pos -= 0.05
            
            if y_pos < 0.05:
                break
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
