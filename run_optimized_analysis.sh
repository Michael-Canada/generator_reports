#!/bin/bash
# Optimized Generator Analysis Runner
# This script runs your optimized analysis with monitoring

echo "🚀 STARTING OPTIMIZED GENERATOR ANALYSIS"
echo "========================================"
echo "Timestamp: $(date)"
echo ""

echo "📊 Expected Performance:"
echo "• Previous runtime: ~90 minutes"
echo "• Optimized runtime: ~18-25 minutes"
echo "• Expected improvement: 60-80% faster"
echo ""

echo "🔍 Monitoring optimizations..."
echo "Watch for: 🚀 OPTIMIZED, ✅ Bulk fetch, 📊 Performance"
echo ""

# Change to the analysis directory
cd "/Users/michael.simantov/Library/CloudStorage/OneDrive-DrillingInfo/Documents/generator_analysis"

# Run the optimized analysis with timing
echo "⏱️ Starting analysis at: $(date)"
start_time=$(date +%s)

# Run the analysis
conda run -n placebo_api_local python Auto_weekly_generator_analyzer2.py

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$((runtime / 60))
runtime_seconds=$((runtime % 60))

echo ""
echo "✅ ANALYSIS COMPLETED!"
echo "========================================"
echo "Finished at: $(date)"
echo "Total runtime: ${runtime_minutes}m ${runtime_seconds}s"
echo ""

if [ $runtime_minutes -lt 30 ]; then
    echo "🎉 EXCELLENT! Analysis completed faster than expected!"
    improvement=$(echo "scale=1; (90 - $runtime_minutes) / 90 * 100" | bc)
    echo "   Performance improvement: ~${improvement}%"
else
    echo "📊 Analysis completed. Compare with previous 90-minute baseline."
fi

echo ""
echo "📁 Check your output files for:"
echo "• generator_forecast_stats_*.csv"
echo "• generator_anomalies_*.csv" 
echo "• bid_validation_*.csv"
echo "• *.pdf reports"
