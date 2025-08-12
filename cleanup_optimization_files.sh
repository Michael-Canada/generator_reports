#!/bin/bash
# Cleanup script to remove temporary optimization files

echo "🧹 CLEANING UP TEMPORARY OPTIMIZATION FILES"
echo "==========================================="

# Files to delete (temporary/testing files)
files_to_delete=(
    "test_optimizations.py"
    "test_integrated_optimizations.py"
    "final_optimization_results.py"
    "optimization_summary.py"
    "optimization_suggestions.py"
    "optimized_api_client.py"
    "quick_start_optimization.py"
    "OPTIMIZATION_COMPLETE.md"
)

# Check which files exist and delete them
deleted_count=0
for file in "${files_to_delete[@]}"; do
    if [ -f "$file" ]; then
        echo "🗑️  Deleting: $file"
        rm "$file"
        ((deleted_count++))
    else
        echo "⚠️  Not found: $file"
    fi
done

echo ""
echo "✅ CLEANUP COMPLETE"
echo "==================="
echo "Files deleted: $deleted_count"
echo ""
echo "📁 KEPT IMPORTANT FILES:"
echo "• Auto_weekly_generator_analyzer2.py (optimized main code)"
echo "• bid_validation_integration.py (your current file)"
echo "• bid_validation.py (core functionality)"
echo "• performance_report_generator.py (PDF generation)"
echo "• run_optimized_analysis.sh (run script)"
echo ""
echo "🚀 Your optimized analysis is ready to use!"
echo "   All optimizations are integrated into the main code."
