#!/usr/bin/env python3
"""
Project Transfer Verification Script

This script automatically checks all items from TRANSFER_CHECKLIST.md to verify
that the Generator Analysis Platform transferred correctly to the new location.
"""

import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd

def print_status(check_name, status, details=""):
    """Print formatted status message."""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {check_name}")
    if details:
        print(f"   {details}")
    print()

def verify_file_exists(filepath, description):
    """Verify that a file exists."""
    exists = os.path.exists(filepath)
    print_status(f"{description}", exists, f"Path: {filepath}")
    return exists

def verify_python_file(filepath, description):
    """Verify that a Python file exists and is importable."""
    if not os.path.exists(filepath):
        print_status(f"{description}", False, f"File not found: {filepath}")
        return False
    
    try:
        # Try to load the module to check for syntax errors
        spec = importlib.util.spec_from_file_location("temp_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print_status(f"{description}", True, f"File exists and is syntactically valid")
        return True
    except Exception as e:
        print_status(f"{description}", False, f"File exists but has issues: {e}")
        return False

def check_config_settings():
    """Check critical configuration settings."""
    print("🔍 Checking Configuration Integrity...")
    
    config_file = "Auto_weekly_generator_analyzer2.py"
    if not os.path.exists(config_file):
        print_status("Configuration file", False, f"{config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for user-modified settings
        min_mw_check = "MIN_MW_TO_BE_ANALYZED = 20" in content or "MIN_MW_TO_BE_ANALYZED=20" in content
        bid_validation_check = "RUN_BID_VALIDATION = True" in content or "RUN_BID_VALIDATION=True" in content
        
        print_status("MIN_MW_TO_BE_ANALYZED = 20", min_mw_check, "User-modified setting")
        print_status("RUN_BID_VALIDATION = True", bid_validation_check, "User-enabled feature")
        
        # Check for parallel processing settings
        batch_size_check = "BATCH_SIZE" in content
        n_jobs_check = "N_JOBS" in content or "n_jobs" in content
        
        print_status("Parallel processing settings", batch_size_check and n_jobs_check, 
                    "BATCH_SIZE and N_JOBS configurations found")
        
        return min_mw_check and bid_validation_check
        
    except Exception as e:
        print_status("Configuration file reading", False, f"Error reading config: {e}")
        return False

def check_core_functionality():
    """Check that core Python files are present and valid."""
    print("🔍 Checking Core Functionality...")
    
    core_files = [
        ("Auto_weekly_generator_analyzer2.py", "Main analysis orchestrator"),
        ("performance_report_generator.py", "PDF report generator (with Pmax fix)"),
        ("bid_validation.py", "Bid validation analyzer"),
        ("bid_validation_integration.py", "Bid validation integration")
    ]
    
    all_good = True
    for filepath, description in core_files:
        if not verify_python_file(filepath, description):
            all_good = False
    
    return all_good

def check_documentation():
    """Check that all documentation files are present."""
    print("🔍 Checking Documentation Completeness...")
    
    doc_files = [
        ("README.md", "Comprehensive business and technical overview"),
        ("API_REFERENCE.md", "Complete API documentation"),
        ("BID_VALIDATION_README.md", "Bid validation system guide"),
        ("PROJECT_OVERVIEW.md", "Strategic business context"),
        ("COLLABORATION_GUIDE.md", "Development standards"),
        ("CODE_PATTERNS.md", "Technical patterns and best practices"),
        ("COPILOT_CONTEXT.md", "AI assistant continuity context"),
        ("TRANSFER_CHECKLIST.md", "This verification checklist")
    ]
    
    all_good = True
    for filepath, description in doc_files:
        if not verify_file_exists(filepath, description):
            all_good = False
    
    return all_good

def check_file_structure():
    """Check overall file structure."""
    print("🔍 Checking File Structure Validation...")
    
    # Check for Python files
    python_files = [f for f in os.listdir('.') if f.endswith('.py')]
    print_status(f"Python scripts found", len(python_files) > 0, f"Found {len(python_files)} Python files")
    
    # Check for CSV data files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print_status(f"Data files (CSV)", len(csv_files) > 0, f"Found {len(csv_files)} CSV files")
    
    # Check for markdown files
    md_files = [f for f in os.listdir('.') if f.endswith('.md')]
    print_status(f"Documentation files (MD)", len(md_files) >= 7, f"Found {len(md_files)} markdown files (expected ≥7)")
    
    return len(python_files) > 0 and len(md_files) >= 7

def test_basic_functionality():
    """Test basic functionality that doesn't require external dependencies."""
    print("🔍 Testing Basic Functionality...")
    
    try:
        # Test if we can import the performance report generator
        if os.path.exists("performance_report_generator.py"):
            spec = importlib.util.spec_from_file_location("perf_gen", "performance_report_generator.py")
            perf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(perf_module)
            
            # Try to instantiate the class
            generator = perf_module.PerformanceReportGenerator()
            print_status("PerformanceReportGenerator instantiation", True, "Class can be created successfully")
            
            # Check if the fixed method exists
            has_method = hasattr(generator, '_get_pmax_from_resource_db')
            print_status("Pmax lookup method exists", has_method, "_get_pmax_from_resource_db method found")
            
            return True
            
    except Exception as e:
        print_status("Basic functionality test", False, f"Error: {e}")
        return False

def check_copilot_context():
    """Verify GitHub Copilot context file."""
    print("🔍 Checking GitHub Copilot Continuity...")
    
    if not os.path.exists("COPILOT_CONTEXT.md"):
        print_status("COPILOT_CONTEXT.md", False, "Context file missing")
        return False
    
    try:
        with open("COPILOT_CONTEXT.md", 'r') as f:
            content = f.read()
        
        # Check for key sections
        key_sections = [
            "Pmax Lookup Pattern",
            "Performance Fixes Completed", 
            "Configuration Management",
            "Critical Code Patterns"
        ]
        
        sections_found = sum(1 for section in key_sections if section in content)
        print_status("COPILOT_CONTEXT.md content", sections_found >= 3, 
                    f"Found {sections_found}/{len(key_sections)} key sections")
        
        # Check for the critical Pmax fix documentation
        pmax_fix_documented = "orig_name" in content and "Pmax" in content
        print_status("Pmax fix documented", pmax_fix_documented, "Critical fix is documented for future reference")
        
        return sections_found >= 3 and pmax_fix_documented
        
    except Exception as e:
        print_status("COPILOT_CONTEXT.md reading", False, f"Error reading context file: {e}")
        return False

def main():
    """Run complete transfer verification."""
    print("🚀 Generator Analysis Platform - Transfer Verification")
    print("=" * 60)
    print(f"Verifying transfer to: {os.getcwd()}")
    print("=" * 60)
    print()
    
    # Run all checks
    checks = [
        ("Configuration Integrity", check_config_settings),
        ("Core Functionality", check_core_functionality),
        ("Documentation Completeness", check_documentation),
        ("File Structure Validation", check_file_structure),
        ("Basic Functionality Test", test_basic_functionality),
        ("GitHub Copilot Continuity", check_copilot_context)
    ]
    
    results = []
    for check_name, check_function in checks:
        print(f"\n{'=' * 40}")
        print(f"CHECKING: {check_name}")
        print('=' * 40)
        
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print_status(f"{check_name} (EXCEPTION)", False, f"Unexpected error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status_symbol = "✅" if result else "❌"
        print(f"{status_symbol} {check_name}")
    
    print(f"\nOVERALL RESULT: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 SUCCESS! Project transfer completed successfully!")
        print("   All critical components verified and ready for use.")
        print("\n📋 Next Steps:")
        print("   1. Open project in VS Code")
        print("   2. Test running Auto_weekly_generator_analyzer2.py")
        print("   3. Verify PDF generation works correctly")
        print("   4. Ask GitHub Copilot about previous context using COPILOT_CONTEXT.md")
    else:
        print(f"\n⚠️  ISSUES DETECTED: {total - passed} checks failed")
        print("   Please review the failed items above and ensure all files copied correctly.")
        print("   You may need to re-copy some files from the original location.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
