#!/usr/bin/env python3
"""
Debug script to test operation generation
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from agents.react_agents.enhanced_data_exploration_agent import IntelligentOperationGenerator

# Create sample data
tables = {
    'orders': pd.DataFrame({
        'order_id': ['1', '2', '3'],
        'status': ['delivered', 'shipped', 'processing'],
        'price': [100, 200, 150]
    })
}

# Test operation generation
profiles = {
    'orders': type('Profile', (), {
        'temporal_columns': [],
        'measure_columns': ['price'],
        'dimension_columns': ['status']
    })()
}

question_analysis = {
    'analysis_strategy': {
        'steps': ['assess_data_completeness', 'identify_missing_values'],
        'priority_tables': ['orders']
    }
}

generator = IntelligentOperationGenerator(profiles, question_analysis)

for i in range(1, 3):
    print(f"\n=== Iteration {i} ===")
    result = generator.generate_next_operation(i, {})
    print(f"Step: {result['step']}")
    print(f"Operation: {result['operation']}")
    
    # Test execution
    try:
        exec(result['operation'])
        print("✅ Operation executed successfully")
    except Exception as e:
        print(f"❌ Execution failed: {e}")