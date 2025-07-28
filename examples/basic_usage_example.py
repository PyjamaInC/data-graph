"""
Basic usage example for Enhanced Table Intelligence
This example shows how to use the enhanced capabilities without requiring all dependencies
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.table_intelligence import TableIntelligenceLayer

def create_simple_sales_data():
    """Create a simple sales dataset"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_records = len(dates) * 10  # 10 sales per day on average
    
    data = {
        'sale_id': range(1, n_records + 1),
        'sale_date': np.random.choice(dates, n_records),
        'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_records),
        'customer_id': np.random.choice(range(1, 101), n_records),
        'quantity': np.random.poisson(3, n_records),
        'unit_price': np.random.choice([9.99, 19.99, 29.99, 49.99], n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'payment_method': np.random.choice(['Credit', 'Debit', 'Cash'], n_records, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    
    return df

def main():
    """Demonstrate basic table intelligence functionality"""
    print("=== Basic Table Intelligence Example ===\n")
    
    # Create sample data
    df = create_simple_sales_data()
    print(f"Created sales dataset with {len(df)} records")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Initialize basic table intelligence
    intelligence = TableIntelligenceLayer(
        model_name='all-MiniLM-L6-v2',
        enable_profiling=False,  # Disable to avoid dependency issues
        cache_embeddings=True,
        use_llm_summaries=False
    )
    
    # Perform analysis
    print("\n\nAnalyzing table...")
    profile = intelligence.analyze_table('sales_transactions', df)
    
    # Display results
    print("\n=== Analysis Results ===")
    print(f"\nTable: {profile.table_name}")
    print(f"Total Records: {profile.row_count:,}")
    print(f"Total Columns: {profile.column_count}")
    
    print(f"\nBusiness Domain: {profile.business_domain or 'Not detected'}")
    print(f"Table Type: {profile.table_type}")
    print(f"Data Quality Score: {profile.data_quality_score:.2%}")
    
    print(f"\n=== Semantic Summary ===")
    print(profile.semantic_summary)
    
    print(f"\n=== Key Concepts ===")
    print(f"Identified concepts: {', '.join(profile.key_concepts)}")
    
    print(f"\n=== Column Classification ===")
    print(f"Measure Columns ({len(profile.measure_columns)}): {', '.join(profile.measure_columns)}")
    print(f"Dimension Columns ({len(profile.dimension_columns)}): {', '.join(profile.dimension_columns)}")
    print(f"Identifier Columns ({len(profile.identifier_columns)}): {', '.join(profile.identifier_columns)}")
    print(f"Temporal Columns ({len(profile.temporal_columns)}): {', '.join(profile.temporal_columns)}")
    
    # Show column insights for selected columns
    print(f"\n=== Column Insights (Sample) ===")
    column_insights = intelligence._analyze_columns(df)
    
    for insight in column_insights[:3]:  # Show first 3 columns
        print(f"\nColumn: {insight.column_name}")
        print(f"  Data Type: {insight.data_type.value}")
        print(f"  Semantic Role: {insight.semantic_role.value}")
        print(f"  Uniqueness: {insight.uniqueness_ratio:.1%}")
        print(f"  Completeness: {insight.completeness_ratio:.1%}")
        if insight.key_patterns:
            print(f"  Patterns: {', '.join(insight.key_patterns)}")

if __name__ == "__main__":
    main()