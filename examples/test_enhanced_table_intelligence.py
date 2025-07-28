"""
Test script for Enhanced Table Intelligence with ydata-profiling ML capabilities
Uses real e-commerce data from the Olist dataset
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.table_intelligence import EnhancedTableIntelligenceLayer, LLMConfig

def load_olist_data():
    """Load Olist e-commerce datasets"""
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'ecommerce'
    
    datasets = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv'
    }
    
    loaded_data = {}
    for name, filename in datasets.items():
        file_path = data_path / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                loaded_data[name] = df
                print(f"✓ Loaded {name}: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ File not found: {filename}")
    
    return loaded_data

def analyze_table(intelligence, table_name, df):
    """Analyze a single table and display results"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {table_name.upper()}")
    print(f"{'='*80}")
    
    # Perform comprehensive analysis
    profile = intelligence.analyze_table_comprehensive(table_name, df)
    
    # Basic information
    print(f"\nTable: {profile.table_name}")
    print(f"Rows: {profile.row_count:,}, Columns: {profile.column_count}")
    print(f"Business Domain: {profile.business_domain or 'Not detected'}")
    print(f"Table Type: {profile.table_type}")
    print(f"\nSemantic Summary: {profile.semantic_summary}")
    
    # Data Quality Analysis
    if profile.quality_profile:
        print(f"\n=== Data Quality Analysis ===")
        print(f"Overall Quality Score: {profile.quality_profile.overall_quality_score}/100")
        print(f"Alert Summary: {profile.quality_profile.alert_summary}")
        
        if profile.quality_profile.quality_recommendations:
            print("\nQuality Recommendations:")
            for rec in profile.quality_profile.quality_recommendations[:3]:  # Show top 3
                print(f"  - {rec}")
    
    # Outlier Analysis
    if profile.outlier_analysis:
        print(f"\n=== Outlier Analysis ===")
        outlier_summary = profile.outlier_analysis.get('outlier_summary', {})
        print(f"Columns analyzed: {len(outlier_summary)}")
        
        if profile.outlier_analysis.get('high_impact_outliers'):
            print("\nHigh Impact Outliers:")
            for outlier in profile.outlier_analysis['high_impact_outliers'][:3]:  # Show top 3
                print(f"  - {outlier['column']}: {outlier['percentage']:.1f}% outliers ({outlier['impact']} impact)")
                print(f"    Recommendation: {outlier['recommendation']}")
    
    # Correlation Analysis
    if profile.correlation_analysis:
        print(f"\n=== Correlation Analysis ===")
        linear_rels = profile.correlation_analysis.get('linear_relationships', {})
        print(f"Strong linear relationships: {len(linear_rels.get('strong_linear', []))}")
        print(f"Moderate linear relationships: {len(linear_rels.get('moderate_linear', []))}")
        
        # Show redundant features
        redundant = profile.correlation_analysis.get('feature_redundancy', [])
        if redundant:
            print(f"\nRedundant Features Detected:")
            for r in redundant[:2]:  # Show top 2
                print(f"  - {r['variables']}: correlation = {r['correlation_value']:.3f}")
    
    # ML Readiness Assessment
    print(f"\n=== ML Readiness Assessment ===")
    print(f"ML Readiness Score: {profile.ml_readiness_score}/100")
    print("\nTop ML Readiness Factors:")
    for factor in profile.ml_readiness_factors[:3]:  # Show top 3
        print(f"  - {factor}")
    
    # Column Classification
    print(f"\n=== Column Classification ===")
    print(f"Measures: {', '.join(profile.measure_columns[:5])}{'...' if len(profile.measure_columns) > 5 else ''}")
    print(f"Dimensions: {', '.join(profile.dimension_columns[:5])}{'...' if len(profile.dimension_columns) > 5 else ''}")
    print(f"Identifiers: {', '.join(profile.identifier_columns[:5])}{'...' if len(profile.identifier_columns) > 5 else ''}")
    print(f"Temporal: {', '.join(profile.temporal_columns[:5])}{'...' if len(profile.temporal_columns) > 5 else ''}")
    
    return profile

def test_enhanced_intelligence_with_real_data():
    """Test the enhanced table intelligence capabilities with real e-commerce data"""
    print("=== Enhanced Table Intelligence Test with Olist E-commerce Data ===\n")
    
    # Load datasets
    print("Loading Olist datasets...")
    datasets = load_olist_data()
    
    if not datasets:
        print("No datasets loaded. Please check the data directory.")
        return
    
    # Initialize enhanced intelligence layer
    print("\nInitializing Enhanced Table Intelligence Layer...")
    intelligence = EnhancedTableIntelligenceLayer(
        model_name='all-MiniLM-L6-v2',
        enable_profiling=True,
        cache_embeddings=True,
        use_llm_summaries=False,  # Set to True if you have Ollama configured
        enable_advanced_quality=True,
        enable_outlier_detection=True,
        enable_correlation_analysis=True
    )
    
    # Analyze key tables
    # Focus on the most interesting tables for analysis
    priority_tables = ['orders', 'order_items', 'order_payments', 'customers', 'products']
    
    profiles = {}
    for table_name in priority_tables:
        if table_name in datasets:
            profiles[table_name] = analyze_table(intelligence, table_name, datasets[table_name])
    
    # Cross-table insights
    print(f"\n{'='*80}")
    print("CROSS-TABLE INSIGHTS")
    print(f"{'='*80}")
    
    # Compare table types
    print("\n=== Table Type Distribution ===")
    table_types = {}
    for name, profile in profiles.items():
        table_type = profile.table_type
        table_types[table_type] = table_types.get(table_type, []) + [name]
    
    for t_type, tables in table_types.items():
        print(f"{t_type}: {', '.join(tables)}")
    
    # Compare data quality scores
    print("\n=== Data Quality Comparison ===")
    quality_scores = []
    for name, profile in profiles.items():
        if profile.quality_profile:
            quality_scores.append((name, profile.quality_profile.overall_quality_score))
    
    quality_scores.sort(key=lambda x: x[1], reverse=True)
    for name, score in quality_scores:
        print(f"{name}: {score}/100 {'⭐' * int(score/20)}")
    
    # Compare ML readiness
    print("\n=== ML Readiness Comparison ===")
    ml_scores = [(name, profile.ml_readiness_score) for name, profile in profiles.items()]
    ml_scores.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in ml_scores:
        status = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        print(f"{status} {name}: {score}/100")
    
    # Identify tables with high correlations
    print("\n=== Tables with Strong Feature Correlations ===")
    for name, profile in profiles.items():
        if profile.correlation_analysis:
            strong_corr = len(profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', []))
            if strong_corr > 0:
                print(f"{name}: {strong_corr} strong correlations found")

if __name__ == "__main__":
    test_enhanced_intelligence_with_real_data()