#!/usr/bin/env python3
"""
Example: Testing LLM-Enhanced Table Intelligence

This example demonstrates how to use the LLM-enhanced table intelligence
features to get richer semantic summaries and relationship descriptions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.table_intelligence import TableIntelligenceLayer, LLMConfig
from src.knowledge_graph.llm_config import LLMConfigManager


def create_sample_data():
    """Create sample tables for testing"""
    
    # Customer table
    customers = pd.DataFrame({
        'customer_id': range(1, 101),
        'first_name': [f'Customer_{i}' for i in range(1, 101)],
        'last_name': [f'LastName_{i}' for i in range(1, 101)],
        'email': [f'customer{i}@example.com' for i in range(1, 101)],
        'registration_date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 100),
        'lifetime_value': np.random.uniform(100, 10000, 100)
    })
    
    # Orders table
    orders = pd.DataFrame({
        'order_id': range(1, 501),
        'customer_id': np.random.choice(range(1, 101), 500),
        'order_date': pd.date_range('2023-01-01', periods=500, freq='6H'),
        'total_amount': np.random.uniform(10, 1000, 500),
        'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled'], 500),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], 500),
        'shipping_cost': np.random.uniform(0, 50, 500)
    })
    
    # Products table
    products = pd.DataFrame({
        'product_id': range(1, 51),
        'product_name': [f'Product_{i}' for i in range(1, 51)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], 50),
        'price': np.random.uniform(10, 500, 50),
        'stock_quantity': np.random.randint(0, 1000, 50),
        'supplier_id': np.random.randint(1, 11, 50)
    })
    
    return {
        'customers': customers,
        'orders': orders,
        'products': products
    }


def test_basic_intelligence():
    """Test basic table intelligence without LLM"""
    print("=" * 80)
    print("TESTING BASIC TABLE INTELLIGENCE (WITHOUT LLM)")
    print("=" * 80)
    
    # Create table intelligence layer without LLM
    table_intel = TableIntelligenceLayer(
        use_llm_summaries=False
    )
    
    # Create sample data
    tables = create_sample_data()
    
    # Analyze each table
    for table_name, df in tables.items():
        print(f"\nAnalyzing table: {table_name}")
        profile = table_intel.analyze_table(table_name, df)
        
        print(f"Semantic Summary: {profile.semantic_summary}")
        print(f"Business Domain: {profile.business_domain}")
        print(f"Table Type: {profile.table_type}")
        print(f"Key Concepts: {', '.join(profile.key_concepts[:5])}")


def test_llm_enhanced_intelligence():
    """Test LLM-enhanced table intelligence"""
    print("\n" + "=" * 80)
    print("TESTING LLM-ENHANCED TABLE INTELLIGENCE")
    print("=" * 80)
    
    # Setup LLM configuration
    llm_config_manager = LLMConfigManager()
    
    # Test Ollama connection
    print("\nTesting Ollama connection...")
    if not llm_config_manager.test_provider_connection("ollama"):
        print("⚠️  Ollama is not available. Please ensure Ollama is running.")
        print("   Run: ollama serve")
        print("   Then: ollama pull llama3.2:3b")
        return
    
    print("✅ Ollama is available!")
    
    # Get LLM config
    llm_params = llm_config_manager.get_llm_config_for_table_intelligence()
    llm_config = LLMConfig(**llm_params)
    
    # Create table intelligence layer with LLM
    table_intel = TableIntelligenceLayer(
        use_llm_summaries=True,
        llm_config=llm_config
    )
    
    # Create sample data
    tables = create_sample_data()
    
    # Analyze each table
    profiles = {}
    for table_name, df in tables.items():
        print(f"\n{'='*60}")
        print(f"Analyzing table: {table_name}")
        print(f"{'='*60}")
        
        profile = table_intel.analyze_table(table_name, df)
        profiles[table_name] = profile
        
        print(f"\n📊 Basic Info:")
        print(f"   Rows: {profile.row_count}")
        print(f"   Columns: {profile.column_count}")
        print(f"   Table Type: {profile.table_type}")
        print(f"   Business Domain: {profile.business_domain}")
        
        print(f"\n📝 LLM-Enhanced Semantic Summary:")
        print(f"   {profile.semantic_summary}")
        
        print(f"\n🔑 Key Concepts:")
        print(f"   {', '.join(profile.key_concepts[:8])}")
        
        print(f"\n📏 Column Distribution:")
        print(f"   Measures: {', '.join(profile.measure_columns[:3])}")
        print(f"   Dimensions: {', '.join(profile.dimension_columns[:3])}")
        print(f"   Identifiers: {', '.join(profile.identifier_columns[:3])}")
    
    # Test relationship description generation
    if table_intel.llm_summarizer:
        print(f"\n{'='*80}")
        print("TESTING LLM-ENHANCED RELATIONSHIP DESCRIPTIONS")
        print(f"{'='*80}")
        
        # Generate a relationship description between customers and orders
        rel_desc = table_intel.llm_summarizer.generate_relationship_description(
            table1_name="customers",
            table1_summary=profiles['customers'].semantic_summary,
            table1_columns=['customer_id', 'customer_segment', 'lifetime_value'],
            table2_name="orders",
            table2_summary=profiles['orders'].semantic_summary,
            table2_columns=['order_id', 'customer_id', 'total_amount', 'order_date'],
            relationship_type="one-to-many",
            confidence=0.95,
            linking_columns=[('customer_id', 'customer_id')]
        )
        
        print(f"\n🔗 Relationship: customers → orders")
        print(f"   {rel_desc}")


def compare_summaries():
    """Compare basic vs LLM-enhanced summaries"""
    print("\n" + "=" * 80)
    print("COMPARING BASIC VS LLM-ENHANCED SUMMARIES")
    print("=" * 80)
    
    # Create sample data
    tables = create_sample_data()
    customers_df = tables['customers']
    
    # Basic summary
    basic_intel = TableIntelligenceLayer(use_llm_summaries=False)
    basic_profile = basic_intel.analyze_table('customers', customers_df)
    
    print("\n📊 BASIC SUMMARY:")
    print(f"   {basic_profile.semantic_summary}")
    
    # LLM-enhanced summary (if available)
    llm_config_manager = LLMConfigManager()
    if llm_config_manager.test_provider_connection("ollama"):
        llm_params = llm_config_manager.get_llm_config_for_table_intelligence()
        llm_config = LLMConfig(**llm_params)
        
        llm_intel = TableIntelligenceLayer(
            use_llm_summaries=True,
            llm_config=llm_config
        )
        llm_profile = llm_intel.analyze_table('customers', customers_df)
        
        print("\n🤖 LLM-ENHANCED SUMMARY:")
        print(f"   {llm_profile.semantic_summary}")
        
        print("\n✨ Key Differences:")
        print("   - LLM provides business context and purpose")
        print("   - LLM identifies relationships between columns")
        print("   - LLM suggests how the table fits into business processes")
    else:
        print("\n⚠️  LLM not available for comparison")


if __name__ == "__main__":
    # Test basic intelligence
    test_basic_intelligence()
    
    # Test LLM-enhanced intelligence
    test_llm_enhanced_intelligence()
    
    # Compare summaries
    compare_summaries()
    
    print("\n" + "=" * 80)
    print("✅ Testing completed!")
    print("=" * 80)