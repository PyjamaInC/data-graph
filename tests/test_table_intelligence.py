#!/usr/bin/env python3
"""
Test script for the Table Intelligence Layer

This script demonstrates the enhanced table analysis capabilities
including semantic profiling, embedding generation, and concept extraction.
"""

import sys
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.knowledge_graph.table_intelligence import TableIntelligenceLayer, TableProfile
    from src.schema.schema_manager import SchemaManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure to install required dependencies: pip install sentence-transformers")
    sys.exit(1)


def test_table_intelligence():
    """Test the table intelligence layer with sample data"""
    logger.info("🧠 Testing Table Intelligence Layer")
    
    # Initialize the intelligence layer
    intelligence = TableIntelligenceLayer(
        model_name='all-MiniLM-L6-v2',
        enable_profiling=True,
        cache_embeddings=True
    )
    
    # Load sample ecommerce data
    data_path = Path("data/raw/ecommerce")
    
    # Test with customers table
    if (data_path / "olist_customers_dataset.csv").exists():
        logger.info("📊 Analyzing customers table...")
        customers_df = pd.read_csv(data_path / "olist_customers_dataset.csv")
        
        # Analyze table
        customers_profile = intelligence.analyze_table("customers", customers_df)
        
        print(f"\\n=== CUSTOMERS TABLE ANALYSIS ===")
        print(f"Table Type: {customers_profile.table_type}")
        print(f"Business Domain: {customers_profile.business_domain}")
        print(f"Data Quality Score: {customers_profile.data_quality_score}")
        print(f"\\nSemantic Summary:")
        print(f"  {customers_profile.semantic_summary}")
        print(f"\\nKey Concepts: {', '.join(customers_profile.key_concepts)}")
        print(f"\\nColumn Categories:")
        print(f"  - Identifiers: {customers_profile.identifier_columns}")
        print(f"  - Dimensions: {customers_profile.dimension_columns}")
        print(f"  - Measures: {customers_profile.measure_columns}")
        print(f"  - Temporal: {customers_profile.temporal_columns}")
        
        if customers_profile.embedding is not None:
            print(f"\\nEmbedding Shape: {customers_profile.embedding.shape}")
        else:
            print("\\nEmbedding: Not generated (sentence-transformers not available)")
    
    # Test with orders table
    if (data_path / "olist_orders_dataset.csv").exists():
        logger.info("📊 Analyzing orders table...")
        orders_df = pd.read_csv(data_path / "olist_orders_dataset.csv")
        
        # Analyze table
        orders_profile = intelligence.analyze_table("orders", orders_df)
        
        print(f"\\n=== ORDERS TABLE ANALYSIS ===")
        print(f"Table Type: {orders_profile.table_type}")
        print(f"Business Domain: {orders_profile.business_domain}")
        print(f"Data Quality Score: {orders_profile.data_quality_score}")
        print(f"\\nSemantic Summary:")
        print(f"  {orders_profile.semantic_summary}")
        print(f"\\nKey Concepts: {', '.join(orders_profile.key_concepts)}")
        print(f"\\nColumn Categories:")
        print(f"  - Identifiers: {orders_profile.identifier_columns}")
        print(f"  - Dimensions: {orders_profile.dimension_columns}")
        print(f"  - Measures: {orders_profile.measure_columns}")
        print(f"  - Temporal: {orders_profile.temporal_columns}")
        
        # Test similarity comparison if both profiles exist
        if 'customers_profile' in locals() and customers_profile.embedding is not None and orders_profile.embedding is not None:
            similarity = intelligence.compare_table_similarity(customers_profile, orders_profile)
            print(f"\\n🔍 Semantic Similarity (customers ↔ orders): {similarity:.3f}")
    
    # Test with products table
    if (data_path / "olist_products_dataset.csv").exists():
        logger.info("📊 Analyzing products table...")
        products_df = pd.read_csv(data_path / "olist_products_dataset.csv")
        
        products_profile = intelligence.analyze_table("products", products_df)
        
        print(f"\\n=== PRODUCTS TABLE ANALYSIS ===")
        print(f"Table Type: {products_profile.table_type}")
        print(f"Business Domain: {products_profile.business_domain}")
        print(f"Data Quality Score: {products_profile.data_quality_score}")
        print(f"\\nSemantic Summary:")
        print(f"  {products_profile.semantic_summary}")
        print(f"\\nKey Concepts: {', '.join(products_profile.key_concepts)}")
        
        # Compare with orders if available
        if 'orders_profile' in locals() and orders_profile.embedding is not None and products_profile.embedding is not None:
            similarity = intelligence.compare_table_similarity(orders_profile, products_profile)
            print(f"\\n🔍 Semantic Similarity (orders ↔ products): {similarity:.3f}")


def create_sample_data_test():
    """Create sample data for testing if real data is not available"""
    logger.info("📝 Creating sample data for testing...")
    
    # Sample customer data
    customers_data = {
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'customer_city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'customer_state': ['NY', 'CA', 'IL', 'TX', 'AZ'],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-01-30', '2023-03-10', '2023-02-05'],
        'is_premium': [True, False, True, False, True],
        'lifetime_value': [1250.50, 890.25, 2100.75, 450.00, 1800.30]
    }
    
    # Sample order data
    orders_data = {
        'order_id': ['O001', 'O002', 'O003', 'O004', 'O005'],
        'customer_id': ['C001', 'C002', 'C001', 'C003', 'C002'],
        'order_date': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-03', '2023-06-04'],
        'order_status': ['delivered', 'processing', 'delivered', 'shipped', 'delivered'],
        'order_amount': [125.50, 89.25, 210.75, 45.00, 180.30],
        'shipping_cost': [5.99, 7.50, 12.99, 3.99, 8.99],
        'payment_type': ['credit_card', 'debit_card', 'credit_card', 'paypal', 'credit_card']
    }
    
    customers_df = pd.DataFrame(customers_data)
    orders_df = pd.DataFrame(orders_data)
    
    # Initialize intelligence layer
    intelligence = TableIntelligenceLayer()
    
    # Analyze sample tables
    customers_profile = intelligence.analyze_table("customers", customers_df)
    orders_profile = intelligence.analyze_table("orders", orders_df)
    
    print("\\n=== SAMPLE DATA ANALYSIS ===")
    print(f"\\n📊 Customers Table:")
    print(f"  Type: {customers_profile.table_type}")
    print(f"  Domain: {customers_profile.business_domain}")
    print(f"  Quality: {customers_profile.data_quality_score}")
    print(f"  Summary: {customers_profile.semantic_summary}")
    
    print(f"\\n📊 Orders Table:")
    print(f"  Type: {orders_profile.table_type}")
    print(f"  Domain: {orders_profile.business_domain}")
    print(f"  Quality: {orders_profile.data_quality_score}")
    print(f"  Summary: {orders_profile.semantic_summary}")
    
    # Test similarity
    similarity = intelligence.compare_table_similarity(customers_profile, orders_profile)
    print(f"\\n🔍 Table Similarity: {similarity:.3f}")


if __name__ == "__main__":
    logger.info("🚀 Starting Table Intelligence Layer Tests")
    
    # Check if real data exists
    data_path = Path("data/raw/ecommerce")
    if data_path.exists() and any(data_path.glob("*.csv")):
        test_table_intelligence()
    else:
        logger.warning("Real ecommerce data not found, using sample data")
        create_sample_data_test()
    
    logger.info("✅ Table Intelligence Layer testing complete!")