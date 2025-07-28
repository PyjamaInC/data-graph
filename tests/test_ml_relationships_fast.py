import asyncio
import sys
sys.path.append('.')

from src.data.connectors.csv_connector import CSVConnector
from src.knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
from src.knowledge_graph.relationship_detector import MLRelationshipDetector
import pandas as pd

async def test_ml_relationships_fast():
    """Fast test of ML-powered relationship detection with specific examples"""
    
    # Load data
    print("1. Loading dataset...")
    connector = CSVConnector()
    tables = await connector.load_data({
        'data_path': 'data/raw/ecommerce'
    })
    
    # Test specific column pairs to demonstrate different relationship types
    print("\n2. Testing ML Relationship Detection on Specific Column Pairs:")
    print("=" * 80)
    
    detector = MLRelationshipDetector()
    
    # Test cases for different relationship types
    test_cases = [
        # Foreign Key relationships
        {
            'table1': 'olist_orders_dataset',
            'col1': 'customer_id',
            'table2': 'olist_customers_dataset', 
            'col2': 'customer_id',
            'expected': 'FOREIGN_KEY'
        },
        # Same domain (similar categorical values)
        {
            'table1': 'olist_customers_dataset',
            'col1': 'customer_state',
            'table2': 'olist_sellers_dataset',
            'col2': 'seller_state',
            'expected': 'SAME_DOMAIN or SIMILAR_VALUES'
        },
        # Numeric relationships within same table
        {
            'table1': 'olist_products_dataset',
            'col1': 'product_length_cm',
            'table2': 'olist_products_dataset',
            'col2': 'product_width_cm',
            'expected': 'CORRELATION or INFORMATION_DEPENDENCY'
        },
        # Date/time relationships
        {
            'table1': 'olist_orders_dataset',
            'col1': 'order_purchase_timestamp',
            'table2': 'olist_orders_dataset',
            'col2': 'order_approved_at',
            'expected': 'SIMILAR_VALUES or SAME_DOMAIN'
        },
        # Price relationships
        {
            'table1': 'olist_order_items_dataset',
            'col1': 'price',
            'table2': 'olist_order_items_dataset',
            'col2': 'freight_value',
            'expected': 'WEAK_RELATIONSHIP or INFORMATION_DEPENDENCY'
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting: {test['table1']}.{test['col1']} ↔ {test['table2']}.{test['col2']}")
        print(f"Expected: {test['expected']}")
        
        df1 = tables[test['table1']]
        df2 = tables[test['table2']]
        
        result = detector.detect_relationships(
            df1, df2, 
            test['col1'], test['col2'],
            test['table1'], test['table2']
        )
        
        print(f"Detected: {result['relationship_type']} (confidence: {result['confidence']:.2%})")
        
        # Show key features
        if 'features' in result:
            print("Key Features:")
            important_features = [
                'value_overlap_ratio', 'name_similarity', 'mutual_information',
                'pearson_correlation', 'spearman_correlation', 'fk_pattern_match',
                'same_dtype', 'both_numeric', 'both_categorical'
            ]
            for feature in important_features:
                if feature in result['features'] and result['features'][feature] > 0:
                    print(f"  - {feature}: {result['features'][feature]:.3f}")
        
        print("-" * 80)
    
    # Now build a small knowledge graph to demonstrate
    print("\n3. Building Enhanced Knowledge Graph (subset for speed)...")
    
    # Use only 3 tables for faster processing
    small_tables = {
        'olist_customers_dataset': tables['olist_customers_dataset'].sample(1000),  # Sample for speed
        'olist_orders_dataset': tables['olist_orders_dataset'].sample(1000),
        'olist_products_dataset': tables['olist_products_dataset'].sample(1000)
    }
    
    kg_builder = EnhancedKnowledgeGraphBuilder()
    graph = kg_builder.add_dataset(small_tables, "brazilian_ecommerce_sample")
    
    # Get summary
    summary = kg_builder.get_relationship_summary()
    
    print("\n4. Enhanced Graph Summary:")
    print(f"   Total Relationships: {sum(summary['relationship_types'].values())}")
    print("\n   Relationship Type Breakdown:")
    for rel_type, count in sorted(summary['relationship_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"   - {rel_type}: {count}")
    
    # Show top relationships
    print("\n5. Top 10 Relationships by Confidence:")
    for i, rel in enumerate(summary['top_relationships'][:10], 1):
        source = rel['source'].split('.')[-1]
        target = rel['target'].split('.')[-1] 
        source_table = rel['source'].split('.')[1]
        target_table = rel['target'].split('.')[1]
        print(f"   {i}. {source_table}.{source} → {target_table}.{target}")
        print(f"      Type: {rel['type']}, Confidence: {rel['weight']:.2%}")
    
    return kg_builder, detector

if __name__ == "__main__":
    kg_builder, detector = asyncio.run(test_ml_relationships_fast())