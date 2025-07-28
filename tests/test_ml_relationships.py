import asyncio
import sys
sys.path.append('.')

from src.data.connectors.csv_connector import CSVConnector
from src.knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
import pandas as pd

async def test_ml_relationships():
    """Test ML-powered relationship detection"""
    
    # Load data
    print("1. Loading dataset...")
    connector = CSVConnector()
    tables = await connector.load_data({
        'data_path': 'data/raw/ecommerce'
    })
    
    # For faster testing, let's use a subset of tables
    test_tables = {
        'olist_customers_dataset': tables['olist_customers_dataset'],
        'olist_orders_dataset': tables['olist_orders_dataset'],
        'olist_order_items_dataset': tables['olist_order_items_dataset'],
        'olist_products_dataset': tables['olist_products_dataset']
    }
    
    # Build enhanced knowledge graph
    print("\n2. Building enhanced knowledge graph with ML detection...")
    print("   (This may take a few minutes as it analyzes all column pairs)")
    kg_builder = EnhancedKnowledgeGraphBuilder()
    graph = kg_builder.add_dataset(test_tables, "brazilian_ecommerce")
    
    # Get relationship summary
    print("\n3. Relationship Summary:")
    summary = kg_builder.get_relationship_summary()
    
    print(f"\n   Graph Statistics:")
    print(f"   - Total Nodes: {summary['total_nodes']}")
    print(f"   - Total Edges: {summary['total_edges']}")
    
    print(f"\n   Discovered Relationship Types:")
    for rel_type, count in summary['relationship_types'].items():
        print(f"   - {rel_type}: {count}")
    
    # Show examples of each relationship type
    print("\n4. Examples of Each Relationship Type:")
    
    for rel_type in summary['relationship_types'].keys():
        print(f"\n   {rel_type}:")
        relationships = kg_builder.get_relationships_by_type(rel_type)
        
        for rel in relationships[:3]:  # Show top 3 examples
            source_parts = rel['source'].split('.')
            target_parts = rel['target'].split('.')
            source_desc = f"{source_parts[1]}.{source_parts[2]}" if len(source_parts) > 2 else rel['source']
            target_desc = f"{target_parts[1]}.{target_parts[2]}" if len(target_parts) > 2 else rel['target']
            
            print(f"   - {source_desc} → {target_desc}")
            print(f"     Confidence: {rel['weight']:.2%}")
            if 'key_features' in rel['evidence']:
                print(f"     Evidence: {rel['evidence']['key_features']}")
    
    # Analyze specific interesting relationships
    print("\n5. Detailed Analysis of Select Relationships:")
    
    # Check for categorical associations
    print("\n   Categorical Relationships:")
    categorical_rels = [r for r in kg_builder.get_relationships() 
                       if r['relationship'] in ['SAME_DOMAIN', 'SIMILAR_VALUES', 'INFORMATION_DEPENDENCY']]
    
    for rel in categorical_rels[:5]:
        source_parts = rel['source'].split('.')
        target_parts = rel['target'].split('.')
        if len(source_parts) > 2 and len(target_parts) > 2:
            print(f"   - {source_parts[1]}.{source_parts[2]} ↔ {target_parts[1]}.{target_parts[2]}")
            print(f"     Type: {rel['relationship']}, Confidence: {rel['weight']:.2%}")
    
    # Generate enhanced visualization
    print("\n6. Generating enhanced visualization...")
    kg_builder.visualize_enhanced_graph(max_nodes=80, min_confidence=0.6)
    
    return kg_builder

if __name__ == "__main__":
    kg_builder = asyncio.run(test_ml_relationships())