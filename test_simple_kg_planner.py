"""
Simple test for knowledge graph traversal without LLM dependencies
"""

import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from schema.schema_manager import SchemaManager
from knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
from agents.schema_driven_query_planner import KnowledgeGraphTraverser

def test_knowledge_graph_traversal():
    """Test basic knowledge graph traversal capabilities"""
    
    print("🚀 Testing Knowledge Graph Traversal")
    print("=" * 70)
    
    # Load sample data
    data_dir = Path("data/raw/ecommerce")
    datasets = {}
    
    print("\n📁 Loading data...")
    files = ['olist_customers_dataset.csv', 'olist_orders_dataset.csv', 'olist_order_items_dataset.csv']
    
    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, nrows=1000)
            table_name = filename.replace('olist_', '').replace('_dataset.csv', '')
            datasets[table_name] = df
            print(f"  ✓ Loaded {table_name}: {df.shape}")
    
    # Build knowledge graph
    print("\n🕸️  Building knowledge graph...")
    kg_builder = EnhancedKnowledgeGraphBuilder()
    kg_builder.add_dataset(datasets, "ecommerce")
    knowledge_graph = kg_builder.graph
    
    print(f"\n📊 Knowledge Graph Statistics:")
    print(f"  • Nodes: {knowledge_graph.number_of_nodes()}")
    print(f"  • Edges: {knowledge_graph.number_of_edges()}")
    
    # Get relationships
    relationships = kg_builder.get_relationships()
    print(f"  • Relationships: {len(relationships)}")
    
    # Show some relationships
    print("\n🔗 Sample Relationships:")
    for rel in relationships[:10]:
        print(f"  • {rel['source']} → {rel['target']}")
        print(f"    Type: {rel['relationship']}, Weight: {rel['weight']:.2f}")
    
    # Test graph traversal
    print("\n🔍 Testing Graph Traversal:")
    traverser = KnowledgeGraphTraverser(knowledge_graph)
    
    # Find some column nodes
    column_nodes = [node for node in knowledge_graph.nodes() if node.startswith('COLUMN:')]
    print(f"\nFound {len(column_nodes)} column nodes")
    
    if len(column_nodes) >= 2:
        # Find paths between first two columns
        start_node = column_nodes[0]
        end_node = column_nodes[10] if len(column_nodes) > 10 else column_nodes[-1]
        
        print(f"\nFinding paths from:")
        print(f"  Start: {start_node}")
        print(f"  End: {end_node}")
        
        try:
            paths = traverser.find_paths_between_concepts([start_node], [end_node], max_path_length=4)
            
            if paths:
                print(f"\nFound {len(paths)} paths:")
                for i, path in enumerate(paths[:3], 1):
                    print(f"\n  Path {i} (weight: {path.total_weight:.3f}):")
                    for j, node in enumerate(path.path):
                        print(f"    {j+1}. {node}")
                    print(f"  Tables involved: {path.get_tables_in_path()}")
            else:
                print("  No paths found")
        except Exception as e:
            print(f"  Error finding paths: {e}")
    
    # Test finding related columns
    print("\n📊 Testing Related Column Discovery:")
    if column_nodes:
        seed = column_nodes[0]
        print(f"\nFinding columns related to: {seed}")
        
        related = traverser.find_related_columns([seed], max_distance=2)
        
        if seed in related and related[seed]:
            print(f"\nFound {len(related[seed])} related columns:")
            for rel in related[seed][:5]:
                print(f"  • {rel['column']}")
                print(f"    Distance: {rel['distance']}, Strength: {rel['strength']:.3f}")
        else:
            print("  No related columns found")
    
    print("\n✅ Knowledge Graph Traversal Test Complete!")

if __name__ == "__main__":
    test_knowledge_graph_traversal()