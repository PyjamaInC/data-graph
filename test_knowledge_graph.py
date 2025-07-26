import asyncio
import sys
sys.path.append('.')

from src.data.connectors.csv_connector import CSVConnector
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

async def test_knowledge_graph():
    """Test building knowledge graph from e-commerce dataset"""
    
    # Load data
    print("1. Loading dataset...")
    connector = CSVConnector()
    tables = await connector.load_data({
        'data_path': 'data/raw/ecommerce'
    })
    
    # Build knowledge graph
    print("\n2. Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder()
    graph = kg_builder.add_dataset(tables, "brazilian_ecommerce")
    
    # Print graph statistics
    print("\n3. Graph Statistics:")
    print(f"   Total Nodes: {graph.number_of_nodes()}")
    print(f"   Total Edges: {graph.number_of_edges()}")
    
    # Node breakdown
    node_types = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\n   Node Types:")
    for node_type, count in node_types.items():
        print(f"     - {node_type}: {count}")
    
    # Edge breakdown
    edge_types = {}
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('relationship', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\n   Relationship Types:")
    for edge_type, count in edge_types.items():
        print(f"     - {edge_type}: {count}")
    
    # Find relationships
    print("\n4. Discovered Relationships:")
    relationships = kg_builder.get_relationships()
    
    # Filter for foreign key relationships
    fk_relationships = [r for r in relationships if r['relationship'] == 'FOREIGN_KEY']
    print(f"\n   Foreign Key Relationships: {len(fk_relationships)}")
    for rel in fk_relationships[:10]:  # Show first 10
        source_name = rel['source'].split('.')[-1]
        target_name = rel['target'].split('.')[-1]
        source_table = rel['source'].split('.')[1]
        target_table = rel['target'].split('.')[1]
        print(f"     - {source_table}.{source_name} -> {target_table}.{target_name} (confidence: {rel['weight']:.2f})")
    
    # Filter for correlations
    correlations = [r for r in relationships if 'CORRELATED' in r['relationship']]
    print(f"\n   Correlation Relationships: {len(correlations)}")
    for rel in correlations[:10]:  # Show first 10
        source_name = rel['source'].split('.')[-1]
        target_name = rel['target'].split('.')[-1]
        table = rel['source'].split('.')[1]
        print(f"     - {table}: {source_name} <-> {target_name} ({rel['relationship']}, strength: {rel['weight']:.2f})")
    
    # Example: Find related columns for a specific column
    print("\n5. Example: Finding related columns...")
    test_column = "COLUMN:brazilian_ecommerce.olist_orders_dataset.customer_id"
    if test_column in graph:
        related = kg_builder.find_related_columns(test_column, max_distance=3)
        print(f"\n   Columns related to 'customer_id' in orders table:")
        for rel in related[:10]:  # Show first 10
            col_name = rel['column'].split('.')[-1]
            table_name = rel['column'].split('.')[1]
            print(f"     - {table_name}.{col_name} (distance: {rel['distance']}, weight: {rel['weight']:.3f})")
    
    # Visualize a subset of the graph
    print("\n6. Generating graph visualization...")
    kg_builder.visualize_graph(max_nodes=50)
    
    return kg_builder

if __name__ == "__main__":
    kg_builder = asyncio.run(test_knowledge_graph())