"""
Test the schema-driven query planner with knowledge graph traversal
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from schema.schema_manager import SchemaManager
from knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
from agents.schema_driven_query_planner import SchemaDrivenQueryPlanner

# Set OpenAI API key if available
if 'OPENAI_API_KEY' not in os.environ:
    print("⚠️  Warning: OPENAI_API_KEY not set. LLM analysis will be limited.")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")

def test_schema_driven_planning():
    """Test the schema-driven query planner"""
    
    print("🚀 Testing Schema-Driven Query Planner")
    print("=" * 70)
    
    # Load Brazilian e-commerce data
    data_dir = Path("data/raw/ecommerce")
    datasets = {}
    
    print("\n📁 Loading data...")
    files_to_load = [
        'olist_customers_dataset.csv',
        'olist_orders_dataset.csv', 
        'olist_order_items_dataset.csv',
        'olist_products_dataset.csv',
        'olist_sellers_dataset.csv'
    ]
    
    for filename in files_to_load:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, nrows=2000)  # Limit for testing
            table_name = filename.replace('olist_', '').replace('_dataset.csv', '')
            datasets[table_name] = df
            print(f"  ✓ Loaded {table_name}: {df.shape}")
    
    # Step 1: Discover schema
    print("\n📊 Discovering schema...")
    schema_manager = SchemaManager()
    schema_manager.auto_discovery.use_profiling = False  # Faster for demo
    schema = schema_manager.discover_schema_from_data(datasets, "ecommerce")
    
    print(f"  ✓ Schema discovered with {len(schema.tables)} tables")
    
    # Step 2: Build knowledge graph
    print("\n🕸️  Building knowledge graph...")
    kg_builder = EnhancedKnowledgeGraphBuilder()
    kg_builder.add_dataset(datasets, "ecommerce")
    knowledge_graph = kg_builder.graph
    
    print(f"  ✓ Knowledge graph built:")
    print(f"    - Nodes: {knowledge_graph.number_of_nodes()}")
    print(f"    - Edges: {knowledge_graph.number_of_edges()}")
    
    # Show some discovered relationships
    relationships = kg_builder.get_relationships()
    print(f"    - Relationships: {len(relationships)}")
    
    # Step 3: Initialize query planner
    print("\n🤖 Initializing Query Planner...")
    
    try:
        planner = SchemaDrivenQueryPlanner(schema, knowledge_graph)
        print("  ✓ Query planner ready")
    except Exception as e:
        print(f"  ✗ Error initializing planner: {e}")
        return
    
    # Step 4: Test various queries
    print("\n" + "="*70)
    print("📝 Testing Query Planning")
    print("="*70)
    
    test_queries = [
        "Show me sales analysis by different factors",
        "What products are customers buying the most?",
        "Compare order values across different seller locations",
        "How are orders distributed over time?",
        "Which sellers have the highest revenue?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print("="*70)
        
        try:
            # Plan the query
            plan = planner.plan_query(query)
            
            # Display results
            print("\n📊 Query Understanding:")
            understanding = plan.get('understanding', {})
            print(f"  • Analysis Type: {understanding.get('analysis_type', 'N/A')}")
            print(f"  • Concepts: {', '.join(understanding.get('concepts', []))}")
            print(f"  • Needs Relationships: {understanding.get('relationships_needed', False)}")
            
            print("\n📋 Selected Columns:")
            for col_info in plan['selected_columns'][:8]:  # Show max 8
                print(f"  • {col_info['column']} ({col_info['type']}) - Role: {col_info['role']}")
            
            if len(plan['selected_columns']) > 8:
                print(f"  ... and {len(plan['selected_columns']) - 8} more columns")
            
            if plan['aggregations']:
                print("\n📊 Aggregations:")
                for agg in plan['aggregations'][:5]:
                    print(f"  • {agg['column']}: {', '.join(agg['methods'])}")
            
            if plan['group_by']:
                print("\n📈 Group By:")
                for col in plan['group_by'][:5]:
                    print(f"  • {col}")
            
            if plan['joins']:
                print("\n🔗 Required Joins:")
                for join in plan['joins'][:3]:
                    print(f"  • {join['from']} → {join['to']} ({join['type']})")
            
            print("\n💡 Explanation:")
            print(plan['explanation'])
            
        except Exception as e:
            print(f"\n❌ Error planning query: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 5: Demonstrate knowledge graph traversal
    print("\n" + "="*70)
    print("🔍 Knowledge Graph Traversal Example")
    print("="*70)
    
    # Find a specific column in the graph
    product_price_node = None
    customer_city_node = None
    
    for node in knowledge_graph.nodes():
        if 'product' in node.lower() and 'price' in node.lower():
            product_price_node = node
        elif 'customer' in node.lower() and 'city' in node.lower():
            customer_city_node = node
    
    if product_price_node and customer_city_node:
        print(f"\nFinding path from product price to customer location:")
        print(f"  Start: {product_price_node}")
        print(f"  End: {customer_city_node}")
        
        try:
            # Find shortest path
            path = nx.shortest_path(knowledge_graph, product_price_node, customer_city_node)
            print(f"\n  Path found (length {len(path)-1}):")
            for i, node in enumerate(path):
                if i > 0:
                    # Get edge info
                    edge_data = knowledge_graph.get_edge_data(path[i-1], node)
                    if edge_data:
                        rel_type = "unknown"
                        if isinstance(edge_data, dict) and 0 in edge_data:
                            rel_type = edge_data[0].get('relationship', 'unknown')
                        print(f"    ↓ ({rel_type})")
                print(f"  {i+1}. {node}")
        except nx.NetworkXNoPath:
            print("  No path found between these columns")
    
    print("\n" + "="*70)
    print("✅ Schema-Driven Query Planning Test Complete!")
    print("="*70)

if __name__ == "__main__":
    test_schema_driven_planning()