#!/usr/bin/env python3
"""
Test script for the Semantic Table Graph Builder

This script demonstrates the enhanced table relationship detection capabilities
including structural, semantic, temporal, and business process relationships.
"""

import sys
import pandas as pd
from pathlib import Path
import logging
import networkx as nx
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.knowledge_graph.semantic_table_graph import SemanticTableGraphBuilder
    from src.knowledge_graph.table_intelligence import TableIntelligenceLayer
    from src.knowledge_graph.table_relationships import TableRelationshipType
    from src.knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required dependencies are installed")
    sys.exit(1)


def test_semantic_table_graph():
    """Test the semantic table graph builder with sample data"""
    logger.info("🧠 Testing Semantic Table Graph Builder")
    
    # Load sample ecommerce data
    data_path = Path("data/raw/ecommerce")
    tables_data = {}
    
    # Load available tables
    csv_files = list(data_path.glob("*.csv")) if data_path.exists() else []
    
    if csv_files:
        logger.info(f"Loading {len(csv_files)} CSV files...")
        for csv_file in csv_files[:9]:  # Limit to first 5 files for testing
            table_name = csv_file.stem.replace('olist_', '').replace('_dataset', '')
            try:
                df = pd.read_csv(csv_file)
                tables_data[table_name] = df
                logger.info(f"  - Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"  - Failed to load {csv_file}: {e}")
    else:
        logger.info("No real data found, creating sample data...")
        tables_data = create_sample_tables()
    
    if not tables_data:
        logger.error("No data available for testing")
        return
    
    # Initialize components
    logger.info("🔧 Initializing components...")
    
    # Create enhanced knowledge graph from existing data
    kg_builder = EnhancedKnowledgeGraphBuilder()
    knowledge_graph = kg_builder.add_dataset(tables_data, "ecommerce")
    
    # Initialize table intelligence
    table_intelligence = TableIntelligenceLayer(
        model_name='all-MiniLM-L6-v2',
        enable_profiling=False,  # Disable for faster testing
        cache_embeddings=True
    )
    
    # Initialize semantic table graph builder
    graph_builder = SemanticTableGraphBuilder(knowledge_graph, table_intelligence)
    
    # Build the semantic table graph
    logger.info("🏗️ Building semantic table graph...")
    semantic_graph = graph_builder.build_table_graph(tables_data)
    
    # Display results
    print_graph_analysis(semantic_graph, graph_builder)


def create_sample_tables() -> Dict[str, pd.DataFrame]:
    """Create sample tables for testing if real data is not available"""
    
    # Sample customers table
    customers = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'customer_city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'customer_state': ['NY', 'CA', 'IL', 'TX', 'AZ'],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-01-30', '2023-03-10', '2023-02-05']
    })
    
    # Sample orders table
    orders = pd.DataFrame({
        'order_id': ['O001', 'O002', 'O003', 'O004', 'O005'],
        'customer_id': ['C001', 'C002', 'C001', 'C003', 'C002'],
        'order_date': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-03', '2023-06-04'],
        'order_status': ['delivered', 'processing', 'delivered', 'shipped', 'delivered'],
        'order_amount': [125.50, 89.25, 210.75, 45.00, 180.30]
    })
    
    # Sample products table
    products = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
        'product_category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
        'product_price': [999.99, 29.99, 79.99, 299.99, 59.99],
        'stock_quantity': [10, 50, 25, 15, 30]
    })
    
    # Sample order items table (junction table)
    order_items = pd.DataFrame({
        'order_id': ['O001', 'O001', 'O002', 'O003', 'O004'],
        'product_id': ['P001', 'P002', 'P002', 'P001', 'P003'],
        'quantity': [1, 2, 1, 1, 1],
        'unit_price': [999.99, 29.99, 29.99, 999.99, 79.99]
    })
    
    # Sample reviews table (supplementary)
    reviews = pd.DataFrame({
        'review_id': ['R001', 'R002', 'R003', 'R004'],
        'order_id': ['O001', 'O002', 'O003', 'O001'],
        'product_id': ['P001', 'P002', 'P001', 'P002'],
        'rating': [5, 4, 5, 3],
        'review_date': ['2023-06-10', '2023-06-12', '2023-06-15', '2023-06-11']
    })
    
    return {
        'customers': customers,
        'orders': orders,
        'products': products,
        'order_items': order_items,
        'reviews': reviews
    }


def print_graph_analysis(graph: nx.MultiDiGraph, builder: SemanticTableGraphBuilder):
    """Print comprehensive analysis of the semantic table graph"""
    
    print("\\n" + "="*80)
    print("🎯 SEMANTIC TABLE GRAPH ANALYSIS")
    print("="*80)
    
    # Basic graph statistics
    print(f"\\n📊 Graph Statistics:")
    print(f"  • Tables (nodes): {graph.number_of_nodes()}")
    print(f"  • Relationships (edges): {graph.number_of_edges()}")
    print(f"  • Graph density: {nx.density(graph):.3f}")
    print(f"  • Is connected: {nx.is_weakly_connected(graph)}")
    
    # Export comprehensive summary
    summary = builder.export_graph_summary()
    
    # Relationship types
    print(f"\\n🔗 Relationship Types:")
    for rel_type, count in summary['relationship_types'].items():
        print(f"  • {rel_type}: {count}")
    
    # Business domains
    print(f"\\n🏢 Business Domains:")
    for domain, count in summary['business_domains'].items():
        print(f"  • {domain}: {count} tables")
    
    # Table types
    print(f"\\n📋 Table Types:")
    for table_type, count in summary['table_types'].items():
        print(f"  • {table_type}: {count} tables")
    
    # Most important tables
    print(f"\\n⭐ Most Important Tables:")
    for i, table_info in enumerate(summary['most_important_tables'], 1):
        print(f"  {i}. {table_info['table']} "
              f"(importance: {table_info['importance_score']:.3f}, "
              f"domain: {table_info['business_domain']}, "
              f"type: {table_info['table_type']})")
    
    # Detection methods
    print(f"\\n🔍 Detection Methods:")
    for method, count in summary['detection_methods'].items():
        print(f"  • {method}: {count} relationships")
    
    # Detailed relationship analysis
    print(f"\\n🔗 Detailed Relationships:")
    for source, target, data in graph.edges(data=True):
        rel_type = data.get('relationship_type', 'unknown')
        confidence = data.get('confidence', 0.0)
        description = data.get('semantic_description', 'No description')
        detection_method = data.get('detection_method', 'unknown')
        
        # Handle enum types
        if hasattr(rel_type, 'value'):
            rel_type_str = rel_type.value
        else:
            rel_type_str = str(rel_type)
        
        print(f"  • {source} → {target}")
        print(f"    - Type: {rel_type_str}")
        print(f"    - Confidence: {confidence:.3f}")
        print(f"    - Method: {detection_method}")
        print(f"    - Description: {description}")
        print()
    
    # Individual table analysis
    print(f"\\n🏷️ Individual Table Analysis:")
    for table_name in graph.nodes():
        node_data = graph.nodes[table_name]
        relationships = builder.get_table_relationships(table_name)
        
        print(f"\\n  📊 {table_name}:")
        print(f"    - Domain: {node_data.get('business_domain', 'unknown')}")
        print(f"    - Type: {node_data.get('table_type', 'unknown')}")
        print(f"    - Rows: {node_data.get('row_count', 0):,}")
        print(f"    - Columns: {node_data.get('column_count', 0)}")
        print(f"    - Quality Score: {node_data.get('data_quality_score', 0):.3f}")
        print(f"    - Importance: {node_data.get('importance_score', 0):.3f}")
        print(f"    - Relationships: {len(relationships)}")
        
        # Show key concepts
        key_concepts = node_data.get('key_concepts', [])
        if key_concepts:
            print(f"    - Key Concepts: {', '.join(key_concepts[:5])}")
        
        # Show column categories
        measures = node_data.get('measure_columns', [])
        dimensions = node_data.get('dimension_columns', [])
        identifiers = node_data.get('identifier_columns', [])
        
        if measures:
            print(f"    - Measures: {', '.join(measures[:3])}")
        if dimensions:
            print(f"    - Dimensions: {', '.join(dimensions[:3])}")
        if identifiers:
            print(f"    - Identifiers: {', '.join(identifiers[:3])}")


if __name__ == "__main__":
    logger.info("🚀 Starting Semantic Table Graph Tests")
    
    try:
        test_semantic_table_graph()
        logger.info("✅ Semantic Table Graph testing complete!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()