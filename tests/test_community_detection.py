#!/usr/bin/env python3
"""
Test script for Community Detection Integration

This script demonstrates the community detection working with semantic table graph
including table communities, column communities, and intelligent query routing.
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
    from src.knowledge_graph.community_detection import GraphCommunityDetector, CommunityAwareQueryRouter
    from src.knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required dependencies are installed")
    sys.exit(1)


def test_community_detection_integration():
    """Test community detection with semantic table graph"""
    logger.info("🧠 Testing Community Detection Integration")
    
    # Load sample ecommerce data
    data_path = Path("data/raw/ecommerce")
    tables_data = {}
    
    # Load available tables
    csv_files = list(data_path.glob("*.csv")) if data_path.exists() else []
    
    if csv_files:
        logger.info(f"Loading {len(csv_files)} CSV files...")
        for csv_file in csv_files[:9]:  # Load more files for community detection
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
    
    # Create enhanced knowledge graph
    kg_builder = EnhancedKnowledgeGraphBuilder()
    knowledge_graph = kg_builder.add_dataset(tables_data, "ecommerce")
    
    # Initialize table intelligence
    table_intelligence = TableIntelligenceLayer(
        model_name='all-MiniLM-L6-v2',
        enable_profiling=False,
        cache_embeddings=True
    )
    
    # Build semantic table graph
    graph_builder = SemanticTableGraphBuilder(knowledge_graph, table_intelligence)
    semantic_graph = graph_builder.build_table_graph(tables_data)
    
    # Get table profiles for community detection
    table_profiles = graph_builder.table_profiles
    
    # Initialize community detector
    logger.info("🏘️ Detecting communities...")
    community_detector = GraphCommunityDetector(semantic_graph, knowledge_graph)
    community_results = community_detector.detect_all_communities(table_profiles)
    
    # Initialize query router
    query_router = CommunityAwareQueryRouter(community_detector)
    
    # Display comprehensive analysis
    print_community_analysis(community_results, community_detector, query_router)
    
    # Test query routing scenarios
    test_query_routing_scenarios(query_router)


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
    
    # Sample sellers table
    sellers = pd.DataFrame({
        'seller_id': ['S001', 'S002', 'S003', 'S004'],
        'seller_name': ['TechStore', 'GadgetWorld', 'ElectroHub', 'DeviceDepo'],
        'seller_city': ['San Francisco', 'Austin', 'Seattle', 'Denver'],
        'seller_state': ['CA', 'TX', 'WA', 'CO']
    })
    
    return {
        'customers': customers,
        'orders': orders,
        'products': products,
        'order_items': order_items,
        'sellers': sellers
    }


def print_community_analysis(results: Dict, detector: GraphCommunityDetector, 
                           router: CommunityAwareQueryRouter):
    """Print comprehensive community analysis"""
    
    print("\\n" + "="*80)
    print("🏘️ COMMUNITY DETECTION ANALYSIS")
    print("="*80)
    
    # Community summary
    summary = results.get('summary', {})
    
    print(f"\\n📊 Community Summary:")
    table_summary = summary.get('table_communities', {})
    print(f"  • Table Communities: {table_summary.get('count', 0)}")
    print(f"  • Avg Table Community Size: {table_summary.get('avg_size', 0):.1f}")
    print(f"  • Avg Cohesion Score: {table_summary.get('avg_cohesion', 0):.3f}")
    
    column_summary = summary.get('column_communities', {})
    print(f"  • Column Communities: {column_summary.get('count', 0)}")
    print(f"  • Avg Column Community Size: {column_summary.get('avg_size', 0):.1f}")
    
    # Table communities analysis
    table_communities = results.get('table_communities', {})
    if table_communities:
        print(f"\\n🏢 Table Communities ({len(table_communities)}):")
        
        for community_id, community in table_communities.items():
            print(f"\\n  📋 {community.community_name} (ID: {community_id})")
            print(f"    - Type: {community.community_type}")
            print(f"    - Domain: {community.dominant_domain}")
            print(f"    - Tables: {', '.join(sorted(community.tables))}")
            print(f"    - Cohesion: {community.cohesion_score:.3f}")
            print(f"    - Central Table: {community.central_table}")
            
            if community.key_concepts:
                print(f"    - Key Concepts: {', '.join(community.key_concepts[:5])}")
            
            print(f"    - Description: {community.description}")
    
    # Column communities analysis
    column_communities = results.get('column_communities', {})
    if column_communities:
        print(f"\\n🔗 Column Communities:")
        
        # Intra-table communities
        intra_communities = column_communities.get('intra_table', {})
        if intra_communities:
            print(f"\\n  📝 Intra-Table Communities ({len(intra_communities)}):")
            for community_id, community in intra_communities.items():
                print(f"    • {community.community_name}")
                print(f"      - Role: {community.semantic_role}")
                print(f"      - Columns: {len(community.columns)}")
                print(f"      - Table: {community.representative_table}")
                print(f"      - Cohesion: {community.cohesion_score:.3f}")
        
        # Cross-table communities  
        cross_communities = column_communities.get('cross_table', {})
        if cross_communities:
            print(f"\\n  🌐 Cross-Table Communities ({len(cross_communities)}):")
            for community_id, community in cross_communities.items():
                tables_involved = set()
                for col in community.columns:
                    table = col.split('.')[1] if '.' in col else 'unknown'
                    tables_involved.add(table)
                
                print(f"    • {community.community_name}")
                print(f"      - Semantic Role: {community.semantic_role}")
                print(f"      - Columns: {len(community.columns)}")
                print(f"      - Tables Involved: {', '.join(sorted(tables_involved))}")
                print(f"      - Cohesion: {community.cohesion_score:.3f}")
    
    # Community relationships
    print(f"\\n🔄 Community Relationships:")
    for community_id, community in table_communities.items():
        if len(community.tables) > 0:
            sample_table = list(community.tables)[0]
            related = detector.get_related_communities(sample_table, max_communities=2)
            
            if related:
                related_names = []
                for rel_id, similarity in related:
                    rel_community = table_communities.get(rel_id)
                    if rel_community:
                        related_names.append(f"{rel_community.community_name} ({similarity:.2f})")
                
                if related_names:
                    print(f"  • {community.community_name} → {', '.join(related_names)}")


def test_query_routing_scenarios(router: CommunityAwareQueryRouter):
    """Test various query routing scenarios"""
    
    print(f"\\n" + "="*80)
    print("🎯 QUERY ROUTING SCENARIOS")
    print("="*80)
    
    # Test scenarios
    scenarios = [
        {
            'query': 'Show customer purchase patterns by product category',
            'intent': {
                'action_type': 'aggregation',
                'target_concepts': ['customer', 'purchase', 'product', 'category']
            }
        },
        {
            'query': 'Find orders with high shipping costs',
            'intent': {
                'action_type': 'filtering', 
                'target_concepts': ['order', 'shipping', 'cost']
            }
        },
        {
            'query': 'Analyze seller performance across regions',
            'intent': {
                'action_type': 'trend_analysis',
                'target_concepts': ['seller', 'performance', 'region']
            }
        },
        {
            'query': 'Join customer data with order history',
            'intent': {
                'action_type': 'joining',
                'target_concepts': ['customer', 'order', 'history']
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\\n🔍 Scenario {i}: {scenario['query']}")
        
        # Route query to communities
        community_scores = router.route_query_to_communities(
            scenario['intent'], 
            scenario['intent']['target_concepts']
        )
        
        print(f"  📊 Community Routing Results:")
        
        # Sort communities by score
        sorted_communities = sorted(community_scores.items(), key=lambda x: x[1], reverse=True)
        
        for community_id, score in sorted_communities[:3]:  # Top 3 communities
            if score > 0.05:  # Only show meaningful scores
                community_info = router.get_community_info(community_id)
                if community_info:
                    print(f"    • {community_info['community_name']}: {score:.3f}")
                    print(f"      - Domain: {community_info['dominant_domain']}")
                    print(f"      - Tables: {', '.join(community_info['tables'][:3])}")
                    print(f"      - Concepts: {', '.join(community_info['key_concepts'][:3])}")
        
        # Get suggestions for top community
        if sorted_communities:
            top_community_id = sorted_communities[0][0]
            suggestions = router.suggest_related_communities(top_community_id, max_suggestions=2)
            
            if suggestions:
                print(f"  💡 Related Communities:")
                for suggestion in suggestions:
                    print(f"    • {suggestion['community_name']}: {suggestion['relatedness_score']:.3f}")
                    print(f"      - Reason: {suggestion['relationship_reason']}")
    
    print(f"\\n✅ Query routing scenarios completed!")


if __name__ == "__main__":
    logger.info("🚀 Starting Community Detection Integration Tests")
    
    try:
        test_community_detection_integration()
        logger.info("✅ Community Detection testing complete!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()