import asyncio
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data.connectors.csv_connector import CSVConnector
from src.knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder

async def demo_ml_relationships():
    """Demo ML-powered relationship detection with visualization"""
    
    # Load data
    print("Loading dataset...")
    connector = CSVConnector()
    tables = await connector.load_data({
        'data_path': 'data/raw/ecommerce'
    })
    
    # For demo, use a subset of tables
    demo_tables = {
        'customers': tables['olist_customers_dataset'].sample(5000),
        'orders': tables['olist_orders_dataset'].sample(5000),
        'order_items': tables['olist_order_items_dataset'].sample(5000),
        'products': tables['olist_products_dataset'].sample(5000)
    }
    
    # Build enhanced knowledge graph
    print("\nBuilding enhanced knowledge graph...")
    kg_builder = EnhancedKnowledgeGraphBuilder()
    graph = kg_builder.add_dataset(demo_tables, "ecommerce")
    
    # Get summary
    summary = kg_builder.get_relationship_summary()
    
    # Create visualization
    print("\nGenerating visualization...")
    
    # 1. Relationship Type Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart of relationship types
    rel_types = list(summary['relationship_types'].keys())
    rel_counts = list(summary['relationship_types'].values())
    
    colors = {
        'FOREIGN_KEY': '#e74c3c',
        'SAME_DOMAIN': '#9b59b6', 
        'SIMILAR_VALUES': '#16a085',
        'INFORMATION_DEPENDENCY': '#f39c12',
        'POSITIVELY_CORRELATED': '#27ae60',
        'NEGATIVELY_CORRELATED': '#e67e22',
        'WEAK_RELATIONSHIP': '#95a5a6'
    }
    
    pie_colors = [colors.get(rt, '#7f8c8d') for rt in rel_types]
    
    ax1.pie(rel_counts, labels=rel_types, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribution of Relationship Types', fontsize=14, weight='bold')
    
    # Bar chart of top relationships
    top_rels = summary['top_relationships'][:15]
    
    rel_labels = []
    rel_weights = []
    rel_colors_bar = []
    
    for rel in top_rels:
        source = rel['source'].split('.')[-1]
        target = rel['target'].split('.')[-1]
        source_table = rel['source'].split('.')[1] if len(rel['source'].split('.')) > 1 else 'unknown'
        target_table = rel['target'].split('.')[1] if len(rel['target'].split('.')) > 1 else 'unknown'
        
        label = f"{source_table}.{source}\n→ {target_table}.{target}"
        rel_labels.append(label)
        rel_weights.append(rel['weight'])
        rel_colors_bar.append(colors.get(rel['type'], '#7f8c8d'))
    
    bars = ax2.barh(rel_labels, rel_weights, color=rel_colors_bar)
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_title('Top 15 Relationships by Confidence', fontsize=14, weight='bold')
    ax2.set_xlim(0, max(rel_weights) * 1.1)
    
    # Add value labels on bars
    for bar, weight in zip(bars, rel_weights):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{weight:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ml_relationships_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Enhanced graph visualization
    kg_builder.visualize_enhanced_graph(max_nodes=60, min_confidence=0.7)
    
    # Print detailed summary
    print("\n" + "="*80)
    print("ML-POWERED RELATIONSHIP DETECTION SUMMARY")
    print("="*80)
    
    print(f"\nDataset: Brazilian E-commerce (sampled)")
    print(f"Tables analyzed: {len(demo_tables)}")
    print(f"Total relationships discovered: {sum(summary['relationship_types'].values())}")
    
    print("\nRelationship Type Breakdown:")
    for rel_type, count in sorted(summary['relationship_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type:.<30} {count:>3} relationships")
    
    print("\nKey Insights:")
    print("  ✓ ML detection found multiple relationship types beyond simple FK")
    print("  ✓ Information dependencies reveal hidden connections between columns")
    print("  ✓ Similar values detection identifies semantically related columns")
    print("  ✓ Correlation analysis works across numeric columns")
    
    print("\nTop 5 Strongest Relationships:")
    for i, rel in enumerate(summary['top_relationships'][:5], 1):
        source = rel['source'].split('.')
        target = rel['target'].split('.')
        s_desc = f"{source[1]}.{source[2]}" if len(source) > 2 else rel['source']
        t_desc = f"{target[1]}.{target[2]}" if len(target) > 2 else rel['target']
        
        print(f"  {i}. {s_desc} → {t_desc}")
        print(f"     Type: {rel['type']}, Confidence: {rel['weight']:.1%}")
    
    return kg_builder

if __name__ == "__main__":
    kg_builder = asyncio.run(demo_ml_relationships())