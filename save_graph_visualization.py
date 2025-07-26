import asyncio
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import networkx as nx

from src.data.connectors.csv_connector import CSVConnector
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

async def save_graph_visualization():
    """Save knowledge graph visualization to file"""
    
    # Load data
    print("Loading dataset...")
    connector = CSVConnector()
    tables = await connector.load_data({
        'data_path': 'data/raw/ecommerce'
    })
    
    # Build knowledge graph
    print("Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder()
    graph = kg_builder.add_dataset(tables, "brazilian_ecommerce")
    
    # Create visualization
    plt.figure(figsize=(20, 15))
    
    # Select nodes to show (prioritize tables and foreign key relationships)
    # Show all dataset and table nodes
    nodes_to_show = [n for n in graph.nodes() if graph.nodes[n]['type'] in ['dataset', 'table']]
    
    # Add columns that have foreign key relationships
    for u, v, data in graph.edges(data=True):
        if data.get('relationship') == 'FOREIGN_KEY':
            nodes_to_show.extend([u, v])
    
    nodes_to_show = list(set(nodes_to_show))  # Remove duplicates
    subgraph = graph.subgraph(nodes_to_show)
    
    # Create layout with better spacing
    pos = nx.spring_layout(subgraph, k=5, iterations=100, seed=42)
    
    # Draw nodes by type
    node_types = {
        'dataset': {'color': '#ff6b6b', 'size': 1000},
        'table': {'color': '#4ecdc4', 'size': 800},
        'column': {'color': '#45b7d1', 'size': 600}
    }
    
    for node_type, style in node_types.items():
        nodes_of_type = [n for n in subgraph.nodes() 
                       if subgraph.nodes[n].get('type') == node_type]
        
        if nodes_of_type:
            nx.draw_networkx_nodes(subgraph, pos, 
                                 nodelist=nodes_of_type,
                                 node_color=style['color'],
                                 node_size=style['size'],
                                 alpha=0.8)
    
    # Draw edges by relationship type
    edge_styles = {
        'CONTAINS': {'color': 'gray', 'width': 1, 'style': 'dashed'},
        'HAS_COLUMN': {'color': 'lightgray', 'width': 1, 'style': 'dotted'},
        'FOREIGN_KEY': {'color': '#e74c3c', 'width': 3, 'style': 'solid'}
    }
    
    for relationship, style in edge_styles.items():
        edges_of_type = [(u, v) for u, v, d in subgraph.edges(data=True)
                       if d.get('relationship') == relationship]
        
        if edges_of_type:
            nx.draw_networkx_edges(subgraph, pos,
                                 edgelist=edges_of_type,
                                 edge_color=style['color'],
                                 width=style['width'],
                                 style=style['style'],
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=10)
    
    # Add labels
    labels = {}
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        if node_data.get('type') == 'column':
            # Show only column name for columns
            labels[node] = node_data.get('name', node.split('.')[-1])
        elif node_data.get('type') == 'table':
            # Shorten table names
            name = node_data.get('name', node)
            labels[node] = name.replace('olist_', '').replace('_dataset', '')
        else:
            labels[node] = node_data.get('name', node)
    
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=10, font_weight='bold')
    
    # Add title and legend
    plt.title("Brazilian E-commerce Knowledge Graph\nForeign Key Relationships", size=20, weight='bold')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', markersize=15, label='Dataset'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4', markersize=15, label='Table'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#45b7d1', markersize=15, label='Column'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='Foreign Key'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('knowledge_graph_visualization.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved to: knowledge_graph_visualization.png")
    
    # Also create a summary report
    print("\nGenerating summary report...")
    
    summary = f"""
Brazilian E-commerce Knowledge Graph Summary
============================================

Dataset Statistics:
- Tables: 9
- Total Columns: 52
- Total Relationships: {len([(u, v, d) for u, v, d in graph.edges(data=True) if d.get('relationship') not in ['CONTAINS', 'HAS_COLUMN']])}

Discovered Foreign Key Relationships:
"""
    
    for u, v, data in graph.edges(data=True):
        if data.get('relationship') == 'FOREIGN_KEY':
            source_table = u.split('.')[1].replace('olist_', '').replace('_dataset', '')
            source_col = u.split('.')[-1]
            target_table = v.split('.')[1].replace('olist_', '').replace('_dataset', '')
            target_col = v.split('.')[-1]
            confidence = data.get('weight', 0)
            summary += f"- {source_table}.{source_col} -> {target_table}.{target_col} (confidence: {confidence:.2%})\n"
    
    with open('knowledge_graph_summary.txt', 'w') as f:
        f.write(summary)
    
    print("Summary report saved to: knowledge_graph_summary.txt")
    
    return kg_builder

if __name__ == "__main__":
    kg_builder = asyncio.run(save_graph_visualization())