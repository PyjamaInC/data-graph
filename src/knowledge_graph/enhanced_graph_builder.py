import networkx as nx
import pandas as pd
from typing import Dict, List, Any

try:
    from ..data.models.data_models import Relationship
    from .graph_builder import KnowledgeGraphBuilder
    from .relationship_detector import MLRelationshipDetector
except ImportError:
    from data.models.data_models import Relationship
    from knowledge_graph.graph_builder import KnowledgeGraphBuilder
    from knowledge_graph.relationship_detector import MLRelationshipDetector

class EnhancedKnowledgeGraphBuilder(KnowledgeGraphBuilder):
    def __init__(self):
        super().__init__()
        self.ml_detector = MLRelationshipDetector()
    
    def _discover_inter_table_relationships(self, tables: Dict[str, pd.DataFrame], dataset_name: str):
        """Enhanced inter-table relationship discovery using ML"""
        
        table_names = list(tables.keys())
        
        for i, table1_name in enumerate(table_names):
            for table2_name in table_names[i+1:]:
                df1, df2 = tables[table1_name], tables[table2_name]
                
                print(f"Analyzing relationships between {table1_name} and {table2_name}...")
                
                # Analyze all column pairs
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        # Use ML detector
                        relationship_info = self.ml_detector.detect_relationships(
                            df1, df2, col1, col2, table1_name, table2_name
                        )
                        
                        # Add relationship if confidence is high enough
                        if relationship_info['confidence'] > 0.5:  # Threshold
                            source_node = f"COLUMN:{dataset_name}.{table1_name}.{col1}"
                            target_node = f"COLUMN:{dataset_name}.{table2_name}.{col2}"
                            
                            self.graph.add_edge(source_node, target_node,
                                              relationship=relationship_info['relationship_type'],
                                              weight=relationship_info['confidence'],
                                              evidence=relationship_info['evidence'],
                                              ml_features=relationship_info['features'])
    
    def _discover_intra_table_relationships(self, tables: Dict[str, pd.DataFrame], dataset_name: str):
        """Enhanced intra-table relationship discovery using ML"""
        
        for table_name, df in tables.items():
            print(f"Analyzing relationships within {table_name}...")
            
            columns = list(df.columns)
            
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    # Use ML detector for same-table relationships
                    relationship_info = self.ml_detector.detect_relationships(
                        df, df, col1, col2, table_name, table_name
                    )
                    
                    # Add relationship if confidence is high enough
                    if relationship_info['confidence'] > 0.5:
                        source_node = f"COLUMN:{dataset_name}.{table_name}.{col1}"
                        target_node = f"COLUMN:{dataset_name}.{table_name}.{col2}"
                        
                        self.graph.add_edge(source_node, target_node,
                                          relationship=relationship_info['relationship_type'],
                                          weight=relationship_info['confidence'],
                                          evidence=relationship_info['evidence'],
                                          ml_features=relationship_info['features'])
    
    def get_relationships_by_type(self, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Get relationships filtered by type"""
        relationships = []
        
        for source, target, data in self.graph.edges(data=True):
            if relationship_type is None or data.get('relationship') == relationship_type:
                relationships.append({
                    'source': source,
                    'target': target,
                    'relationship': data.get('relationship', 'UNKNOWN'),
                    'weight': data.get('weight', 0.0),
                    'evidence': data.get('evidence', {}),
                    'features': data.get('ml_features', {})
                })
        
        return sorted(relationships, key=lambda x: x['weight'], reverse=True)
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of all discovered relationships"""
        summary = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'relationship_types': {},
            'top_relationships': []
        }
        
        # Count node types
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            summary['node_types'][node_type] = summary['node_types'].get(node_type, 0) + 1
        
        # Count and analyze relationship types
        all_relationships = []
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            if rel_type not in ['CONTAINS', 'HAS_COLUMN']:  # Exclude structural relationships
                summary['relationship_types'][rel_type] = summary['relationship_types'].get(rel_type, 0) + 1
                all_relationships.append({
                    'source': source,
                    'target': target,
                    'type': rel_type,
                    'weight': data.get('weight', 0.0)
                })
        
        # Get top relationships by weight
        summary['top_relationships'] = sorted(all_relationships, key=lambda x: x['weight'], reverse=True)[:10]
        
        return summary
    
    def visualize_enhanced_graph(self, max_nodes: int = 60, min_confidence: float = 0.5):
        """Enhanced visualization showing different relationship types"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.patches import Rectangle
        import matplotlib.patches as mpatches
        
        # Filter nodes based on relationships
        nodes_to_show = set()
        
        # Add all datasets and tables
        for node, data in self.graph.nodes(data=True):
            if data.get('type') in ['dataset', 'table']:
                nodes_to_show.add(node)
        
        # Add columns with significant relationships
        for u, v, data in self.graph.edges(data=True):
            if data.get('weight', 0) >= min_confidence and data.get('relationship') not in ['CONTAINS', 'HAS_COLUMN']:
                nodes_to_show.add(u)
                nodes_to_show.add(v)
        
        # Limit to max_nodes
        nodes_to_show = list(nodes_to_show)[:max_nodes]
        subgraph = self.graph.subgraph(nodes_to_show)
        
        # Create layout with better spacing
        pos = nx.spring_layout(subgraph, k=4, iterations=100, seed=42)
        
        # Set up the plot
        plt.figure(figsize=(20, 15))
        ax = plt.gca()
        
        # Define styles for different node types
        node_styles = {
            'dataset': {'color': '#e74c3c', 'size': 1200, 'shape': 's'},  # Square
            'table': {'color': '#3498db', 'size': 800, 'shape': 'o'},      # Circle
            'column': {'color': '#2ecc71', 'size': 400, 'shape': 'o'}      # Circle
        }
        
        # Draw nodes by type
        for node_type, style in node_styles.items():
            nodes_of_type = [n for n in subgraph.nodes() 
                           if subgraph.nodes[n].get('type') == node_type]
            
            if nodes_of_type:
                nx.draw_networkx_nodes(subgraph, pos, 
                                     nodelist=nodes_of_type,
                                     node_color=style['color'],
                                     node_size=style['size'],
                                     node_shape=style['shape'],
                                     alpha=0.8)
        
        # Define styles for different relationship types
        edge_styles = {
            'CONTAINS': {'color': '#95a5a6', 'width': 1, 'style': 'dashed', 'alpha': 0.3},
            'HAS_COLUMN': {'color': '#bdc3c7', 'width': 1, 'style': 'dotted', 'alpha': 0.3},
            'FOREIGN_KEY': {'color': '#e74c3c', 'width': 3, 'style': 'solid', 'alpha': 0.8},
            'POSITIVELY_CORRELATED': {'color': '#27ae60', 'width': 2, 'style': 'solid', 'alpha': 0.7},
            'NEGATIVELY_CORRELATED': {'color': '#e67e22', 'width': 2, 'style': 'solid', 'alpha': 0.7},
            'SAME_DOMAIN': {'color': '#9b59b6', 'width': 2, 'style': 'dashed', 'alpha': 0.7},
            'INFORMATION_DEPENDENCY': {'color': '#f39c12', 'width': 2, 'style': 'dashdot', 'alpha': 0.7},
            'SIMILAR_VALUES': {'color': '#16a085', 'width': 2, 'style': 'dotted', 'alpha': 0.7},
            'WEAK_RELATIONSHIP': {'color': '#7f8c8d', 'width': 1, 'style': 'dotted', 'alpha': 0.5}
        }
        
        # Draw edges by relationship type
        for relationship, style in edge_styles.items():
            edges_of_type = [(u, v) for u, v, d in subgraph.edges(data=True)
                           if d.get('relationship') == relationship]
            
            if edges_of_type:
                # Get edge weights for varying line thickness
                edge_weights = [subgraph[u][v][0].get('weight', 0.5) * style['width'] 
                               for u, v in edges_of_type]
                
                nx.draw_networkx_edges(subgraph, pos,
                                     edgelist=edges_of_type,
                                     edge_color=style['color'],
                                     width=edge_weights,
                                     style=style['style'],
                                     alpha=style['alpha'],
                                     arrows=True,
                                     arrowsize=15,
                                     arrowstyle='->')
        
        # Add labels
        labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            if node_data.get('type') == 'column':
                # Show table.column for columns
                parts = node.split('.')
                if len(parts) >= 3:
                    table = parts[1].replace('olist_', '').replace('_dataset', '')
                    col = parts[2]
                    labels[node] = f"{table}.{col}"
                else:
                    labels[node] = node_data.get('name', node)
            elif node_data.get('type') == 'table':
                # Shorten table names
                name = node_data.get('name', node)
                labels[node] = name.replace('olist_', '').replace('_dataset', '')
            else:
                labels[node] = node_data.get('name', node)
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
        
        # Add title
        plt.title("Enhanced Knowledge Graph with ML-Detected Relationships", 
                  size=20, weight='bold', pad=20)
        
        # Create legend
        legend_elements = []
        
        # Node types
        legend_elements.append(mpatches.Patch(color='none', label='Node Types:'))
        for node_type, style in node_styles.items():
            legend_elements.append(
                mpatches.Patch(color=style['color'], label=f'  {node_type.title()}')
            )
        
        # Add space
        legend_elements.append(mpatches.Patch(color='none', label=''))
        legend_elements.append(mpatches.Patch(color='none', label='Relationships:'))
        
        # Relationship types (exclude structural)
        for rel_type, style in edge_styles.items():
            if rel_type not in ['CONTAINS', 'HAS_COLUMN']:
                legend_elements.append(
                    mpatches.Patch(color=style['color'], label=f'  {rel_type.replace("_", " ").title()}')
                )
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), 
                  fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        summary = self.get_relationship_summary()
        print(f"\nEnhanced Graph Statistics:")
        print(f"  Total Nodes: {summary['total_nodes']}")
        print(f"  Total Edges: {summary['total_edges']}")
        print(f"\n  Discovered Relationships:")
        for rel_type, count in summary['relationship_types'].items():
            print(f"    - {rel_type}: {count}")
        
        return summary