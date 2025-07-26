import networkx as nx
import pandas as pd
from typing import Dict, List, Any
try:
    from ..data.models.data_models import Relationship
except ImportError:
    from data.models.data_models import Relationship

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_metadata = {}
        self.edge_metadata = {}
        
    def add_dataset(self, tables: Dict[str, pd.DataFrame], dataset_name: str = "dataset"):
        """Add entire dataset to knowledge graph"""
        
        # Add dataset node
        dataset_node = f"DATASET:{dataset_name}"
        self.graph.add_node(dataset_node, type="dataset", name=dataset_name)
        
        # Add table and column nodes
        for table_name, df in tables.items():
            self._add_table_to_graph(table_name, df, dataset_name)
        
        # Discover relationships
        self._discover_intra_table_relationships(tables, dataset_name)
        self._discover_inter_table_relationships(tables, dataset_name)
        
        return self.graph
    
    def _add_table_to_graph(self, table_name: str, df: pd.DataFrame, dataset_name: str):
        """Add table and its columns to graph"""
        
        # Add table node
        table_node = f"TABLE:{dataset_name}.{table_name}"
        self.graph.add_node(table_node, 
                           type="table", 
                           name=table_name,
                           dataset=dataset_name,
                           row_count=len(df),
                           column_count=len(df.columns))
        
        # Connect dataset to table
        dataset_node = f"DATASET:{dataset_name}"
        self.graph.add_edge(dataset_node, table_node, 
                           relationship="CONTAINS", weight=1.0)
        
        # Add column nodes
        for column in df.columns:
            col_node = f"COLUMN:{dataset_name}.{table_name}.{column}"
            
            # Column statistics
            col_stats = self._calculate_column_stats(df[column])
            
            self.graph.add_node(col_node,
                               type="column",
                               name=column,
                               table=table_name,
                               dataset=dataset_name,
                               **col_stats)
            
            # Connect table to column
            self.graph.add_edge(table_node, col_node,
                               relationship="HAS_COLUMN", weight=1.0)
    
    def _calculate_column_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate column statistics"""
        stats = {
            'data_type': str(series.dtype),
            'null_count': int(series.isnull().sum()),
            'null_percentage': float(series.isnull().mean()),
            'unique_count': int(series.nunique()),
            'unique_percentage': float(series.nunique() / len(series)) if len(series) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': float(series.mean()) if not series.empty else 0,
                'std': float(series.std()) if not series.empty else 0,
                'min': float(series.min()) if not series.empty else 0,
                'max': float(series.max()) if not series.empty else 0,
                'median': float(series.median()) if not series.empty else 0
            })
        
        return stats
    
    def _discover_intra_table_relationships(self, tables: Dict[str, pd.DataFrame], dataset_name: str):
        """Discover relationships within each table"""
        
        for table_name, df in tables.items():
            # Correlation analysis for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                        corr_value = corr_matrix.iloc[i, j]
                        
                        if abs(corr_value) > 0.6:  # Strong correlation threshold
                            source_node = f"COLUMN:{dataset_name}.{table_name}.{col1}"
                            target_node = f"COLUMN:{dataset_name}.{table_name}.{col2}"
                            
                            rel_type = "POSITIVELY_CORRELATED" if corr_value > 0 else "NEGATIVELY_CORRELATED"
                            
                            self.graph.add_edge(source_node, target_node,
                                              relationship=rel_type,
                                              weight=abs(corr_value),
                                              correlation_value=corr_value,
                                              evidence="pearson_correlation")
    
    def _discover_inter_table_relationships(self, tables: Dict[str, pd.DataFrame], dataset_name: str):
        """Discover relationships between tables"""
        
        table_names = list(tables.keys())
        
        for i, table1_name in enumerate(table_names):
            for table2_name in table_names[i+1:]:
                df1, df2 = tables[table1_name], tables[table2_name]
                
                # Look for potential foreign key relationships
                self._find_foreign_key_relationships(df1, df2, table1_name, table2_name, dataset_name)
                
                # Look for similar columns
                self._find_similar_columns(df1, df2, table1_name, table2_name, dataset_name)
    
    def _find_foreign_key_relationships(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                      table1: str, table2: str, dataset_name: str):
        """Find potential foreign key relationships"""
        
        for col1 in df1.columns:
            for col2 in df2.columns:
                # Check if column names suggest FK relationship
                if self._is_potential_fk(col1, col2, table1, table2):
                    # Validate with value overlap analysis
                    overlap_ratio = self._calculate_value_overlap(df1[col1], df2[col2])
                    
                    if overlap_ratio > 0.7:  # High overlap suggests FK relationship
                        source_node = f"COLUMN:{dataset_name}.{table1}.{col1}"
                        target_node = f"COLUMN:{dataset_name}.{table2}.{col2}"
                        
                        self.graph.add_edge(source_node, target_node,
                                          relationship="FOREIGN_KEY",
                                          weight=overlap_ratio,
                                          overlap_ratio=overlap_ratio,
                                          evidence="value_overlap_analysis")
    
    def _is_potential_fk(self, col1: str, col2: str, table1: str, table2: str) -> bool:
        """Check if columns are potential foreign key relationships"""
        # Simple heuristics
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        table1_lower = table1.lower()
        table2_lower = table2.lower()
        
        # Check if col1 references table2 or vice versa
        if table2_lower in col1_lower or table1_lower in col2_lower:
            return True
        
        # Check if both columns have 'id' and similar names
        if 'id' in col1_lower and 'id' in col2_lower:
            # Remove 'id' and compare remaining parts
            base1 = col1_lower.replace('id', '').replace('_', '').replace('-', '')
            base2 = col2_lower.replace('id', '').replace('_', '').replace('-', '')
            
            if base1 == base2 and base1:  # Same base name
                return True
        
        return False
    
    def _calculate_value_overlap(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate overlap ratio between two series"""
        set1 = set(series1.dropna().unique())
        set2 = set(series2.dropna().unique())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_similar_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             table1: str, table2: str, dataset_name: str):
        """Find columns with similar names or content"""
        # This is a placeholder for more advanced similarity detection
        # Could be enhanced with semantic similarity, data distribution analysis, etc.
        pass
    
    def get_relationships(self, node: str = None) -> List[Dict[str, Any]]:
        """Get relationships from the graph"""
        relationships = []
        
        edges = self.graph.edges(data=True) if node is None else self.graph.edges(node, data=True)
        
        for source, target, data in edges:
            relationships.append({
                'source': source,
                'target': target,
                'relationship': data.get('relationship', 'UNKNOWN'),
                'weight': data.get('weight', 0.0),
                'evidence': data.get('evidence', 'unknown')
            })
        
        return relationships
    
    def find_related_columns(self, column_node: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """Find columns related to the given column"""
        if column_node not in self.graph:
            return []
        
        related = []
        
        try:
            # Find all nodes within max_distance
            for target in nx.single_source_shortest_path(self.graph, column_node, max_distance):
                if target != column_node and self.graph.nodes[target].get('type') == 'column':
                    # Calculate path weight
                    try:
                        path = nx.shortest_path(self.graph, column_node, target)
                        path_weight = self._calculate_path_weight(path)
                        
                        related.append({
                            'column': target,
                            'distance': len(path) - 1,
                            'weight': path_weight,
                            'path': path
                        })
                    except nx.NetworkXNoPath:
                        continue
                        
        except Exception as e:
            print(f"Error finding related columns: {e}")
        
        return sorted(related, key=lambda x: x['weight'], reverse=True)
    
    def _calculate_path_weight(self, path: List[str]) -> float:
        """Calculate the weight of a path through the graph"""
        if len(path) < 2:
            return 0.0
        
        total_weight = 1.0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            
            # Get edge data (handle multiple edges)
            edge_data = self.graph.get_edge_data(source, target)
            if edge_data:
                if isinstance(edge_data, dict):
                    # Single edge
                    weight = edge_data.get('weight', 0.5)
                else:
                    # Multiple edges, take maximum weight
                    weight = max(edge.get('weight', 0.5) for edge in edge_data.values())
                
                total_weight *= weight
            else:
                total_weight *= 0.5  # Default weight for missing edges
        
        return total_weight
    
    def visualize_graph(self, max_nodes: int = 50):
        """Create visualization of the knowledge graph"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Limit nodes for readability
        nodes_to_show = list(self.graph.nodes())[:max_nodes]
        subgraph = self.graph.subgraph(nodes_to_show)
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # Draw nodes by type
        node_types = {
            'dataset': {'color': 'lightcoral', 'size': 800},
            'table': {'color': 'lightblue', 'size': 600},
            'column': {'color': 'lightgreen', 'size': 400}
        }
        
        for node_type, style in node_types.items():
            nodes_of_type = [n for n in subgraph.nodes() 
                           if subgraph.nodes[n].get('type') == node_type]
            
            if nodes_of_type:
                nx.draw_networkx_nodes(subgraph, pos, 
                                     nodelist=nodes_of_type,
                                     node_color=style['color'],
                                     node_size=style['size'],
                                     alpha=0.7)
        
        # Draw edges by relationship type
        edge_colors = {
            'CONTAINS': 'gray',
            'HAS_COLUMN': 'blue', 
            'FOREIGN_KEY': 'red',
            'POSITIVELY_CORRELATED': 'green',
            'NEGATIVELY_CORRELATED': 'orange'
        }
        
        for relationship, color in edge_colors.items():
            edges_of_type = [(u, v) for u, v, d in subgraph.edges(data=True)
                           if d.get('relationship') == relationship]
            
            if edges_of_type:
                nx.draw_networkx_edges(subgraph, pos,
                                     edgelist=edges_of_type,
                                     edge_color=color,
                                     alpha=0.6,
                                     width=2)
        
        # Add labels (simplified for readability)
        labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            if node_data.get('type') == 'column':
                # Show only column name for columns
                labels[node] = node_data.get('name', node.split('.')[-1])
            else:
                labels[node] = node_data.get('name', node)
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        # Add legend
        legend_elements = []
        for node_type, style in node_types.items():
            legend_elements.append(plt.scatter([], [], c=style['color'], 
                                             s=style['size']/10, label=node_type.title()))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title("Knowledge Graph Structure", size=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print graph statistics
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {subgraph.number_of_nodes()}")
        print(f"  Edges: {subgraph.number_of_edges()}")
        print(f"  Node Types: {set(d.get('type', 'unknown') for n, d in subgraph.nodes(data=True))}")
        print(f"  Relationship Types: {set(d.get('relationship', 'unknown') for u, v, d in subgraph.edges(data=True))}")