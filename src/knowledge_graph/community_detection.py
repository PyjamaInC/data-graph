"""
Community Detection for Semantic Table Graph

This module implements sophisticated community detection algorithms for both
table-level and column-level graphs, enabling intelligent query routing and
hierarchical analysis patterns.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, field

try:
    from networkx.algorithms import community
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity
    CLUSTERING_AVAILABLE = True
except ImportError:
    community = None
    SpectralClustering = None
    cosine_similarity = None
    CLUSTERING_AVAILABLE = False

# Import our components
try:
    from .table_relationships import TableRelationshipType
    from .table_intelligence import TableProfile
except ImportError:
    from table_relationships import TableRelationshipType
    from table_intelligence import TableProfile


@dataclass
class TableCommunity:
    """Represents a community of related tables"""
    community_id: str
    community_name: str
    tables: Set[str]
    dominant_domain: str
    key_concepts: List[str]
    community_type: str  # 'business_domain', 'structural', 'semantic'
    cohesion_score: float
    central_table: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnCommunity:
    """Represents a community of related columns"""
    community_id: str
    community_name: str
    columns: Set[str]
    community_type: str  # 'intra_table', 'cross_table'
    semantic_role: Optional[str] = None
    cohesion_score: float = 0.0
    representative_table: Optional[str] = None
    description: str = ""


class GraphCommunityDetector:
    """Detect communities in both table and column graphs using multiple algorithms"""
    
    def __init__(self, table_graph: nx.MultiDiGraph, column_graph: nx.MultiDiGraph):
        self.table_graph = table_graph
        self.column_graph = column_graph
        self.logger = logging.getLogger(f"{__name__}.GraphCommunityDetector")
        
        # Community storage
        self.table_communities: Dict[str, TableCommunity] = {}
        self.column_communities: Dict[str, ColumnCommunity] = {}
        self.table_to_community: Dict[str, str] = {}
        self.column_to_community: Dict[str, str] = {}
        
        # Algorithm parameters
        self.min_community_size = 2
        self.resolution = 1.0  # Louvain resolution parameter
        self.cohesion_threshold = 0.3
        
        if not CLUSTERING_AVAILABLE:
            self.logger.warning("Clustering libraries not available. Install scikit-learn and networkx[community]")
    
    def detect_all_communities(self, table_profiles: Optional[Dict[str, TableProfile]] = None) -> Dict[str, Any]:
        """Detect both table and column communities"""
        self.logger.info("Starting comprehensive community detection...")
        
        results = {}
        
        # Detect table communities
        if self.table_graph.number_of_nodes() > 1:
            table_communities = self.detect_table_communities(table_profiles)
            results['table_communities'] = table_communities
            self.logger.info(f"Detected {len(table_communities)} table communities")
        
        # Detect column communities
        if self.column_graph.number_of_nodes() > 1:
            column_communities = self.detect_column_communities_hierarchical()
            results['column_communities'] = column_communities
            self.logger.info(f"Detected {len(column_communities)} column communities")
        
        # Generate community summary
        results['summary'] = self._generate_community_summary()
        
        return results
    
    def detect_table_communities(self, table_profiles: Optional[Dict[str, TableProfile]] = None) -> Dict[str, TableCommunity]:
        """Detect business domain communities in table graph"""
        
        if not CLUSTERING_AVAILABLE:
            self.logger.warning("Using fallback community detection")
            return self._fallback_table_communities(table_profiles)
        
        # Convert to undirected for community detection
        undirected_graph = self.table_graph.to_undirected()
        
        if undirected_graph.number_of_nodes() < 2:
            return {}
        
        # Method 1: Louvain community detection
        louvain_communities = self._detect_louvain_communities(undirected_graph)
        
        # Method 2: Business domain clustering (if profiles available)
        domain_communities = {}
        if table_profiles:
            domain_communities = self._detect_domain_communities(table_profiles)
        
        # Method 3: Structural similarity communities
        structural_communities = self._detect_structural_communities(undirected_graph)
        
        # Combine and validate communities
        final_communities = self._merge_community_detections(
            louvain_communities, domain_communities, structural_communities, table_profiles
        )
        
        # Store communities
        self.table_communities = final_communities
        self._update_table_community_mapping()
        
        return final_communities
    
    def _detect_louvain_communities(self, graph: nx.Graph) -> Dict[str, Set[str]]:
        """Detect communities using Louvain algorithm"""
        try:
            communities = community.louvain_communities(graph, resolution=self.resolution, seed=42)
            
            labeled_communities = {}
            for i, comm in enumerate(communities):
                if len(comm) >= self.min_community_size:
                    community_name = f"louvain_community_{i}"
                    labeled_communities[community_name] = comm
            
            return labeled_communities
            
        except Exception as e:
            self.logger.warning(f"Louvain community detection failed: {e}")
            return {}
    
    def _detect_domain_communities(self, table_profiles: Dict[str, TableProfile]) -> Dict[str, Set[str]]:
        """Detect communities based on business domains"""
        domain_groups = defaultdict(set)
        
        for table_name, profile in table_profiles.items():
            domain = profile.business_domain or 'unknown'
            domain_groups[domain].add(table_name)
        
        # Filter out small communities and unknown domain
        filtered_communities = {}
        for domain, tables in domain_groups.items():
            if len(tables) >= self.min_community_size and domain != 'unknown':
                filtered_communities[f"domain_{domain}"] = tables
        
        return filtered_communities
    
    def _detect_structural_communities(self, graph: nx.Graph) -> Dict[str, Set[str]]:
        """Detect communities based on structural similarity"""
        try:
            # Use modularity-based community detection
            communities = community.greedy_modularity_communities(graph)
            
            structural_communities = {}
            for i, comm in enumerate(communities):
                if len(comm) >= self.min_community_size:
                    structural_communities[f"structural_community_{i}"] = comm
            
            return structural_communities
            
        except Exception as e:
            self.logger.warning(f"Structural community detection failed: {e}")
            return {}
    
    def _merge_community_detections(self, louvain_comms: Dict[str, Set[str]], 
                                   domain_comms: Dict[str, Set[str]],
                                   structural_comms: Dict[str, Set[str]],
                                   table_profiles: Optional[Dict[str, TableProfile]]) -> Dict[str, TableCommunity]:
        """Merge results from different community detection methods"""
        
        all_communities = {}
        community_counter = 0
        
        # Prioritize domain-based communities (highest quality)
        for domain_name, tables in domain_comms.items():
            community_id = f"community_{community_counter}"
            
            community = TableCommunity(
                community_id=community_id,
                community_name=domain_name,
                tables=tables,
                dominant_domain=domain_name.replace('domain_', ''),
                key_concepts=self._extract_community_concepts(tables, table_profiles),
                community_type='business_domain',
                cohesion_score=self._calculate_community_cohesion(tables),
                central_table=self._find_central_table(tables),
                description=f"Business domain community: {domain_name.replace('domain_', '')}"
            )
            
            all_communities[community_id] = community
            community_counter += 1
        
        # Add Louvain communities that don't overlap significantly with domain communities
        for louvain_name, tables in louvain_comms.items():
            if not self._has_significant_overlap(tables, all_communities):
                community_id = f"community_{community_counter}"
                
                # Analyze community characteristics
                dominant_domain = self._analyze_community_domain(tables, table_profiles)
                
                community = TableCommunity(
                    community_id=community_id,
                    community_name=f"algorithmic_{community_counter}",
                    tables=tables,
                    dominant_domain=dominant_domain,
                    key_concepts=self._extract_community_concepts(tables, table_profiles),
                    community_type='algorithmic',
                    cohesion_score=self._calculate_community_cohesion(tables),
                    central_table=self._find_central_table(tables),
                    description=f"Algorithmically detected community with {len(tables)} tables"
                )
                
                all_communities[community_id] = community
                community_counter += 1
        
        # Add any remaining isolated tables as singleton communities
        all_assigned_tables = set()
        for community in all_communities.values():
            all_assigned_tables.update(community.tables)
        
        for table in self.table_graph.nodes():
            if table not in all_assigned_tables:
                community_id = f"community_{community_counter}"
                
                community = TableCommunity(
                    community_id=community_id,
                    community_name=f"singleton_{table}",
                    tables={table},
                    dominant_domain=self._get_table_domain(table, table_profiles),
                    key_concepts=self._get_table_concepts(table, table_profiles),
                    community_type='singleton',
                    cohesion_score=1.0,
                    central_table=table,
                    description=f"Singleton community containing only {table}"
                )
                
                all_communities[community_id] = community
                community_counter += 1
        
        return all_communities
    
    def _fallback_table_communities(self, table_profiles: Optional[Dict[str, TableProfile]]) -> Dict[str, TableCommunity]:
        """Fallback community detection when advanced algorithms aren't available"""
        communities = {}
        
        if not table_profiles:
            # Create single community with all tables
            community = TableCommunity(
                community_id="community_0",
                community_name="all_tables",
                tables=set(self.table_graph.nodes()),
                dominant_domain="mixed",
                key_concepts=[],
                community_type="fallback",
                cohesion_score=0.5,
                description="Fallback community containing all tables"
            )
            communities["community_0"] = community
            return communities
        
        # Group by business domain
        domain_groups = defaultdict(set)
        for table_name, profile in table_profiles.items():
            domain = profile.business_domain or 'unknown'
            domain_groups[domain].add(table_name)
        
        # Create communities from domain groups
        for i, (domain, tables) in enumerate(domain_groups.items()):
            community_id = f"community_{i}"
            community = TableCommunity(
                community_id=community_id,
                community_name=domain,
                tables=tables,
                dominant_domain=domain,
                key_concepts=self._extract_community_concepts(tables, table_profiles),
                community_type='business_domain',
                cohesion_score=0.7,
                description=f"Domain-based community: {domain}"
            )
            communities[community_id] = community
        
        return communities
    
    def detect_column_communities_hierarchical(self) -> Dict[str, Dict[str, ColumnCommunity]]:
        """Detect column communities with table-awareness"""
        
        communities = {
            'intra_table': {},
            'cross_table': {}
        }
        
        # Level 1: Communities within each table
        intra_table_communities = self._detect_intra_table_communities()
        communities['intra_table'] = intra_table_communities
        
        # Level 2: Cross-table column communities
        cross_table_communities = self._detect_cross_table_communities()
        communities['cross_table'] = cross_table_communities
        
        # Store in instance variables
        self.column_communities = {**intra_table_communities, **cross_table_communities}
        self._update_column_community_mapping()
        
        return communities
    
    def _detect_intra_table_communities(self) -> Dict[str, ColumnCommunity]:
        """Detect semantic groups within each table"""
        intra_communities = {}
        community_counter = 0
        
        # Get table names from table graph
        table_names = set(self.table_graph.nodes())
        
        for table in table_names:
            # Find columns belonging to this table
            table_columns = [
                node for node in self.column_graph.nodes()
                if self._extract_table_from_column_node(node) == table
            ]
            
            if len(table_columns) <= 1:
                continue
            
            # Create subgraph for this table's columns
            try:
                subgraph = self.column_graph.subgraph(table_columns)
                
                if subgraph.number_of_nodes() >= 2:
                    communities = self._detect_column_subcommunities(subgraph, table)
                    
                    for community_type, columns in communities.items():
                        if len(columns) >= 1:  # Allow single-column communities for important columns
                            community_id = f"intra_table_{community_counter}"
                            
                            community = ColumnCommunity(
                                community_id=community_id,
                                community_name=f"{table}_{community_type}",
                                columns=columns,
                                community_type='intra_table',
                                semantic_role=community_type,
                                cohesion_score=self._calculate_column_cohesion(columns, subgraph),
                                representative_table=table,
                                description=f"{community_type} columns in {table}"
                            )
                            
                            intra_communities[community_id] = community
                            community_counter += 1
                            
            except Exception as e:
                self.logger.warning(f"Failed to detect communities in table {table}: {e}")
        
        return intra_communities
    
    def _detect_column_subcommunities(self, subgraph: nx.Graph, table_name: str) -> Dict[str, Set[str]]:
        """Detect semantic groups within a table"""
        
        if subgraph.number_of_nodes() < 2:
            return {}
        
        communities = {}
        
        # Group columns by semantic patterns (simple heuristic approach)
        identifier_cols = set()
        measure_cols = set()
        dimension_cols = set()
        temporal_cols = set()
        other_cols = set()
        
        for node in subgraph.nodes():
            col_name = node.split('.')[-1].lower()  # Extract column name
            
            # Simple pattern matching for column types
            if any(pattern in col_name for pattern in ['id', 'key', 'code']):
                identifier_cols.add(node)
            elif any(pattern in col_name for pattern in ['amount', 'price', 'cost', 'value', 'count', 'qty', 'quantity']):
                measure_cols.add(node)
            elif any(pattern in col_name for pattern in ['date', 'time', 'timestamp', 'created', 'updated']):
                temporal_cols.add(node)
            elif any(pattern in col_name for pattern in ['status', 'type', 'category', 'state', 'region']):
                dimension_cols.add(node)
            else:
                other_cols.add(node)
        
        # Add non-empty communities
        if identifier_cols:
            communities['identifiers'] = identifier_cols
        if measure_cols:
            communities['measures'] = measure_cols
        if dimension_cols:
            communities['dimensions'] = dimension_cols
        if temporal_cols:
            communities['temporal'] = temporal_cols
        if other_cols:
            communities['descriptive'] = other_cols
        
        return communities
    
    def _detect_cross_table_communities(self) -> Dict[str, ColumnCommunity]:
        """Detect column communities that span multiple tables"""
        cross_communities = {}
        community_counter = 0
        
        # Group columns by semantic similarity across tables
        semantic_groups = self._group_columns_by_semantics()
        
        for semantic_type, columns in semantic_groups.items():
            if len(columns) >= 2:  # At least 2 columns to form a community
                # Check if columns span multiple tables
                tables = set()
                for col in columns:
                    table = self._extract_table_from_column_node(col)
                    if table:
                        tables.add(table)
                
                if len(tables) > 1:  # Cross-table community
                    community_id = f"cross_table_{community_counter}"
                    
                    community = ColumnCommunity(
                        community_id=community_id,
                        community_name=f"cross_table_{semantic_type}",
                        columns=set(columns),
                        community_type='cross_table',
                        semantic_role=semantic_type,
                        cohesion_score=self._calculate_cross_table_cohesion(columns),
                        description=f"Cross-table {semantic_type} columns spanning {len(tables)} tables"
                    )
                    
                    cross_communities[community_id] = community
                    community_counter += 1
        
        return cross_communities
    
    def _group_columns_by_semantics(self) -> Dict[str, List[str]]:
        """Group columns by semantic patterns across all tables"""
        semantic_groups = defaultdict(list)
        
        for node in self.column_graph.nodes():
            col_name = node.split('.')[-1].lower()  # Extract column name
            
            # Categorize by semantic meaning
            if any(pattern in col_name for pattern in ['customer', 'user', 'client']):
                semantic_groups['customer_identifiers'].append(node)
            elif any(pattern in col_name for pattern in ['product', 'item']):
                semantic_groups['product_identifiers'].append(node)
            elif any(pattern in col_name for pattern in ['order', 'transaction']):
                semantic_groups['order_identifiers'].append(node)
            elif any(pattern in col_name for pattern in ['seller', 'vendor', 'supplier']):
                semantic_groups['seller_identifiers'].append(node)
            elif any(pattern in col_name for pattern in ['price', 'cost', 'amount', 'value']):
                semantic_groups['monetary_values'].append(node)
            elif any(pattern in col_name for pattern in ['date', 'time', 'timestamp']):
                semantic_groups['temporal_columns'].append(node)
            elif any(pattern in col_name for pattern in ['zip', 'postal', 'city', 'state', 'country']):
                semantic_groups['location_columns'].append(node)
            elif any(pattern in col_name for pattern in ['status', 'state']):
                semantic_groups['status_columns'].append(node)
        
        return semantic_groups
    
    def _extract_table_from_column_node(self, node: str) -> Optional[str]:
        """Extract table name from column node identifier"""
        if "COLUMN:" in node:
            # Format: COLUMN:dataset.table.column
            parts = node.replace("COLUMN:", "").split(".")
            if len(parts) >= 3:
                return parts[1]  # Return table name
        return None
    
    def _calculate_community_cohesion(self, tables: Set[str]) -> float:
        """Calculate cohesion score for a table community"""
        if len(tables) <= 1:
            return 1.0
        
        # Count internal edges vs possible edges
        internal_edges = 0
        possible_edges = 0
        
        table_list = list(tables)
        for i in range(len(table_list)):
            for j in range(i + 1, len(table_list)):
                possible_edges += 1
                if self.table_graph.has_edge(table_list[i], table_list[j]):
                    internal_edges += 1
        
        return internal_edges / possible_edges if possible_edges > 0 else 0.0
    
    def _calculate_column_cohesion(self, columns: Set[str], subgraph: nx.Graph) -> float:
        """Calculate cohesion score for column community"""
        if len(columns) <= 1:
            return 1.0
        
        # Calculate density of connections within the community
        community_subgraph = subgraph.subgraph(columns)
        return nx.density(community_subgraph)
    
    def _calculate_cross_table_cohesion(self, columns: List[str]) -> float:
        """Calculate cohesion for cross-table column community"""
        if len(columns) <= 1:
            return 1.0
        
        # Simple heuristic: columns with similar names have higher cohesion
        name_similarity = 0.0
        comparisons = 0
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i].split('.')[-1].lower()
                col2 = columns[j].split('.')[-1].lower()
                
                # Simple string similarity
                if col1 == col2:
                    name_similarity += 1.0
                elif any(word in col1 and word in col2 for word in ['id', 'date', 'price', 'name']):
                    name_similarity += 0.5
                
                comparisons += 1
        
        return name_similarity / comparisons if comparisons > 0 else 0.0
    
    def _has_significant_overlap(self, tables: Set[str], existing_communities: Dict[str, TableCommunity], 
                               threshold: float = 0.5) -> bool:
        """Check if a table set has significant overlap with existing communities"""
        for community in existing_communities.values():
            overlap = len(tables.intersection(community.tables))
            if overlap / len(tables) > threshold:
                return True
        return False
    
    def _analyze_community_domain(self, tables: Set[str], 
                                table_profiles: Optional[Dict[str, TableProfile]]) -> str:
        """Determine the dominant business domain of a community"""
        if not table_profiles:
            return "unknown"
        
        domains = []
        for table in tables:
            if table in table_profiles:
                domain = table_profiles[table].business_domain
                if domain:
                    domains.append(domain)
        
        if domains:
            domain_counts = Counter(domains)
            return domain_counts.most_common(1)[0][0]
        
        # Fallback: analyze table names
        table_names_text = ' '.join(tables).lower()
        if 'customer' in table_names_text:
            return 'customer_management'
        elif 'order' in table_names_text:
            return 'order_processing'
        elif 'product' in table_names_text:
            return 'product_catalog'
        elif 'seller' in table_names_text:
            return 'seller_management'
        
        return 'mixed'
    
    def _extract_community_concepts(self, tables: Set[str], 
                                  table_profiles: Optional[Dict[str, TableProfile]]) -> List[str]:
        """Extract key concepts from a community of tables"""
        if not table_profiles:
            return list(tables)  # Use table names as concepts
        
        all_concepts = []
        for table in tables:
            if table in table_profiles:
                all_concepts.extend(table_profiles[table].key_concepts)
        
        # Count concept frequency and return most common
        concept_counts = Counter(all_concepts)
        return [concept for concept, count in concept_counts.most_common(10)]
    
    def _find_central_table(self, tables: Set[str]) -> Optional[str]:
        """Find the most central table in a community"""
        if not tables:
            return None
        
        if len(tables) == 1:
            return list(tables)[0]
        
        # Calculate centrality within the community subgraph
        try:
            subgraph = self.table_graph.subgraph(tables)
            centrality = nx.degree_centrality(subgraph)
            
            if centrality:
                return max(centrality.items(), key=lambda x: x[1])[0]
        except:
            pass
        
        # Fallback: return first table
        return list(tables)[0]
    
    def _get_table_domain(self, table: str, table_profiles: Optional[Dict[str, TableProfile]]) -> str:
        """Get domain for a single table"""
        if table_profiles and table in table_profiles:
            return table_profiles[table].business_domain or "unknown"
        return "unknown"
    
    def _get_table_concepts(self, table: str, table_profiles: Optional[Dict[str, TableProfile]]) -> List[str]:
        """Get concepts for a single table"""
        if table_profiles and table in table_profiles:
            return table_profiles[table].key_concepts
        return [table]
    
    def _update_table_community_mapping(self):
        """Update table to community mapping"""
        self.table_to_community = {}
        for community_id, community in self.table_communities.items():
            for table in community.tables:
                self.table_to_community[table] = community_id
    
    def _update_column_community_mapping(self):
        """Update column to community mapping"""
        self.column_to_community = {}
        for community_id, community in self.column_communities.items():
            for column in community.columns:
                self.column_to_community[column] = community_id
    
    def _generate_community_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of detected communities"""
        return {
            'table_communities': {
                'count': len(self.table_communities),
                'types': Counter([c.community_type for c in self.table_communities.values()]),
                'domains': Counter([c.dominant_domain for c in self.table_communities.values()]),
                'avg_size': np.mean([len(c.tables) for c in self.table_communities.values()]) if self.table_communities else 0,
                'avg_cohesion': np.mean([c.cohesion_score for c in self.table_communities.values()]) if self.table_communities else 0
            },
            'column_communities': {
                'count': len(self.column_communities),
                'types': Counter([c.community_type for c in self.column_communities.values()]),
                'semantic_roles': Counter([c.semantic_role for c in self.column_communities.values() if c.semantic_role]),
                'avg_size': np.mean([len(c.columns) for c in self.column_communities.values()]) if self.column_communities else 0,
                'avg_cohesion': np.mean([c.cohesion_score for c in self.column_communities.values()]) if self.column_communities else 0
            }
        }
    
    def get_table_community(self, table_name: str) -> Optional[TableCommunity]:
        """Get the community that contains the specified table"""
        community_id = self.table_to_community.get(table_name)
        if community_id:
            return self.table_communities.get(community_id)
        return None
    
    def get_column_community(self, column_name: str) -> Optional[ColumnCommunity]:
        """Get the community that contains the specified column"""
        community_id = self.column_to_community.get(column_name)
        if community_id:
            return self.column_communities.get(community_id)
        return None
    
    def get_community_tables(self, community_id: str) -> Set[str]:
        """Get all tables in a specific community"""
        community = self.table_communities.get(community_id)
        return community.tables if community else set()
    
    def get_related_communities(self, table_name: str, max_communities: int = 3) -> List[Tuple[str, float]]:
        """Get communities related to the given table's community"""
        current_community = self.get_table_community(table_name)
        if not current_community:
            return []
        
        # Calculate similarity to other communities based on shared concepts and domains
        related = []
        for community_id, community in self.table_communities.items():
            if community_id != current_community.community_id:
                similarity = self._calculate_community_similarity(current_community, community)
                if similarity > 0.1:  # Minimum similarity threshold
                    related.append((community_id, similarity))
        
        # Sort by similarity and return top communities
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_communities]
    
    def _calculate_community_similarity(self, community1: TableCommunity, community2: TableCommunity) -> float:
        """Calculate similarity between two communities"""
        similarity = 0.0
        
        # Domain similarity
        if community1.dominant_domain == community2.dominant_domain:
            similarity += 0.5
        
        # Concept overlap
        concepts1 = set(community1.key_concepts)
        concepts2 = set(community2.key_concepts)
        if concepts1 and concepts2:
            concept_overlap = len(concepts1.intersection(concepts2)) / len(concepts1.union(concepts2))
            similarity += concept_overlap * 0.3
        
        # Table connectivity (check if communities have connecting tables)
        for table1 in community1.tables:
            for table2 in community2.tables:
                if self.table_graph.has_edge(table1, table2):
                    similarity += 0.2
                    break
        
        return min(similarity, 1.0)


class CommunityAwareQueryRouter:
    """Route queries to relevant communities for focused analysis"""
    
    def __init__(self, community_detector: GraphCommunityDetector):
        self.communities = community_detector
        self.logger = logging.getLogger(f"{__name__}.CommunityAwareQueryRouter")
        
        # Build routing patterns and caches
        self.concept_to_communities = self._build_concept_mapping()
        self.domain_to_communities = self._build_domain_mapping()
        
    def route_query_to_communities(self, intent: Dict[str, Any], 
                                 target_concepts: List[str]) -> Dict[str, float]:
        """Route query to relevant table communities with confidence scores"""
        
        community_scores = defaultdict(float)
        
        # Score communities based on concept matches
        for concept in target_concepts:
            concept_lower = concept.lower()
            
            # Direct concept matching
            if concept_lower in self.concept_to_communities:
                for community_id in self.concept_to_communities[concept_lower]:
                    community_scores[community_id] += 0.4
            
            # Fuzzy concept matching
            for mapped_concept, community_ids in self.concept_to_communities.items():
                if concept_lower in mapped_concept or mapped_concept in concept_lower:
                    for community_id in community_ids:
                        community_scores[community_id] += 0.2
        
        # Score based on intent alignment
        action_type = intent.get('action_type', '')
        for community_id, community in self.communities.table_communities.items():
            intent_score = self._score_community_intent_alignment(community, action_type)
            community_scores[community_id] += intent_score
        
        # Score based on domain relevance
        for domain, community_ids in self.domain_to_communities.items():
            if any(domain_word in ' '.join(target_concepts).lower() 
                   for domain_word in domain.split('_')):
                for community_id in community_ids:
                    community_scores[community_id] += 0.3
        
        # Normalize scores
        total_score = sum(community_scores.values())
        if total_score > 0:
            community_scores = {k: v/total_score for k, v in community_scores.items()}
        
        return dict(community_scores)
    
    def _build_concept_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from concepts to communities"""
        concept_mapping = defaultdict(list)
        
        for community_id, community in self.communities.table_communities.items():
            for concept in community.key_concepts:
                concept_mapping[concept.lower()].append(community_id)
        
        return dict(concept_mapping)
    
    def _build_domain_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from domains to communities"""
        domain_mapping = defaultdict(list)
        
        for community_id, community in self.communities.table_communities.items():
            domain_mapping[community.dominant_domain].append(community_id)
        
        return dict(domain_mapping)
    
    def _score_community_intent_alignment(self, community: TableCommunity, action_type: str) -> float:
        """Score how well a community aligns with query intent"""
        scores = {
            'aggregation': {
                'fact': 0.4, 'bridge': 0.2, 'dimension': 0.1
            },
            'filtering': {
                'dimension': 0.4, 'bridge': 0.3, 'fact': 0.2
            },
            'joining': {
                'bridge': 0.4, 'fact': 0.3, 'dimension': 0.1
            },
            'trend_analysis': {
                'fact': 0.4, 'bridge': 0.2, 'dimension': 0.1
            }
        }
        
        # Get table types in community (from table graph node data)
        community_table_types = []
        for table in community.tables:
            if table in self.communities.table_graph.nodes:
                table_type = self.communities.table_graph.nodes[table].get('table_type', 'unknown')
                community_table_types.append(table_type)
        
        # Calculate alignment score
        alignment_score = 0.0
        if action_type in scores:
            type_scores = scores[action_type]
            for table_type in community_table_types:
                alignment_score += type_scores.get(table_type, 0.0)
        
        return alignment_score / len(community_table_types) if community_table_types else 0.0
    
    def get_community_tables(self, community_id: str) -> List[str]:
        """Get all tables in a community"""
        community = self.communities.table_communities.get(community_id)
        return list(community.tables) if community else []
    
    def get_community_info(self, community_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a community"""
        community = self.communities.table_communities.get(community_id)
        if not community:
            return None
        
        return {
            'community_id': community.community_id,
            'community_name': community.community_name,
            'tables': list(community.tables),
            'dominant_domain': community.dominant_domain,
            'key_concepts': community.key_concepts,
            'community_type': community.community_type,
            'cohesion_score': community.cohesion_score,
            'central_table': community.central_table,
            'description': community.description,
            'table_count': len(community.tables)
        }
    
    def suggest_related_communities(self, primary_community_id: str, 
                                  max_suggestions: int = 3) -> List[Dict[str, Any]]:
        """Suggest related communities that might be relevant for analysis"""
        primary_community = self.communities.table_communities.get(primary_community_id)
        if not primary_community:
            return []
        
        suggestions = []
        
        # Find communities with similar domains or concepts
        for community_id, community in self.communities.table_communities.items():
            if community_id == primary_community_id:
                continue
            
            # Calculate relatedness
            relatedness = 0.0
            
            # Same domain bonus
            if community.dominant_domain == primary_community.dominant_domain:
                relatedness += 0.5
            
            # Shared concepts
            shared_concepts = set(community.key_concepts).intersection(set(primary_community.key_concepts))
            if shared_concepts:
                relatedness += len(shared_concepts) / len(set(community.key_concepts).union(set(primary_community.key_concepts))) * 0.3
            
            # Table connectivity
            for table1 in primary_community.tables:
                for table2 in community.tables:
                    if self.communities.table_graph.has_edge(table1, table2):
                        relatedness += 0.2
                        break
            
            if relatedness > 0.1:
                suggestions.append({
                    'community_id': community_id,
                    'community_name': community.community_name,
                    'relatedness_score': relatedness,
                    'shared_concepts': list(shared_concepts) if shared_concepts else [],
                    'relationship_reason': self._explain_relationship(primary_community, community)
                })
        
        # Sort by relatedness and return top suggestions
        suggestions.sort(key=lambda x: x['relatedness_score'], reverse=True)
        return suggestions[:max_suggestions]
    
    def _explain_relationship(self, community1: TableCommunity, community2: TableCommunity) -> str:
        """Explain why two communities are related"""
        reasons = []
        
        if community1.dominant_domain == community2.dominant_domain:
            reasons.append(f"Same business domain ({community1.dominant_domain})")
        
        shared_concepts = set(community1.key_concepts).intersection(set(community2.key_concepts))
        if shared_concepts:
            reasons.append(f"Shared concepts: {', '.join(list(shared_concepts)[:3])}")
        
        if not reasons:
            reasons.append("Structural connectivity")
        
        return "; ".join(reasons)