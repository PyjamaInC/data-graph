## Implementation Plan

### Phase 1: Foundation Enhancement (Week 1-2)

1. **Enhance Knowledge Graph Builder**
   - Add `table_intelligence.py` with `TableIntelligenceLayer`
   - Implement table embedding generation using sentence-transformers
   - Create table profile generation with semantic summaries

2. **Build Semantic Table Graph**
   - Implement `semantic_table_graph.py` with `SemanticTableGraphBuilder`
   - Define `TableRelationshipType` enum and `TableRelationship` dataclass
   - Create relationship detectors (structural, semantic, temporal, business)
   - Build table-level graph with rich metadata

3. **Add Community Detection**
   - Implement `community_detection.py` with `GraphCommunityDetector`
   - Add table community detection using Louvain algorithm
   - Add hierarchical column community detection
   - Create `CommunityAwareQueryRouter` for query routing

### Phase 2: Query Planning Enhancement (Week 2-3)

4. **Replace Schema Validator**
   - Create `graph_aware_schema_validator.py` replacing current validator
   - Implement `HierarchicalTableSelector` with community → table → column flow
   - Add `_find_concept_via_graph` method for graph-based matching
   - Integrate table intelligence into concept mapping

5. **Enhance Orchestrator**
   - Update `enhanced_orchestrator.py` to use new components
   - Add `BidirectionalGraphEnhancer` for graph synergy
   - Implement `IntegratedGraphQueryPlanner`
   - Add initialization of table intelligence and community detection

6. **Add Query Pattern Learning**
   - Implement `query_pattern_learner.py`
   - Add pattern recording after successful queries
   - Create concept → table mapping history
   - Add pattern-based suggestions to planning

### Phase 3: Integration & Optimization (Week 3-4)

7. **Update Existing Components**
   - Modify `KnowledgeGraphContextExtractor` to use table graph
   - Update `RelationshipExplorerAgent` to leverage communities
   - Enhance error handling with community-aware fallbacks

8. **Create Configuration System**
   - Implement `system_config.py` for centralized configuration
   - Add flags for enabling/disabling features
   - Set confidence thresholds
   - Configure embedding models

9. **Add Visualization & Monitoring**
   - Implement `table_graph_visualizer.py` for graph visualization
   - Add performance metrics tracking
   - Create debugging utilities for graph exploration

### Phase 4: Testing & Refinement (Week 4)

10. **Testing Suite**
    - Unit tests for each new component
    - Integration tests for full query planning flow
    - Performance benchmarks comparing old vs new approach
    - Edge case handling tests

11. **Documentation & Examples**
    - Update module documentation
    - Create usage examples
    - Document new APIs
    - Add troubleshooting guide

### Key Dependencies

- Install: `sentence-transformers`, `networkx>=2.8`, `scikit-learn`
- Ensure: `ydata-profiling` integration still works
- Update: Requirements.txt

### Migration Strategy

1. Keep old components initially (feature flag to switch)
2. Run both systems in parallel for comparison
3. Gradually migrate after validation
4. Remove old components after stable operation

### Success Metrics

- Table selection accuracy > 90%
- Token usage reduced by 70%
- Query planning time < 2 seconds
- Pattern learning improves accuracy over time

======================================================
Excellent insights! Yes, community clustering and the semantic table graph can significantly enhance the existing column-level knowledge graph. Let me show you how these concepts can work together synergistically.

## 1. Community Clustering for Table & Column Groupings

### Community Detection Implementation

```python
# src/knowledge_graph/community_detection.py
from typing import Dict, List, Set, Tuple
import networkx as nx
from networkx.algorithms import community
import numpy as np
from sklearn.cluster import SpectralClustering

class GraphCommunityDetector:
    """Detect communities in both table and column graphs"""
    
    def __init__(self, table_graph: nx.MultiDiGraph, column_graph: nx.MultiDiGraph):
        self.table_graph = table_graph
        self.column_graph = column_graph
        self.table_communities = {}
        self.column_communities = {}
        
    def detect_table_communities(self) -> Dict[str, Set[str]]:
        """Detect business domain communities in table graph"""
        
        # Convert to undirected for community detection
        undirected = self.table_graph.to_undirected()
        
        # Method 1: Louvain community detection
        communities = community.louvain_communities(undirected, seed=42)
        
        # Label communities based on dominant characteristics
        labeled_communities = {}
        for i, comm in enumerate(communities):
            # Analyze community to determine its business domain
            domain = self._analyze_community_domain(comm)
            labeled_communities[domain] = comm
            
            # Store community membership
            for table in comm:
                self.table_communities[table] = domain
        
        return labeled_communities
    
    def detect_column_communities_hierarchical(self) -> Dict[str, Dict[str, Set[str]]]:
        """Detect column communities with table-awareness"""
        
        # First level: Communities within each table
        intra_table_communities = {}
        
        for table in self.table_graph.nodes():
            table_columns = [
                node for node in self.column_graph.nodes()
                if node.startswith(f"COLUMN:") and f".{table}." in node
            ]
            
            if len(table_columns) > 1:
                # Create subgraph for this table's columns
                subgraph = self.column_graph.subgraph(table_columns)
                
                # Detect communities within table
                communities = self._detect_column_subcommunities(subgraph, table)
                intra_table_communities[table] = communities
        
        # Second level: Cross-table column communities
        cross_table_communities = self._detect_cross_table_communities()
        
        return {
            'intra_table': intra_table_communities,
            'cross_table': cross_table_communities
        }
    
    def _detect_column_subcommunities(self, subgraph: nx.Graph, 
                                    table_name: str) -> Dict[str, Set[str]]:
        """Detect semantic groups within a table"""
        
        if subgraph.number_of_nodes() < 2:
            return {}
        
        # Use spectral clustering for column grouping
        adj_matrix = nx.adjacency_matrix(subgraph)
        n_clusters = min(3, subgraph.number_of_nodes() // 2)
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        labels = clustering.fit_predict(adj_matrix)
        
        # Group columns by cluster
        communities = {}
        nodes = list(subgraph.nodes())
        
        for i in range(n_clusters):
            cluster_nodes = [nodes[j] for j in range(len(nodes)) if labels[j] == i]
            
            # Analyze cluster to determine its semantic meaning
            cluster_type = self._analyze_column_cluster(cluster_nodes, table_name)
            communities[cluster_type] = set(cluster_nodes)
        
        return communities
    
    def _analyze_community_domain(self, tables: Set[str]) -> str:
        """Determine the business domain of a table community"""
        
        # Collect all business domains and concepts
        domains = []
        concepts = []
        
        for table in tables:
            node_data = self.table_graph.nodes[table]
            if node_data.get('business_domain'):
                domains.append(node_data['business_domain'])
            concepts.extend(node_data.get('key_concepts', []))
        
        # Most common domain
        if domains:
            from collections import Counter
            domain_counts = Counter(domains)
            return domain_counts.most_common(1)[0][0]
        
        # Fallback: analyze table names
        if 'customer' in ' '.join(tables).lower():
            return 'customer_management'
        elif 'order' in ' '.join(tables).lower():
            return 'order_processing'
        elif 'product' in ' '.join(tables).lower():
            return 'product_catalog'
        
        return f'community_{len(self.table_communities)}'
```

### Community-Based Query Routing

```python
# src/knowledge_graph/community_query_router.py
class CommunityAwareQueryRouter:
    """Route queries to relevant communities for focused analysis"""
    
    def __init__(self, community_detector: GraphCommunityDetector):
        self.communities = community_detector
        self.routing_patterns = self._build_routing_patterns()
        
    def route_query_to_communities(self, intent: Dict, 
                                 initial_concepts: List[str]) -> Dict[str, float]:
        """Route query to relevant table communities"""
        
        community_scores = {}
        
        # Score each community based on concept matches
        for community_name, tables in self.communities.table_communities.items():
            score = 0.0
            
            # Check concept presence in community
            for concept in initial_concepts:
                concept_lower = concept.lower()
                
                for table in tables:
                    table_data = self.communities.table_graph.nodes[table]
                    
                    # Check table name
                    if concept_lower in table.lower():
                        score += 0.4
                    
                    # Check key concepts
                    if concept_lower in ' '.join(table_data.get('key_concepts', [])).lower():
                        score += 0.3
                    
                    # Check column names
                    for col in table_data.get('measure_columns', []) + \
                              table_data.get('dimension_columns', []):
                        if concept_lower in col.lower():
                            score += 0.2
            
            # Boost score based on intent alignment
            if self._community_matches_intent(community_name, intent):
                score *= 1.5
            
            community_scores[community_name] = score
        
        # Normalize scores
        total = sum(community_scores.values())
        if total > 0:
            community_scores = {k: v/total for k, v in community_scores.items()}
        
        return community_scores
    
    def get_community_tables(self, community_name: str) -> List[str]:
        """Get all tables in a community"""
        return [
            table for table, comm in self.communities.table_communities.items()
            if comm == community_name
        ]
```

## 2. Hierarchical Table Selection Using Communities

```python
# src/agents/react_agents/hierarchical_table_selector.py
class HierarchicalTableSelector:
    """Select tables using community structure and semantic graph"""
    
    def __init__(self, table_graph: nx.MultiDiGraph, 
                 column_graph: nx.MultiDiGraph,
                 community_detector: GraphCommunityDetector):
        self.table_graph = table_graph
        self.column_graph = column_graph
        self.communities = community_detector
        self.router = CommunityAwareQueryRouter(community_detector)
        
    def select_tables_hierarchical(self, query: str, intent: Dict) -> List[Tuple[str, float, str]]:
        """Hierarchical table selection: Community → Table → Validation"""
        
        # Step 1: Route to communities
        community_scores = self.router.route_query_to_communities(
            intent, 
            intent.get('target_concepts', [])
        )
        
        print("🎯 Community Routing:")
        for comm, score in sorted(community_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"   - {comm}: {score:.3f}")
        
        # Step 2: Select tables within top communities
        table_candidates = []
        
        for community, comm_score in community_scores.items():
            if comm_score > 0.1:  # Threshold
                # Get tables in this community
                community_tables = self.router.get_community_tables(community)
                
                # Score tables within community
                for table in community_tables:
                    table_score = self._score_table_in_context(
                        table, query, intent, community
                    )
                    
                    # Combined score: community relevance × table relevance
                    final_score = comm_score * 0.4 + table_score * 0.6
                    
                    table_candidates.append((table, final_score, community))
        
        # Step 3: Validate using column-level insights
        validated_tables = self._validate_with_column_graph(table_candidates, intent)
        
        # Sort and return top tables
        validated_tables.sort(key=lambda x: x[1], reverse=True)
        return validated_tables[:3]
    
    def _validate_with_column_graph(self, candidates: List[Tuple[str, float, str]], 
                                   intent: Dict) -> List[Tuple[str, float, str]]:
        """Validate table selection using column-level graph"""
        
        validated = []
        
        for table, score, community in candidates:
            # Check if table has required column types for the intent
            validation_boost = self._validate_table_columns(table, intent)
            
            # Adjust score based on column validation
            adjusted_score = score * (0.7 + 0.3 * validation_boost)
            
            validated.append((table, adjusted_score, community))
        
        return validated
```

## 3. Column Graph Enhancement from Table Graph

```python
# src/knowledge_graph/bidirectional_enhancement.py
class BidirectionalGraphEnhancer:
    """Enhance column graph using table-level insights and vice versa"""
    
    def __init__(self, table_graph: nx.MultiDiGraph, column_graph: nx.MultiDiGraph):
        self.table_graph = table_graph
        self.column_graph = column_graph
        
    def enhance_column_graph_with_table_context(self):
        """Use table relationships to infer column relationships"""
        
        # For each table relationship, infer potential column relationships
        for source, target, data in self.table_graph.edges(data=True):
            rel_type = data['relationship_type']
            confidence = data['confidence']
            
            # Get columns from both tables
            source_cols = self._get_table_columns(source)
            target_cols = self._get_table_columns(target)
            
            # Infer column relationships based on table relationship
            if rel_type == TableRelationshipType.TEMPORAL_SEQUENCE:
                # Temporal columns likely related
                self._infer_temporal_column_relationships(
                    source_cols, target_cols, confidence
                )
                
            elif rel_type == TableRelationshipType.ACTOR_ACTION:
                # ID columns likely connected
                self._infer_actor_action_relationships(
                    source_cols, target_cols, source, target, confidence
                )
                
            elif rel_type == TableRelationshipType.HIERARCHICAL:
                # Hierarchical key relationships
                self._infer_hierarchical_relationships(
                    source_cols, target_cols, confidence
                )
    
    def enhance_table_graph_with_column_patterns(self):
        """Use column-level patterns to discover table relationships"""
        
        # Analyze cross-table column relationships
        cross_table_patterns = self._analyze_cross_table_patterns()
        
        for pattern in cross_table_patterns:
            if pattern['confidence'] > 0.7:
                # Add or strengthen table relationship
                self.table_graph.add_edge(
                    pattern['source_table'],
                    pattern['target_table'],
                    relationship_type=pattern['inferred_relationship'],
                    confidence=pattern['confidence'],
                    evidence=[{
                        'method': 'column_pattern_analysis',
                        'pattern': pattern['pattern_type'],
                        'supporting_columns': pattern['columns']
                    }],
                    semantic_description=pattern['description']
                )
    
    def _analyze_cross_table_patterns(self) -> List[Dict]:
        """Analyze patterns in column relationships across tables"""
        
        patterns = []
        
        # Group column relationships by table pairs
        table_pair_relationships = {}
        
        for source, target, data in self.column_graph.edges(data=True):
            if data.get('relationship') in ['FOREIGN_KEY', 'CORRELATED', 'SAME_DOMAIN']:
                source_table = self._extract_table_from_column(source)
                target_table = self._extract_table_from_column(target)
                
                if source_table != target_table:
                    key = (source_table, target_table)
                    if key not in table_pair_relationships:
                        table_pair_relationships[key] = []
                    
                    table_pair_relationships[key].append({
                        'source_col': source,
                        'target_col': target,
                        'relationship': data['relationship'],
                        'weight': data.get('weight', 0.5)
                    })
        
        # Analyze patterns for each table pair
        for (t1, t2), relationships in table_pair_relationships.items():
            pattern = self._classify_relationship_pattern(relationships)
            if pattern:
                patterns.append({
                    'source_table': t1,
                    'target_table': t2,
                    'pattern_type': pattern['type'],
                    'confidence': pattern['confidence'],
                    'inferred_relationship': pattern['table_relationship'],
                    'description': pattern['description'],
                    'columns': [(r['source_col'], r['target_col']) for r in relationships]
                })
        
        return patterns
```

## 4. Integrated Query Planning with Both Graphs

```python
# src/agents/react_agents/integrated_graph_planner.py
class IntegratedGraphQueryPlanner:
    """Query planner using both table and column graphs synergistically"""
    
    def __init__(self, table_graph: nx.MultiDiGraph, 
                 column_graph: nx.MultiDiGraph,
                 community_detector: GraphCommunityDetector):
        self.table_graph = table_graph
        self.column_graph = column_graph
        self.communities = community_detector
        self.table_selector = HierarchicalTableSelector(
            table_graph, column_graph, community_detector
        )
        self.enhancer = BidirectionalGraphEnhancer(table_graph, column_graph)
        
    def plan_query_integrated(self, query: str, intent: Dict) -> Dict[str, Any]:
        """Plan query using full graph intelligence"""
        
        # Step 1: Hierarchical table selection
        selected_tables = self.table_selector.select_tables_hierarchical(query, intent)
        
        print("\n📊 Selected Tables:")
        for table, score, community in selected_tables:
            print(f"   - {table} (score: {score:.3f}, community: {community})")
        
        # Step 2: Find optimal column mappings within selected tables
        column_mappings = self._map_concepts_to_columns_focused(
            intent['target_concepts'],
            [t[0] for t in selected_tables]
        )
        
        # Step 3: Determine join strategy using both graphs
        join_strategy = self._determine_join_strategy_integrated(
            selected_tables,
            column_mappings
        )
        
        # Step 4: Identify analysis opportunities from community structure
        analysis_suggestions = self._suggest_analysis_from_communities(
            selected_tables,
            intent
        )
        
        return {
            'selected_tables': selected_tables,
            'column_mappings': column_mappings,
            'join_strategy': join_strategy,
            'analysis_suggestions': analysis_suggestions,
            'community_context': self._get_community_context(selected_tables)
        }
    
    def _determine_join_strategy_integrated(self, tables: List[Tuple[str, float, str]], 
                                          column_mappings: Dict) -> Dict:
        """Determine joins using both table and column relationships"""
        
        table_names = [t[0] for t in tables]
        
        # Get table-level suggested paths
        table_paths = nx.shortest_path(self.table_graph, table_names[0], table_names[-1])
        
        # Validate with column-level relationships
        join_conditions = []
        
        for i in range(len(table_paths) - 1):
            t1, t2 = table_paths[i], table_paths[i + 1]
            
            # Find best join columns
            best_join = self._find_best_join_columns(t1, t2)
            
            if best_join:
                join_conditions.append({
                    'from_table': t1,
                    'to_table': t2,
                    'from_column': best_join['from_col'],
                    'to_column': best_join['to_col'],
                    'confidence': best_join['confidence'],
                    'type': best_join['type']
                })
        
        return {
            'table_path': table_paths,
            'join_conditions': join_conditions,
            'strategy_confidence': np.mean([j['confidence'] for j in join_conditions])
        }
```

## Benefits of This Integrated Approach

1. **Faster Table Selection**: Community clustering reduces search space from all tables to relevant communities
2. **Better Context Understanding**: Table relationships inform column relationship discovery
3. **Hierarchical Reasoning**: Community → Table → Column provides natural drill-down
4. **Pattern Discovery**: Communities reveal hidden business domains and analysis patterns
5. **Confidence Propagation**: Table-level confidence informs column-level decisions

## Example Flow

```python
# Query: "Show customer purchase patterns by product category"

# 1. Community Detection finds:
#    - "customer_management" community: [customers, customer_segments, loyalty_program]
#    - "order_processing" community: [orders, order_items, payments]
#    - "product_catalog" community: [products, categories, inventory]

# 2. Query routes to relevant communities:
#    - customer_management: 0.4
#    - order_processing: 0.35
#    - product_catalog: 0.25

# 3. Within communities, select best tables:
#    - customers (0.85) from customer_management
#    - orders (0.82) from order_processing
#    - products (0.78) from product_catalog

# 4. Column graph validates selections:
#    - customers has customer_id (identifier)
#    - orders has customer_id (FK), product_id (FK), amount (measure)
#    - products has product_id (identifier), category_id (dimension)

# 5. Result: Optimal path with high confidence
```

This integrated approach creates a powerful synergy where:
- **Table graph** provides semantic understanding and business context
- **Column graph** provides detailed relationships and join paths
- **Communities** provide focused search spaces and domain understanding
- Together, they enable highly accurate and efficient query planning
============================================================================================================
Excellent question! Yes, we can definitely create a sophisticated table-level knowledge graph similar to what Neo4j does with their text graphs. This would be a powerful addition that captures semantic relationships between tables, not just structural ones. Let me design this for you.

## Enhanced Table-Level Knowledge Graph

### 1. **Table Relationship Types**

First, let's define the types of relationships between tables:

```python
# src/knowledge_graph/table_relationships.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class TableRelationshipType(Enum):
    """Types of semantic relationships between tables"""
    
    # Structural Relationships
    FOREIGN_KEY = "foreign_key"                    # Traditional FK relationship
    JUNCTION = "junction"                          # Many-to-many bridge table
    
    # Semantic Relationships
    TEMPORAL_SEQUENCE = "temporal_sequence"        # Events in time order (orders → shipments)
    HIERARCHICAL = "hierarchical"                  # Parent-child (categories → subcategories)
    SUPPLEMENTARY = "supplementary"                # Additional info (products → reviews)
    DIMENSIONAL = "dimensional"                    # Fact-dimension relationship
    AGGREGATION = "aggregation"                    # Detail-summary (transactions → daily_summary)
    
    # Business Process Relationships
    WORKFLOW = "workflow"                          # Business process flow
    ACTOR_ACTION = "actor_action"                  # Entity performs action (customers → orders)
    RESOURCE_USAGE = "resource_usage"              # Entity uses resource (orders → inventory)
    
    # Analytical Relationships
    COMPARATIVE = "comparative"                    # Tables for comparison (actual vs budget)
    VERSIONED = "versioned"                        # Same entity, different versions
    PARTITIONED = "partitioned"                    # Same structure, split by criteria

@dataclass
class TableRelationship:
    """Rich relationship between tables"""
    source_table: str
    target_table: str
    relationship_type: TableRelationshipType
    confidence: float
    evidence: List[Dict[str, Any]]
    semantic_description: str
    bidirectional: bool = False
    strength: float = 1.0
    
    # Analytical properties
    join_cardinality: Optional[str] = None  # "1:1", "1:N", "N:M"
    typical_join_columns: List[Tuple[str, str]] = None
    business_meaning: Optional[str] = None
```

### 2. **Semantic Table Graph Builder**

```python
# src/knowledge_graph/semantic_table_graph.py
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticTableGraphBuilder:
    """Build a semantic graph of table relationships"""
    
    def __init__(self, knowledge_graph: nx.MultiDiGraph, schema_manager):
        self.kg = knowledge_graph
        self.schema = schema_manager.schema
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create separate table graph
        self.table_graph = nx.MultiDiGraph()
        
        # Relationship detectors
        self.detectors = {
            'structural': StructuralRelationshipDetector(),
            'semantic': SemanticRelationshipDetector(),
            'temporal': TemporalRelationshipDetector(),
            'business': BusinessProcessDetector()
        }
    
    def build_table_graph(self) -> nx.MultiDiGraph:
        """Build comprehensive table relationship graph"""
        
        # Step 1: Add table nodes with rich metadata
        self._add_table_nodes()
        
        # Step 2: Detect various relationship types
        relationships = []
        
        # Structural relationships (from existing KG)
        relationships.extend(self._extract_structural_relationships())
        
        # Semantic relationships (from embeddings)
        relationships.extend(self._detect_semantic_relationships())
        
        # Temporal relationships
        relationships.extend(self._detect_temporal_relationships())
        
        # Business process relationships
        relationships.extend(self._detect_business_relationships())
        
        # Step 3: Add relationships to graph
        for rel in relationships:
            self._add_relationship_to_graph(rel)
        
        # Step 4: Compute graph analytics
        self._compute_table_importance()
        
        return self.table_graph
    
    def _add_table_nodes(self):
        """Add table nodes with semantic embeddings and metadata"""
        
        for table_name, table_schema in self.schema.tables.items():
            # Create rich table representation
            table_desc = self._create_table_description(table_name, table_schema)
            
            # Generate embedding
            embedding = self.encoder.encode(table_desc['text'])
            
            # Add node with comprehensive metadata
            self.table_graph.add_node(
                table_name,
                embedding=embedding,
                description=table_desc['text'],
                semantic_summary=table_desc['summary'],
                key_concepts=table_desc['concepts'],
                column_count=len(table_schema.columns),
                row_count=table_desc.get('row_count', 0),
                business_domain=table_schema.business_domain,
                table_type=self._classify_table_type(table_schema),
                measure_columns=[c for c, s in table_schema.columns.items() 
                               if s.semantic_role == SemanticRole.MEASURE],
                dimension_columns=[c for c, s in table_schema.columns.items() 
                                 if s.semantic_role == SemanticRole.DIMENSION]
            )
    
    def _detect_semantic_relationships(self) -> List[TableRelationship]:
        """Detect relationships using semantic similarity"""
        relationships = []
        
        tables = list(self.schema.tables.keys())
        
        # Compute embedding similarities
        embeddings = [self.table_graph.nodes[t]['embedding'] for t in tables]
        similarity_matrix = cosine_similarity(embeddings)
        
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables[i+1:], i+1):
                sim_score = similarity_matrix[i][j]
                
                if sim_score > 0.7:  # High semantic similarity
                    # Determine relationship type based on content
                    rel_type = self._classify_semantic_relationship(
                        table1, table2, sim_score
                    )
                    
                    if rel_type:
                        relationships.append(TableRelationship(
                            source_table=table1,
                            target_table=table2,
                            relationship_type=rel_type,
                            confidence=sim_score,
                            evidence=[{
                                'method': 'semantic_similarity',
                                'score': sim_score
                            }],
                            semantic_description=f"{table1} and {table2} share similar semantic content",
                            bidirectional=True
                        ))
        
        return relationships
    
    def _classify_semantic_relationship(self, table1: str, table2: str, 
                                      similarity: float) -> Optional[TableRelationshipType]:
        """Classify the type of semantic relationship"""
        
        t1_schema = self.schema.tables[table1]
        t2_schema = self.schema.tables[table2]
        
        # Check for supplementary relationship
        if self._is_supplementary(t1_schema, t2_schema):
            return TableRelationshipType.SUPPLEMENTARY
        
        # Check for hierarchical
        if self._is_hierarchical(t1_schema, t2_schema):
            return TableRelationshipType.HIERARCHICAL
        
        # Check for versioned
        if similarity > 0.9 and self._tables_have_similar_structure(t1_schema, t2_schema):
            return TableRelationshipType.VERSIONED
        
        return None
    
    def _detect_temporal_relationships(self) -> List[TableRelationship]:
        """Detect temporal/workflow relationships"""
        relationships = []
        
        # Find tables with temporal columns
        temporal_tables = {}
        for table_name, table_schema in self.schema.tables.items():
            temporal_cols = [
                col for col, schema in table_schema.columns.items()
                if schema.semantic_role == SemanticRole.TEMPORAL
            ]
            if temporal_cols:
                temporal_tables[table_name] = temporal_cols
        
        # Analyze temporal patterns
        for t1, cols1 in temporal_tables.items():
            for t2, cols2 in temporal_tables.items():
                if t1 != t2:
                    # Check if there's a temporal flow
                    if self._has_temporal_flow(t1, t2, cols1, cols2):
                        relationships.append(TableRelationship(
                            source_table=t1,
                            target_table=t2,
                            relationship_type=TableRelationshipType.TEMPORAL_SEQUENCE,
                            confidence=0.8,
                            evidence=[{
                                'method': 'temporal_analysis',
                                'temporal_columns': {t1: cols1, t2: cols2}
                            }],
                            semantic_description=f"{t1} events typically occur before {t2}",
                            bidirectional=False
                        ))
        
        return relationships
    
    def _detect_business_relationships(self) -> List[TableRelationship]:
        """Detect business process relationships"""
        relationships = []
        
        # Actor-Action patterns
        actor_tables = self._find_actor_tables()  # customers, users, employees
        action_tables = self._find_action_tables()  # orders, transactions, activities
        
        for actor in actor_tables:
            for action in action_tables:
                if self._has_actor_action_relationship(actor, action):
                    relationships.append(TableRelationship(
                        source_table=actor,
                        target_table=action,
                        relationship_type=TableRelationshipType.ACTOR_ACTION,
                        confidence=0.85,
                        evidence=[{
                            'method': 'pattern_matching',
                            'pattern': 'actor_performs_action'
                        }],
                        semantic_description=f"{actor} performs actions recorded in {action}",
                        bidirectional=False,
                        business_meaning=f"Business process: {actor} → {action}"
                    ))
        
        return relationships
```

### 3. **Graph-Enhanced Query Planning**

```python
# src/knowledge_graph/table_graph_navigator.py
class TableGraphNavigator:
    """Navigate the semantic table graph for query planning"""
    
    def __init__(self, table_graph: nx.MultiDiGraph):
        self.graph = table_graph
        
    def find_query_subgraph(self, seed_tables: List[str], 
                          expansion_depth: int = 2) -> nx.DiGraph:
        """Find relevant subgraph for query planning"""
        
        subgraph_nodes = set(seed_tables)
        
        # Expand based on relationship strength and type
        for depth in range(expansion_depth):
            new_nodes = set()
            
            for node in subgraph_nodes:
                # Get neighbors with strong relationships
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    
                    # Evaluate if this table should be included
                    if self._should_include_table(edge_data, depth):
                        new_nodes.add(neighbor)
            
            subgraph_nodes.update(new_nodes)
        
        return self.graph.subgraph(subgraph_nodes)
    
    def suggest_analysis_paths(self, intent: Dict) -> List[Dict[str, Any]]:
        """Suggest analysis paths based on graph structure"""
        
        paths = []
        action_type = intent.get('action_type')
        
        if action_type == 'trend_analysis':
            # Find temporal sequences
            temporal_paths = self._find_temporal_paths()
            paths.extend(temporal_paths)
            
        elif action_type == 'aggregation':
            # Find fact-dimension relationships
            fact_dim_paths = self._find_fact_dimension_paths()
            paths.extend(fact_dim_paths)
            
        elif action_type == 'comparison':
            # Find comparable tables
            comparison_paths = self._find_comparison_paths()
            paths.extend(comparison_paths)
        
        return paths
    
    def explain_table_relationship(self, table1: str, table2: str) -> str:
        """Generate human-readable explanation of relationship"""
        
        if not self.graph.has_edge(table1, table2):
            return f"No direct relationship found between {table1} and {table2}"
        
        edge_data = self.graph.get_edge_data(table1, table2)
        relationships = []
        
        for key, data in edge_data.items():
            rel_type = data['relationship_type']
            confidence = data['confidence']
            description = data.get('semantic_description', '')
            
            relationships.append(
                f"- {rel_type.value}: {description} (confidence: {confidence:.2f})"
            )
        
        return f"Relationships between {table1} and {table2}:\n" + "\n".join(relationships)
```

### 4. **Integration with Query Planning**

```python
# src/agents/react_agents/table_graph_aware_planner.py
class TableGraphAwareQueryPlanner:
    """Query planner that leverages the semantic table graph"""
    
    def __init__(self, table_graph: nx.MultiDiGraph, table_navigator: TableGraphNavigator):
        self.table_graph = table_graph
        self.navigator = table_navigator
        
    def enhance_table_selection(self, intent: Dict, initial_tables: List[str]) -> List[str]:
        """Enhance table selection using graph intelligence"""
        
        # Find connected tables that might be useful
        subgraph = self.navigator.find_query_subgraph(initial_tables)
        
        # Score additional tables based on relationship types
        scored_tables = []
        
        for table in subgraph.nodes():
            if table not in initial_tables:
                score = self._score_table_relevance(table, initial_tables, intent)
                scored_tables.append((table, score))
        
        # Add high-scoring tables
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        enhanced_tables = initial_tables.copy()
        
        for table, score in scored_tables[:2]:  # Add up to 2 more tables
            if score > 0.7:
                enhanced_tables.append(table)
                print(f"   📊 Graph suggests including: {table} (score: {score:.2f})")
        
        return enhanced_tables
    
    def _score_table_relevance(self, table: str, seed_tables: List[str], 
                              intent: Dict) -> float:
        """Score table relevance based on relationships"""
        
        score = 0.0
        
        # Check relationships to seed tables
        for seed in seed_tables:
            if self.table_graph.has_edge(seed, table):
                edge_data = self.table_graph.get_edge_data(seed, table)
                
                for _, data in edge_data.items():
                    rel_type = data['relationship_type']
                    confidence = data['confidence']
                    
                    # Weight by relationship type relevance to intent
                    type_weight = self._get_relationship_weight(rel_type, intent)
                    score += confidence * type_weight
        
        # Consider table properties
        node_data = self.table_graph.nodes[table]
        if intent['action_type'] == 'aggregation' and node_data['measure_columns']:
            score += 0.2
        
        return min(score, 1.0)
```

### 5. **Visual Graph Analysis**

```python
# src/knowledge_graph/table_graph_visualizer.py
class TableGraphVisualizer:
    """Visualize the semantic table graph"""
    
    def visualize_semantic_graph(self, table_graph: nx.MultiDiGraph):
        """Create an interactive visualization"""
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create 3D layout for better relationship visibility
        pos = nx.spring_layout(table_graph, dim=3, k=5, iterations=50)
        
        # Create traces for different relationship types
        edge_traces = {}
        
        for rel_type in TableRelationshipType:
            edge_traces[rel_type] = {
                'x': [], 'y': [], 'z': [],
                'mode': 'lines',
                'name': rel_type.value,
                'line': {
                    'width': 3,
                    'color': self._get_relationship_color(rel_type)
                }
            }
        
        # Add edges by type
        for edge in table_graph.edges(data=True):
            source, target, data = edge
            rel_type = data['relationship_type']
            
            x0, y0, z0 = pos[source]
            x1, y1, z1 = pos[target]
            
            edge_traces[rel_type]['x'].extend([x0, x1, None])
            edge_traces[rel_type]['y'].extend([y0, y1, None])
            edge_traces[rel_type]['z'].extend([z0, z1, None])
        
        # Create node trace
        node_trace = {
            'x': [pos[node][0] for node in table_graph.nodes()],
            'y': [pos[node][1] for node in table_graph.nodes()],
            'z': [pos[node][2] for node in table_graph.nodes()],
            'mode': 'markers+text',
            'text': list(table_graph.nodes()),
            'marker': {
                'size': [self._get_node_size(table_graph.nodes[n]) 
                        for n in table_graph.nodes()],
                'color': [self._get_node_color(table_graph.nodes[n]) 
                         for n in table_graph.nodes()]
            }
        }
        
        # Create figure
        fig = go.Figure(
            data=[go.Scatter3d(**trace) for trace in edge_traces.values()] + 
                 [go.Scatter3d(**node_trace)],
            layout=go.Layout(
                title="Semantic Table Relationship Graph",
                showlegend=True,
                scene=dict(
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    zaxis=dict(showgrid=False)
                )
            )
        )
        
        return fig
```

## Benefits of This Approach

1. **Rich Relationship Understanding**: Goes beyond simple foreign keys to understand business meaning
2. **Query Path Discovery**: The graph can suggest non-obvious but valuable table combinations
3. **Semantic Navigation**: Tables are connected by meaning, not just structure
4. **Explainable Decisions**: Can explain why certain tables are recommended
5. **Pattern Discovery**: Can identify common analysis patterns in your data

## Example Usage

```python
# Build the semantic table graph
table_graph_builder = SemanticTableGraphBuilder(knowledge_graph, schema_manager)
semantic_table_graph = table_graph_builder.build_table_graph()

# Use in query planning
navigator = TableGraphNavigator(semantic_table_graph)

# For a query like "Compare customer behavior before and after promotions"
intent = {'action_type': 'comparison', 'concepts': ['customer', 'promotion']}
initial_tables = ['customers', 'promotions']

# The graph might suggest:
# - 'orders' (ACTOR_ACTION: customers → orders)
# - 'order_items' (SUPPLEMENTARY: orders → order_items)  
# - 'promotion_usage' (TEMPORAL_SEQUENCE: promotions → usage)

suggested_paths = navigator.suggest_analysis_paths(intent)
```

This semantic table graph provides a powerful abstraction layer that helps the system understand not just what tables exist, but how they relate to each other in meaningful business contexts. This dramatically improves table selection accuracy and enables more sophisticated query planning.
