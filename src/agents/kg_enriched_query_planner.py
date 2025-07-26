"""
Knowledge Graph Enriched Query Planner

This planner leverages the ML-quantified relationships in the knowledge graph
to build enriched context for the LLM, rather than sending raw schema.

Architecture:
1. Knowledge Graph = Complete relationship map with ML weights
2. Extract relationship-enriched context from KG  
3. Send intelligent context to LLM (not raw schema)
4. LLM makes optimal decisions with complete relationship picture
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json
from collections import defaultdict

try:
    from ..schema.schema_manager import DatasetSchema, ColumnSchema, SemanticRole
except ImportError:
    from schema.schema_manager import DatasetSchema, ColumnSchema, SemanticRole


@dataclass
class RelationshipContext:
    """Enriched relationship context extracted from knowledge graph"""
    strong_relationships: List[Dict[str, Any]]  # High-weight relationships
    concept_clusters: Dict[str, List[str]]      # Columns grouped by business concept
    join_paths: List[Dict[str, Any]]            # Optimal join paths with weights
    measure_dimension_pairs: List[Dict[str, Any]]  # Analysis-ready combinations
    temporal_relationships: List[Dict[str, Any]]   # Time-based analysis options


class KnowledgeGraphContextExtractor:
    """Extracts enriched context from the knowledge graph"""
    
    def __init__(self, knowledge_graph: nx.MultiDiGraph, schema: DatasetSchema):
        self.graph = knowledge_graph
        self.schema = schema
    
    def extract_relationship_context(self, focus_query: str = None) -> RelationshipContext:
        """
        Extract enriched relationship context from knowledge graph
        Uses ML weights to identify the most meaningful relationships
        """
        
        # 1. Extract strong relationships (high ML weights)
        strong_relationships = self._extract_strong_relationships()
        
        # 2. Identify concept clusters using relationship weights
        concept_clusters = self._identify_concept_clusters()
        
        # 3. Find optimal join paths with cumulative weights
        join_paths = self._find_optimal_join_paths()
        
        # 4. Identify measure-dimension pairs for analysis
        measure_dimension_pairs = self._find_analysis_pairs()
        
        # 5. Extract temporal analysis opportunities
        temporal_relationships = self._extract_temporal_context()
        
        return RelationshipContext(
            strong_relationships=strong_relationships,
            concept_clusters=concept_clusters,
            join_paths=join_paths,
            measure_dimension_pairs=measure_dimension_pairs,
            temporal_relationships=temporal_relationships
        )
    
    def _extract_strong_relationships(self, min_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Extract relationships with high ML confidence weights"""
        strong_rels = []
        
        for source, target, data in self.graph.edges(data=True):
            weight = data.get('weight', 0.0)
            rel_type = data.get('relationship', 'UNKNOWN')
            
            if weight >= min_weight and source.startswith('COLUMN:') and target.startswith('COLUMN:'):
                # Extract readable column names
                source_col = self._extract_column_name(source)
                target_col = self._extract_column_name(target)
                
                if source_col and target_col:
                    strong_rels.append({
                        'from': source_col,
                        'to': target_col,
                        'type': rel_type,
                        'weight': weight,
                        'evidence': data.get('evidence', {}),
                        'business_meaning': self._interpret_relationship(source_col, target_col, rel_type)
                    })
        
        # Sort by weight descending
        return sorted(strong_rels, key=lambda x: x['weight'], reverse=True)
    
    def _identify_concept_clusters(self) -> Dict[str, List[str]]:
        """
        Group columns into business concept clusters using relationship weights
        """
        clusters = defaultdict(list)
        
        # Use community detection on the column subgraph
        column_nodes = [n for n in self.graph.nodes() if n.startswith('COLUMN:')]
        column_subgraph = self.graph.subgraph(column_nodes)
        
        # Simple clustering based on table names and relationship strength
        for node in column_nodes:
            col_name = self._extract_column_name(node)
            if col_name:
                table_name = col_name.split('.')[0]
                
                # Determine business concept from table name and relationships
                concept = self._determine_business_concept(node, table_name)
                clusters[concept].append(col_name)
        
        return dict(clusters)
    
    def _find_optimal_join_paths(self) -> List[Dict[str, Any]]:
        """
        Find optimal paths between tables using cumulative ML weights
        """
        join_paths = []
        
        # Get all table pairs
        tables = set()
        for node in self.graph.nodes():
            if node.startswith('COLUMN:'):
                col_name = self._extract_column_name(node)
                if col_name:
                    tables.add(col_name.split('.')[0])
        
        tables = list(tables)
        
        # Find paths between each table pair
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                path_info = self._find_best_table_path(table1, table2)
                if path_info:
                    join_paths.append(path_info)
        
        # Sort by path strength
        return sorted(join_paths, key=lambda x: x['total_weight'], reverse=True)
    
    def _find_analysis_pairs(self) -> List[Dict[str, Any]]:
        """
        Find measure-dimension pairs that are well-connected for analysis
        """
        pairs = []
        
        # Get measures and dimensions from schema
        measures = []
        dimensions = []
        
        for table_name, table_schema in self.schema.tables.items():
            for col_name, col_schema in table_schema.columns.items():
                full_name = f"{table_name}.{col_name}"
                
                if col_schema.semantic_role == SemanticRole.MEASURE:
                    measures.append(full_name)
                elif col_schema.semantic_role == SemanticRole.DIMENSION:
                    dimensions.append(full_name)
        
        # Find well-connected measure-dimension pairs
        for measure in measures:
            for dimension in dimensions:
                connection_strength = self._calculate_connection_strength(measure, dimension)
                if connection_strength > 0.5:
                    pairs.append({
                        'measure': measure,
                        'dimension': dimension,
                        'strength': connection_strength,
                        'analysis_type': self._suggest_analysis_type(measure, dimension)
                    })
        
        return sorted(pairs, key=lambda x: x['strength'], reverse=True)
    
    def _extract_temporal_context(self) -> List[Dict[str, Any]]:
        """
        Extract temporal analysis opportunities
        """
        temporal_context = []
        
        # Find temporal columns
        temporal_columns = []
        for table_name, table_schema in self.schema.tables.items():
            for col_name, col_schema in table_schema.columns.items():
                if col_schema.semantic_role == SemanticRole.TEMPORAL:
                    temporal_columns.append(f"{table_name}.{col_name}")
        
        # Find what measures are connected to each temporal column
        for temp_col in temporal_columns:
            connected_measures = self._find_connected_measures(temp_col)
            if connected_measures:
                temporal_context.append({
                    'temporal_column': temp_col,
                    'connected_measures': connected_measures,
                    'analysis_opportunities': ['trend', 'seasonality', 'growth']
                })
        
        return temporal_context
    
    def _extract_column_name(self, node: str) -> Optional[str]:
        """Extract readable column name from graph node"""
        if node.startswith('COLUMN:'):
            parts = node.split('.')
            if len(parts) >= 2:
                return f"{parts[-2]}.{parts[-1]}"
        return None
    
    def _interpret_relationship(self, col1: str, col2: str, rel_type: str) -> str:
        """Provide business interpretation of relationship"""
        interpretations = {
            'FOREIGN_KEY': f"{col1} references {col2} (master-detail relationship)",
            'CORRELATED': f"{col1} and {col2} show statistical correlation (move together)",
            'SAME_DOMAIN': f"{col1} and {col2} represent similar business concepts",
            'INFORMATION_DEPENDENCY': f"{col2} provides context for {col1}",
            'TEMPORAL_SEQUENCE': f"{col1} occurs before {col2} in business process"
        }
        return interpretations.get(rel_type, f"{col1} related to {col2}")
    
    def _determine_business_concept(self, node: str, table_name: str) -> str:
        """Determine business concept from node context"""
        # Simple heuristic based on table name
        concept_mappings = {
            'customer': 'Customer Management',
            'order': 'Order Processing', 
            'product': 'Product Catalog',
            'seller': 'Seller Network',
            'payment': 'Financial Transactions',
            'review': 'Customer Feedback'
        }
        
        for key, concept in concept_mappings.items():
            if key in table_name.lower():
                return concept
        
        return 'General Data'
    
    def _find_best_table_path(self, table1: str, table2: str) -> Optional[Dict[str, Any]]:
        """Find the strongest path between two tables"""
        # Find column nodes for each table
        table1_columns = [n for n in self.graph.nodes() 
                         if n.startswith('COLUMN:') and f'.{table1}.' in n]
        table2_columns = [n for n in self.graph.nodes() 
                         if n.startswith('COLUMN:') and f'.{table2}.' in n]
        
        best_path = None
        best_weight = 0
        
        # Find shortest weighted path between any columns in the tables
        for col1 in table1_columns[:3]:  # Limit for performance
            for col2 in table2_columns[:3]:
                try:
                    path = nx.shortest_path(self.graph, col1, col2)
                    weight = self._calculate_path_weight(path)
                    
                    if weight > best_weight:
                        best_weight = weight
                        best_path = path
                except nx.NetworkXNoPath:
                    continue
        
        if best_path and best_weight > 0.3:
            return {
                'from_table': table1,
                'to_table': table2,
                'path': [self._extract_column_name(n) for n in best_path if n.startswith('COLUMN:')],
                'total_weight': best_weight,
                'join_recommendation': self._generate_join_recommendation(best_path)
            }
        
        return None
    
    def _calculate_connection_strength(self, col1: str, col2: str) -> float:
        """Calculate connection strength between two columns"""
        # Convert to graph nodes
        node1 = None
        node2 = None
        
        for node in self.graph.nodes():
            if node.endswith(col1.replace('.', '.')):
                node1 = node
            elif node.endswith(col2.replace('.', '.')):
                node2 = node
        
        if not node1 or not node2:
            return 0.0
        
        try:
            path = nx.shortest_path(self.graph, node1, node2)
            return self._calculate_path_weight(path)
        except nx.NetworkXNoPath:
            return 0.0
    
    def _calculate_path_weight(self, path: List[str]) -> float:
        """Calculate cumulative weight of a path"""
        if len(path) < 2:
            return 0.0
        
        total_weight = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                if isinstance(edge_data, dict) and 0 in edge_data:
                    weight = edge_data[0].get('weight', 0.5)
                else:
                    weight = edge_data.get('weight', 0.5)
                total_weight *= weight
            else:
                total_weight *= 0.1
        
        return total_weight
    
    def _suggest_analysis_type(self, measure: str, dimension: str) -> str:
        """Suggest analysis type for measure-dimension pair"""
        if 'date' in dimension.lower() or 'time' in dimension.lower():
            return 'trend_analysis'
        elif 'location' in dimension.lower() or 'city' in dimension.lower():
            return 'geographical_analysis'
        elif 'category' in dimension.lower() or 'type' in dimension.lower():
            return 'categorical_breakdown'
        else:
            return 'comparative_analysis'
    
    def _find_connected_measures(self, temporal_col: str) -> List[str]:
        """Find measures connected to a temporal column"""
        measures = []
        
        # Find measures within 2 hops of temporal column
        temp_node = None
        for node in self.graph.nodes():
            if node.endswith(temporal_col.replace('.', '.')):
                temp_node = node
                break
        
        if temp_node:
            # Get nodes within distance 2
            try:
                nearby_nodes = nx.single_source_shortest_path(self.graph, temp_node, cutoff=2)
                for node in nearby_nodes:
                    col_name = self._extract_column_name(node)
                    if col_name:
                        table, col = col_name.split('.', 1)
                        table_schema = self.schema.tables.get(table)
                        if table_schema:
                            col_schema = table_schema.columns.get(col)
                            if col_schema and col_schema.semantic_role == SemanticRole.MEASURE:
                                measures.append(col_name)
            except:
                pass
        
        return measures
    
    def _generate_join_recommendation(self, path: List[str]) -> str:
        """Generate SQL join recommendation from path"""
        column_path = [self._extract_column_name(n) for n in path if n.startswith('COLUMN:')]
        if len(column_path) >= 2:
            return f"JOIN via: {' → '.join(column_path)}"
        return "Direct relationship"


class EnrichedContextLLMAnalyzer:
    """LLM analyzer that receives enriched context instead of raw schema"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
    
    def analyze_with_enriched_context(self, query: str, 
                                    relationship_context: RelationshipContext) -> Dict[str, Any]:
        """
        Analyze query using enriched relationship context from knowledge graph
        This should use significantly fewer tokens than raw schema
        """
        
        # Build enriched context string
        context = self._build_enriched_context(relationship_context)
        
        # Optimized prompt focusing on relationships and analysis opportunities
        prompt = f"""Query: "{query}"

RELATIONSHIP INTELLIGENCE (from ML-analyzed knowledge graph):

{context}

Based on these discovered relationships and analysis opportunities, provide JSON:
{{
  "recommended_approach": "specific analysis strategy",
  "selected_columns": ["col1", "col2"],
  "join_strategy": "how to connect data",
  "analysis_type": "aggregation|trend|comparison",
  "confidence": 0.9
}}

Focus on leveraging the strongest relationships and most promising analysis opportunities."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            result = json.loads(response.content)
            # Add token estimation
            result['estimated_tokens'] = len(prompt.split()) + len(response.content.split())
            return result
        except:
            return {
                "recommended_approach": "general analysis",
                "selected_columns": [],
                "join_strategy": "auto-detect",
                "analysis_type": "general",
                "confidence": 0.5,
                "estimated_tokens": len(prompt.split()) + 100
            }
    
    def _build_enriched_context(self, context: RelationshipContext) -> str:
        """Build enriched context string from relationship analysis"""
        sections = []
        
        # Strong relationships
        if context.strong_relationships:
            sections.append("STRONG RELATIONSHIPS (ML confidence > 0.7):")
            for rel in context.strong_relationships[:5]:
                sections.append(f"• {rel['business_meaning']} (weight: {rel['weight']:.2f})")
        
        # Concept clusters
        if context.concept_clusters:
            sections.append("\nBUSINESS CONCEPT CLUSTERS:")
            for concept, columns in list(context.concept_clusters.items())[:3]:
                sections.append(f"• {concept}: {', '.join(columns[:4])}")
        
        # Analysis-ready pairs
        if context.measure_dimension_pairs:
            sections.append("\nREADY-TO-ANALYZE COMBINATIONS:")
            for pair in context.measure_dimension_pairs[:3]:
                sections.append(f"• {pair['measure']} by {pair['dimension']} "
                               f"(strength: {pair['strength']:.2f}, type: {pair['analysis_type']})")
        
        # Join paths
        if context.join_paths:
            sections.append("\nOPTIMAL JOIN PATHS:")
            for path in context.join_paths[:3]:
                sections.append(f"• {path['from_table']} ↔ {path['to_table']} "
                               f"(weight: {path['total_weight']:.2f}) - {path['join_recommendation']}")
        
        # Temporal opportunities
        if context.temporal_relationships:
            sections.append("\nTIME-BASED ANALYSIS OPPORTUNITIES:")
            for temp in context.temporal_relationships[:2]:
                measures = ', '.join(temp['connected_measures'][:3])
                sections.append(f"• {temp['temporal_column']} → {measures}")
        
        return "\n".join(sections)


class KGEnrichedQueryPlanner:
    """
    Main query planner using knowledge graph enriched context
    """
    
    def __init__(self, schema: DatasetSchema, knowledge_graph: nx.MultiDiGraph):
        self.schema = schema
        self.knowledge_graph = knowledge_graph
        self.context_extractor = KnowledgeGraphContextExtractor(knowledge_graph, schema)
        self.llm_analyzer = EnrichedContextLLMAnalyzer()
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """
        Plan query using enriched knowledge graph context
        Should use ~70% fewer tokens than raw schema approach
        """
        
        print("🧠 Extracting enriched context from knowledge graph...")
        
        # Step 1: Extract enriched context from knowledge graph
        relationship_context = self.context_extractor.extract_relationship_context(query)
        
        print(f"  Found {len(relationship_context.strong_relationships)} strong relationships")
        print(f"  Found {len(relationship_context.concept_clusters)} concept clusters")
        print(f"  Found {len(relationship_context.join_paths)} join paths")
        
        # Step 2: Send enriched context to LLM (not raw schema)
        print("🤖 Analyzing with enriched context...")
        llm_result = self.llm_analyzer.analyze_with_enriched_context(query, relationship_context)
        
        print(f"  Estimated tokens used: {llm_result.get('estimated_tokens', 'unknown')}")
        
        # Step 3: Build final query plan
        plan = {
            'query': query,
            'llm_analysis': llm_result,
            'relationship_context': {
                'strong_relationships_count': len(relationship_context.strong_relationships),
                'concept_clusters': relationship_context.concept_clusters,
                'optimal_joins': relationship_context.join_paths[:3],
                'analysis_opportunities': relationship_context.measure_dimension_pairs[:5]
            },
            'recommended_columns': llm_result.get('selected_columns', []),
            'join_strategy': llm_result.get('join_strategy', ''),
            'confidence': llm_result.get('confidence', 0.5),
            'token_efficiency': 'enriched_context'
        }
        
        return plan