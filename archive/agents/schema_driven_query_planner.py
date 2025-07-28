"""
Schema-Driven Query Planner with Knowledge Graph Traversal

This agent dynamically adapts to any schema and uses the knowledge graph
to discover relationships when building queries.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json

try:
    from ..schema.schema_manager import DatasetSchema, ColumnSchema, SemanticRole
    from ..knowledge_graph.graph_builder import EnhancedKnowledgeGraphBuilder
except ImportError:
    from schema.schema_manager import DatasetSchema, ColumnSchema, SemanticRole
    from knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph"""
    start_column: str
    end_column: str
    path: List[str]
    relationships: List[Dict[str, Any]]
    total_weight: float
    
    def get_tables_in_path(self) -> List[str]:
        """Extract unique tables involved in this path"""
        tables = set()
        for node in self.path:
            if node.startswith('COLUMN:'):
                # Format: COLUMN:dataset.table.column
                parts = node.split('.')
                if len(parts) >= 2:
                    tables.add(parts[-2])
        return list(tables)


@dataclass
class QueryContext:
    """Context for query planning"""
    user_query: str
    schema: DatasetSchema
    knowledge_graph: nx.MultiDiGraph
    selected_columns: List[str] = None
    discovered_paths: List[GraphPath] = None
    llm_analysis: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.selected_columns is None:
            self.selected_columns = []
        if self.discovered_paths is None:
            self.discovered_paths = []


class KnowledgeGraphTraverser:
    """Traverses knowledge graph to find relevant relationships"""
    
    def __init__(self, knowledge_graph: nx.MultiDiGraph):
        self.graph = knowledge_graph
    
    def find_paths_between_concepts(self, 
                                  concept1_columns: List[str], 
                                  concept2_columns: List[str],
                                  max_path_length: int = 3) -> List[GraphPath]:
        """Find paths between two sets of columns representing concepts"""
        paths = []
        
        for start in concept1_columns:
            for end in concept2_columns:
                if start == end:
                    continue
                
                try:
                    # Find all simple paths
                    simple_paths = list(nx.all_simple_paths(
                        self.graph, start, end, cutoff=max_path_length
                    ))
                    
                    for path in simple_paths:
                        graph_path = self._analyze_path(path)
                        if graph_path:
                            paths.append(graph_path)
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by weight (higher is better)
        return sorted(paths, key=lambda p: p.total_weight, reverse=True)
    
    def find_related_columns(self, seed_columns: List[str], 
                           max_distance: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Find columns related to seed columns through relationships"""
        related = {}
        
        for seed in seed_columns:
            if seed not in self.graph:
                continue
            
            # BFS to find related nodes
            distances = nx.single_source_shortest_path_length(
                self.graph, seed, cutoff=max_distance
            )
            
            related[seed] = []
            
            for node, distance in distances.items():
                if node != seed and node.startswith('COLUMN:') and distance > 0:
                    # Get the strongest path
                    try:
                        path = nx.shortest_path(self.graph, seed, node)
                        weight = self._calculate_path_strength(path)
                        
                        related[seed].append({
                            'column': node,
                            'distance': distance,
                            'strength': weight,
                            'path': path
                        })
                    except:
                        continue
            
            # Sort by strength
            related[seed].sort(key=lambda x: x['strength'], reverse=True)
        
        return related
    
    def _analyze_path(self, path: List[str]) -> Optional[GraphPath]:
        """Analyze a path and extract relationship information"""
        if len(path) < 2:
            return None
        
        relationships = []
        total_weight = 1.0
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(source, target)
            
            if edge_data:
                # Handle multi-edge case
                if isinstance(edge_data, dict) and 0 in edge_data:
                    edge_info = edge_data[0]
                else:
                    edge_info = edge_data
                
                relationships.append({
                    'from': source,
                    'to': target,
                    'type': edge_info.get('relationship', 'UNKNOWN'),
                    'weight': edge_info.get('weight', 0.5)
                })
                
                total_weight *= edge_info.get('weight', 0.5)
        
        return GraphPath(
            start_column=path[0],
            end_column=path[-1],
            path=path,
            relationships=relationships,
            total_weight=total_weight
        )
    
    def _calculate_path_strength(self, path: List[str]) -> float:
        """Calculate the strength of a path"""
        if len(path) < 2:
            return 0.0
        
        strength = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                if isinstance(edge_data, dict) and 0 in edge_data:
                    strength *= edge_data[0].get('weight', 0.5)
                else:
                    strength *= 0.5
        
        # Penalize longer paths
        strength *= (0.8 ** (len(path) - 2))
        
        return strength


class LLMQueryAnalyzer:
    """Uses LLM to understand query intent based on actual schema"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
    
    def analyze_query(self, query: str, schema: DatasetSchema) -> Dict[str, Any]:
        """Analyze query intent using LLM with schema context"""
        
        # Build schema summary for LLM
        schema_summary = self._build_schema_summary(schema)
        
        prompt = f"""Analyze this user query in the context of the available data schema.

User Query: "{query}"

Available Data Schema:
{schema_summary}

Provide a JSON response with:
1. "concepts": List of data concepts the user is interested in (based on what's actually in the schema)
2. "analysis_type": What kind of analysis they want (aggregation, comparison, trend, distribution, etc.)
3. "relationships_needed": Whether the query requires joining multiple tables
4. "filters_implied": Any filtering conditions implied in the query
5. "relevant_columns": Column names from the schema that seem most relevant
6. "reasoning": Brief explanation of your interpretation

Focus on what's actually available in the schema, not general concepts."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            return json.loads(response.content)
        except:
            # Fallback parsing
            return {
                "concepts": [],
                "analysis_type": "general",
                "relationships_needed": False,
                "filters_implied": [],
                "relevant_columns": [],
                "reasoning": response.content
            }
    
    def _build_schema_summary(self, schema: DatasetSchema) -> str:
        """Build a concise schema summary for LLM context"""
        lines = []
        
        for table_name, table_schema in schema.tables.items():
            lines.append(f"\nTable: {table_name}")
            
            # Group columns by semantic role
            by_role = {}
            for col_name, col_schema in table_schema.columns.items():
                role = col_schema.semantic_role.value
                if role not in by_role:
                    by_role[role] = []
                by_role[role].append(f"{col_name} ({col_schema.data_type.value})")
            
            # Show columns grouped by role
            for role, columns in by_role.items():
                if columns:
                    lines.append(f"  - {role}: {', '.join(columns[:5])}")
                    if len(columns) > 5:
                        lines.append(f"    ... and {len(columns) - 5} more")
            
            # Show foreign keys
            if table_schema.foreign_keys:
                lines.append(f"  - Foreign Keys: {list(table_schema.foreign_keys.keys())}")
        
        return "\n".join(lines)


class SchemaDrivenQueryPlanner:
    """
    Dynamic query planner that adapts to any schema and uses knowledge graph
    """
    
    def __init__(self, schema: DatasetSchema, knowledge_graph: nx.MultiDiGraph):
        self.schema = schema
        self.knowledge_graph = knowledge_graph
        self.graph_traverser = KnowledgeGraphTraverser(knowledge_graph)
        self.llm_analyzer = LLMQueryAnalyzer()
    
    def plan_query(self, user_query: str) -> Dict[str, Any]:
        """
        Plan a query by:
        1. Using LLM to understand intent based on actual schema
        2. Traversing knowledge graph to find relationships
        3. Building an optimal query plan
        """
        
        # Create query context
        context = QueryContext(
            user_query=user_query,
            schema=self.schema,
            knowledge_graph=self.knowledge_graph
        )
        
        # Step 1: LLM analysis with schema context
        context.llm_analysis = self.llm_analyzer.analyze_query(user_query, self.schema)
        
        # Step 2: Find relevant columns based on LLM analysis
        relevant_columns = self._find_relevant_columns(context)
        context.selected_columns = relevant_columns
        
        # Step 3: Traverse knowledge graph to find relationships
        if len(context.llm_analysis.get('concepts', [])) > 1:
            paths = self._find_concept_connections(context)
            context.discovered_paths = paths
        
        # Step 4: Build query plan
        query_plan = self._build_query_plan(context)
        
        return query_plan
    
    def _find_relevant_columns(self, context: QueryContext) -> List[str]:
        """Find columns relevant to the query based on LLM analysis"""
        relevant = []
        
        # Get columns mentioned by LLM
        llm_columns = context.llm_analysis.get('relevant_columns', [])
        
        # Find columns for each concept
        concepts = context.llm_analysis.get('concepts', [])
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Search through schema
            for table_name, table_schema in context.schema.tables.items():
                for col_name, col_schema in table_schema.columns.items():
                    full_name = f"{table_name}.{col_name}"
                    
                    # Check if column matches concept
                    if (concept_lower in col_name.lower() or 
                        concept_lower in table_name.lower() or
                        col_name in llm_columns):
                        
                        # Add to graph format for traversal
                        graph_node = f"COLUMN:{context.schema.name}.{full_name}"
                        if graph_node in context.knowledge_graph:
                            relevant.append(graph_node)
        
        return list(set(relevant))
    
    def _find_concept_connections(self, context: QueryContext) -> List[GraphPath]:
        """Find connections between concepts using knowledge graph"""
        concepts = context.llm_analysis.get('concepts', [])
        
        if len(concepts) < 2:
            return []
        
        # Group columns by concept
        columns_by_concept = {}
        for concept in concepts:
            columns_by_concept[concept] = [
                col for col in context.selected_columns
                if concept.lower() in col.lower()
            ]
        
        # Find paths between different concepts
        all_paths = []
        concept_list = list(columns_by_concept.keys())
        
        for i in range(len(concept_list)):
            for j in range(i + 1, len(concept_list)):
                concept1, concept2 = concept_list[i], concept_list[j]
                columns1 = columns_by_concept[concept1]
                columns2 = columns_by_concept[concept2]
                
                if columns1 and columns2:
                    paths = self.graph_traverser.find_paths_between_concepts(
                        columns1, columns2, max_path_length=3
                    )
                    all_paths.extend(paths[:3])  # Top 3 paths per concept pair
        
        return all_paths
    
    def _build_query_plan(self, context: QueryContext) -> Dict[str, Any]:
        """Build the final query plan"""
        plan = {
            'query': context.user_query,
            'understanding': context.llm_analysis,
            'selected_columns': [],
            'joins': [],
            'filters': [],
            'aggregations': [],
            'group_by': [],
            'explanation': ""
        }
        
        # Extract columns from paths and direct selections
        all_columns = set(context.selected_columns)
        
        # Add columns from discovered paths
        for path in context.discovered_paths:
            all_columns.update(path.path)
        
        # Convert to readable format and categorize
        for col_node in all_columns:
            if col_node.startswith('COLUMN:'):
                parts = col_node.split('.')
                if len(parts) >= 2:
                    table = parts[-2]
                    column = parts[-1]
                    full_name = f"{table}.{column}"
                    
                    # Get column schema
                    table_schema = self.schema.tables.get(table)
                    if table_schema:
                        col_schema = table_schema.columns.get(column)
                        if col_schema:
                            column_info = {
                                'column': full_name,
                                'type': col_schema.data_type.value,
                                'role': col_schema.semantic_role.value,
                                'aggregations': col_schema.aggregation_methods
                            }
                            
                            plan['selected_columns'].append(column_info)
                            
                            # Categorize based on role and analysis type
                            if col_schema.semantic_role == SemanticRole.MEASURE:
                                if col_schema.aggregation_methods:
                                    plan['aggregations'].append({
                                        'column': full_name,
                                        'methods': col_schema.aggregation_methods
                                    })
                            elif col_schema.semantic_role == SemanticRole.DIMENSION:
                                plan['group_by'].append(full_name)
        
        # Extract joins from paths
        tables_used = set()
        for path in context.discovered_paths:
            tables = path.get_tables_in_path()
            tables_used.update(tables)
            
            # Add join information
            for rel in path.relationships:
                if rel['type'] == 'FOREIGN_KEY':
                    plan['joins'].append({
                        'from': rel['from'],
                        'to': rel['to'],
                        'type': 'INNER JOIN',
                        'confidence': rel['weight']
                    })
        
        # Add filters from LLM analysis
        plan['filters'] = context.llm_analysis.get('filters_implied', [])
        
        # Generate explanation
        plan['explanation'] = self._generate_explanation(context, plan)
        
        return plan
    
    def _generate_explanation(self, context: QueryContext, plan: Dict[str, Any]) -> str:
        """Generate human-readable explanation"""
        parts = [
            f"Query: {context.user_query}",
            "",
            "Understanding:",
            f"- Analysis Type: {context.llm_analysis.get('analysis_type', 'general')}",
            f"- Concepts: {', '.join(context.llm_analysis.get('concepts', []))}",
            "",
            "Query Plan:",
            f"- Selected {len(plan['selected_columns'])} columns",
            f"- {len(plan['joins'])} table joins required",
        ]
        
        if plan['aggregations']:
            parts.append(f"- Aggregating {len(plan['aggregations'])} measures")
        
        if plan['group_by']:
            parts.append(f"- Grouping by {len(plan['group_by'])} dimensions")
        
        if context.discovered_paths:
            parts.append("")
            parts.append("Discovered Relationships:")
            for path in context.discovered_paths[:3]:
                parts.append(f"- {path.start_column} → {path.end_column} (strength: {path.total_weight:.2f})")
        
        return "\n".join(parts)
    
    def get_columns_for_query(self, query: str) -> List[str]:
        """Simple interface to get column recommendations"""
        plan = self.plan_query(query)
        return [col['column'] for col in plan['selected_columns']]


# Tool for LangGraph agents
@tool
def plan_data_query(query: str, schema_name: str) -> str:
    """
    Plan a data analysis query using the knowledge graph and schema.
    
    This tool analyzes the user's query in the context of the actual data schema
    and uses the knowledge graph to find relevant relationships.
    
    Args:
        query: Natural language query describing the analysis
        schema_name: Name of the dataset schema to use
        
    Returns:
        JSON string with the query plan including columns, joins, and aggregations
    """
    # This would be integrated with the actual schema and knowledge graph
    # For now, return a placeholder
    return json.dumps({
        "status": "Tool requires initialization with schema and knowledge graph",
        "query": query,
        "schema": schema_name
    })