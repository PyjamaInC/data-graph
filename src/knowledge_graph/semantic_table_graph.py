"""
Semantic Table Graph Builder

This module builds a semantic graph of table relationships using multiple detection methods:
- Structural relationship detection (foreign keys, schema analysis)
- Semantic relationship detection (embeddings, content similarity)
- Temporal relationship detection (time-based patterns)
- Business process relationship detection (domain knowledge)
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import logging
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    EMBEDDING_AVAILABLE = False

# Import our components
try:
    from .table_relationships import (
        TableRelationshipType, TableRelationship, RelationshipEvidence,
        create_relationship_evidence, merge_relationship_evidence
    )
    from .table_intelligence import TableIntelligenceLayer, TableProfile
    from ..schema.schema_manager import SemanticRole, DataType
except ImportError:
    from table_relationships import (
        TableRelationshipType, TableRelationship, RelationshipEvidence,
        create_relationship_evidence, merge_relationship_evidence
    )
    from table_intelligence import TableIntelligenceLayer, TableProfile
    from schema.schema_manager import SemanticRole, DataType


class StructuralRelationshipDetector:
    """Detect structural relationships from schema and foreign keys"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StructuralRelationshipDetector")
    
    def detect_relationships(self, tables_data: Dict[str, Any], 
                           knowledge_graph: nx.MultiDiGraph) -> List[TableRelationship]:
        """Detect structural relationships between tables"""
        relationships = []
        
        # Extract foreign key relationships from knowledge graph
        fk_relationships = self._extract_foreign_key_relationships(knowledge_graph)
        relationships.extend(fk_relationships)
        
        # Detect junction tables
        junction_relationships = self._detect_junction_tables(tables_data)
        relationships.extend(junction_relationships)
        
        # Detect hierarchical relationships
        hierarchical_relationships = self._detect_hierarchical_relationships(tables_data)
        relationships.extend(hierarchical_relationships)
        
        return relationships
    
    def _extract_foreign_key_relationships(self, kg: nx.MultiDiGraph) -> List[TableRelationship]:
        """Extract foreign key relationships from knowledge graph"""
        relationships = []
        
        # Look for relationship edges in the enhanced knowledge graph
        enhanced_relationships = [
            'FOREIGN_KEY', 'SAME_DOMAIN', 'INFORMATION_DEPENDENCY', 
            'SIMILAR_VALUES', 'POSITIVELY_CORRELATED', 'NEGATIVELY_CORRELATED'
        ]
        
        for source, target, data in kg.edges(data=True):
            relationship_type = data.get('relationship')
            if relationship_type in enhanced_relationships:
                # Extract table names from column nodes
                source_table = self._extract_table_from_column_node(source)
                target_table = self._extract_table_from_column_node(target)
                
                if source_table and target_table and source_table != target_table:
                    # Map enhanced KG relationships to semantic table relationships
                    table_rel_type = self._map_kg_relationship_to_table_relationship(relationship_type)
                    confidence = data.get('weight', 0.7)
                    
                    evidence = create_relationship_evidence(
                        method="enhanced_kg_analysis",
                        score=confidence,
                        source_column=source,
                        target_column=target,
                        kg_relationship=relationship_type,
                        ml_features=data.get('ml_features', {}),
                        evidence_data=data.get('evidence', {})
                    )
                    
                    # Create appropriate semantic description
                    description = self._create_relationship_description(
                        source_table, target_table, relationship_type, table_rel_type
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=source_table,
                        target_table=target_table,
                        relationship_type=table_rel_type,
                        confidence=confidence,
                        evidence=[evidence.to_dict()],
                        semantic_description=description,
                        bidirectional=relationship_type in ['SIMILAR_VALUES', 'SAME_DOMAIN'],
                        join_cardinality=self._infer_join_cardinality(relationship_type),
                        typical_join_columns=[(source.split('.')[-1], target.split('.')[-1])],
                        detection_method="enhanced_structural_analysis"
                    ))
        
        return relationships
    
    def _extract_table_from_column_node(self, node: str) -> Optional[str]:
        """Extract table name from column node identifier"""
        # Handle different node naming conventions
        if "COLUMN:" in node:
            # Format: COLUMN:dataset.table.column
            parts = node.replace("COLUMN:", "").split(".")
            if len(parts) >= 2:
                return parts[1]  # Return table name
        elif "TABLE:" in node:
            # Format: TABLE:dataset.table
            return node.replace("TABLE:", "").split(".")[-1]
        
        return None
    
    def _map_kg_relationship_to_table_relationship(self, kg_rel_type: str) -> TableRelationshipType:
        """Map knowledge graph relationship types to semantic table relationship types"""
        mapping = {
            'FOREIGN_KEY': TableRelationshipType.FOREIGN_KEY,
            'SAME_DOMAIN': TableRelationshipType.DIMENSIONAL,
            'INFORMATION_DEPENDENCY': TableRelationshipType.SUPPLEMENTARY,
            'SIMILAR_VALUES': TableRelationshipType.SEMANTIC_SIMILARITY,
            'POSITIVELY_CORRELATED': TableRelationshipType.DIMENSIONAL,
            'NEGATIVELY_CORRELATED': TableRelationshipType.COMPARATIVE
        }
        return mapping.get(kg_rel_type, TableRelationshipType.SEMANTIC_SIMILARITY)
    
    def _create_relationship_description(self, source_table: str, target_table: str, 
                                       kg_rel_type: str, table_rel_type: TableRelationshipType) -> str:
        """Create semantic description for table relationship"""
        descriptions = {
            'FOREIGN_KEY': f"{source_table} references {target_table} via foreign key relationship",
            'SAME_DOMAIN': f"{source_table} and {target_table} operate in the same business domain",
            'INFORMATION_DEPENDENCY': f"{source_table} depends on information from {target_table}",
            'SIMILAR_VALUES': f"{source_table} and {target_table} contain semantically similar data",
            'POSITIVELY_CORRELATED': f"{source_table} and {target_table} have positively correlated measures",
            'NEGATIVELY_CORRELATED': f"{source_table} and {target_table} have inverse relationship patterns"
        }
        return descriptions.get(kg_rel_type, f"{source_table} and {target_table} are semantically related")
    
    def _infer_join_cardinality(self, kg_rel_type: str) -> Optional[str]:
        """Infer join cardinality from knowledge graph relationship type"""
        cardinality_mapping = {
            'FOREIGN_KEY': 'N:1',
            'SAME_DOMAIN': '1:N',
            'INFORMATION_DEPENDENCY': 'N:1',
            'SIMILAR_VALUES': 'N:M',
            'POSITIVELY_CORRELATED': '1:1',
            'NEGATIVELY_CORRELATED': '1:1'
        }
        return cardinality_mapping.get(kg_rel_type)
    
    def _detect_junction_tables(self, tables_data: Dict[str, Any]) -> List[TableRelationship]:
        """Detect junction/bridge tables for many-to-many relationships"""
        relationships = []
        
        for table_name, table_info in tables_data.items():
            if self._is_junction_table(table_info):
                # Find the two main entities this junction table connects
                connected_tables = self._find_connected_entities(table_name, table_info, tables_data)
                
                for target_table in connected_tables:
                    evidence = create_relationship_evidence(
                        method="junction_table_analysis",
                        score=0.8,
                        junction_indicators=self._get_junction_indicators(table_info)
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=table_name,
                        target_table=target_table,
                        relationship_type=TableRelationshipType.JUNCTION,
                        confidence=0.8,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{table_name} is a junction table connecting to {target_table}",
                        bidirectional=True,
                        join_cardinality="N:M",
                        detection_method="structural_analysis"
                    ))
        
        return relationships
    
    def _is_junction_table(self, table_info: Dict[str, Any]) -> bool:
        """Determine if a table is a junction table"""
        # Junction table indicators
        identifier_cols = table_info.get('identifier_columns', [])
        
        # Has multiple foreign key-like columns (typically 2)
        if len(identifier_cols) >= 2:
            # Check if table name suggests junction (contains common junction words)
            table_name = table_info.get('table_name', '').lower()
            junction_words = ['junction', 'bridge', 'link', 'mapping', 'relationship', 'assoc']
            
            if any(word in table_name for word in junction_words):
                return True
            
            # Check if has minimal non-key columns (junction tables are usually sparse)
            total_cols = table_info.get('column_count', 0)
            if total_cols - len(identifier_cols) <= 2:  # At most 2 non-key columns
                return True
        
        return False
    
    def _find_connected_entities(self, junction_table: str, junction_info: Dict[str, Any], 
                               all_tables: Dict[str, Any]) -> List[str]:
        """Find entities connected by junction table"""
        connected = []
        identifier_cols = junction_info.get('identifier_columns', [])
        
        for table_name, table_info in all_tables.items():
            if table_name == junction_table:
                continue
            
            # Check if junction table has columns that reference this table
            table_identifiers = table_info.get('identifier_columns', [])
            
            for junction_col in identifier_cols:
                for table_col in table_identifiers:
                    # Simple heuristic: column name similarity
                    if (table_name.lower() in junction_col.lower() or 
                        table_col.lower() in junction_col.lower()):
                        connected.append(table_name)
                        break
        
        return connected[:2]  # Limit to 2 main connections
    
    def _get_junction_indicators(self, table_info: Dict[str, Any]) -> List[str]:
        """Get indicators that suggest this is a junction table"""
        indicators = []
        
        if len(table_info.get('identifier_columns', [])) >= 2:
            indicators.append("multiple_identifiers")
        
        if table_info.get('column_count', 0) <= 4:
            indicators.append("minimal_columns")
        
        table_name = table_info.get('table_name', '').lower()
        junction_words = ['junction', 'bridge', 'link', 'mapping']
        if any(word in table_name for word in junction_words):
            indicators.append("junction_naming")
        
        return indicators
    
    def _detect_hierarchical_relationships(self, tables_data: Dict[str, Any]) -> List[TableRelationship]:
        """Detect hierarchical parent-child relationships"""
        relationships = []
        
        for table_name, table_info in tables_data.items():
            # Look for self-referencing tables (parent_id columns)
            identifier_cols = table_info.get('identifier_columns', [])
            
            for col in identifier_cols:
                if any(parent_word in col.lower() for parent_word in ['parent', 'super', 'master']):
                    evidence = create_relationship_evidence(
                        method="hierarchical_analysis",
                        score=0.85,
                        hierarchical_column=col
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=table_name,
                        target_table=table_name,  # Self-referencing
                        relationship_type=TableRelationshipType.HIERARCHICAL,
                        confidence=0.85,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{table_name} has hierarchical structure via {col}",
                        bidirectional=False,
                        join_cardinality="N:1",
                        detection_method="structural_analysis"
                    ))
        
        return relationships


class SemanticRelationshipDetector:
    """Detect semantic relationships using embeddings and content analysis"""
    
    def __init__(self, encoder: Optional[SentenceTransformer] = None):
        self.logger = logging.getLogger(f"{__name__}.SemanticRelationshipDetector")
        self.encoder = encoder
    
    def detect_relationships(self, table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect semantic relationships between tables"""
        relationships = []
        
        if not self.encoder:
            self.logger.warning("No sentence transformer available, skipping semantic detection")
            return relationships
        
        # Detect semantic similarity relationships
        similarity_relationships = self._detect_semantic_similarity(table_profiles)
        relationships.extend(similarity_relationships)
        
        # Detect conceptual overlap
        overlap_relationships = self._detect_conceptual_overlap(table_profiles)
        relationships.extend(overlap_relationships)
        
        # Detect supplementary relationships
        supplementary_relationships = self._detect_supplementary_relationships(table_profiles)
        relationships.extend(supplementary_relationships)
        
        return relationships
    
    def _detect_semantic_similarity(self, table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect tables with high semantic similarity"""
        relationships = []
        
        # Get embeddings
        embeddings_data = {}
        for table_name, profile in table_profiles.items():
            if profile.embedding is not None:
                embeddings_data[table_name] = profile.embedding
        
        if len(embeddings_data) < 2:
            return relationships
        
        # Calculate similarity matrix
        table_names = list(embeddings_data.keys())
        embeddings = np.array([embeddings_data[name] for name in table_names])
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find high similarity pairs
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names[i+1:], i+1):
                similarity = similarity_matrix[i][j]
                
                if similarity > 0.7:  # High similarity threshold
                    evidence = create_relationship_evidence(
                        method="semantic_similarity",
                        score=similarity,
                        embedding_similarity=float(similarity)
                    )
                    
                    # Determine specific relationship type
                    rel_type = self._classify_semantic_relationship(
                        table_profiles[table1], table_profiles[table2], similarity
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=table1,
                        target_table=table2,
                        relationship_type=rel_type,
                        confidence=similarity,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{table1} and {table2} have high semantic similarity ({similarity:.3f})",
                        bidirectional=True,
                        detection_method="semantic_analysis"
                    ))
        
        return relationships
    
    def _classify_semantic_relationship(self, profile1: TableProfile, profile2: TableProfile, 
                                      similarity: float) -> TableRelationshipType:
        """Classify the specific type of semantic relationship"""
        
        # Check for versioned relationship (very high similarity + similar structure)
        if similarity > 0.9:
            if (len(profile1.measure_columns) == len(profile2.measure_columns) and
                len(profile1.dimension_columns) == len(profile2.dimension_columns)):
                return TableRelationshipType.VERSIONED
        
        # Check for supplementary relationship (different table types)
        if profile1.table_type != profile2.table_type:
            return TableRelationshipType.SUPPLEMENTARY
        
        # Check for comparative relationship (same domain, similar structure)
        if (profile1.business_domain == profile2.business_domain and
            abs(len(profile1.measure_columns) - len(profile2.measure_columns)) <= 1):
            return TableRelationshipType.COMPARATIVE
        
        # Default to semantic similarity
        return TableRelationshipType.SEMANTIC_SIMILARITY
    
    def _detect_conceptual_overlap(self, table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect tables with significant conceptual overlap"""
        relationships = []
        
        table_names = list(table_profiles.keys())
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                profile1 = table_profiles[table1]
                profile2 = table_profiles[table2]
                
                # Calculate concept overlap
                concepts1 = set(profile1.key_concepts)
                concepts2 = set(profile2.key_concepts)
                
                if concepts1 and concepts2:
                    overlap = len(concepts1.intersection(concepts2))
                    union = len(concepts1.union(concepts2))
                    jaccard_similarity = overlap / union if union > 0 else 0
                    
                    if jaccard_similarity > 0.4:  # Significant overlap
                        evidence = create_relationship_evidence(
                            method="conceptual_overlap",
                            score=jaccard_similarity,
                            shared_concepts=list(concepts1.intersection(concepts2)),
                            jaccard_similarity=jaccard_similarity
                        )
                        
                        relationships.append(TableRelationship(
                            source_table=table1,
                            target_table=table2,
                            relationship_type=TableRelationshipType.CONCEPTUAL_OVERLAP,
                            confidence=jaccard_similarity,
                            evidence=[evidence.to_dict()],
                            semantic_description=f"{table1} and {table2} share {overlap} key concepts",
                            bidirectional=True,
                            detection_method="semantic_analysis"
                        ))
        
        return relationships
    
    def _detect_supplementary_relationships(self, table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect supplementary relationships (additional information tables)"""
        relationships = []
        
        # Look for fact tables and potential supplementary dimension tables
        fact_tables = [name for name, profile in table_profiles.items() 
                      if profile.table_type == 'fact']
        dimension_tables = [name for name, profile in table_profiles.items() 
                          if profile.table_type == 'dimension']
        
        for fact_table in fact_tables:
            fact_profile = table_profiles[fact_table]
            
            for dim_table in dimension_tables:
                dim_profile = table_profiles[dim_table]
                
                # Check if dimension table provides supplementary info to fact table
                if self._is_supplementary_relationship(fact_profile, dim_profile):
                    evidence = create_relationship_evidence(
                        method="supplementary_analysis",
                        score=0.75,
                        fact_table=fact_table,
                        dimension_table=dim_table
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=fact_table,
                        target_table=dim_table,
                        relationship_type=TableRelationshipType.SUPPLEMENTARY,
                        confidence=0.75,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{dim_table} provides supplementary information for {fact_table}",
                        bidirectional=False,
                        detection_method="semantic_analysis"
                    ))
        
        return relationships
    
    def _is_supplementary_relationship(self, fact_profile: TableProfile, 
                                     dim_profile: TableProfile) -> bool:
        """Check if dimension table supplements fact table"""
        # Simple heuristic: same business domain and dimension has descriptive columns
        return (fact_profile.business_domain == dim_profile.business_domain and
                len(dim_profile.dimension_columns) > 0)


class TemporalRelationshipDetector:
    """Detect temporal and workflow relationships"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TemporalRelationshipDetector")
    
    def detect_relationships(self, table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect temporal relationships between tables"""
        relationships = []
        
        # Find tables with temporal columns
        temporal_tables = {name: profile for name, profile in table_profiles.items()
                         if profile.temporal_columns}
        
        # Detect temporal sequences
        sequence_relationships = self._detect_temporal_sequences(temporal_tables)
        relationships.extend(sequence_relationships)
        
        # Detect workflow relationships
        workflow_relationships = self._detect_workflow_relationships(temporal_tables)
        relationships.extend(workflow_relationships)
        
        return relationships
    
    def _detect_temporal_sequences(self, temporal_tables: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect temporal sequence relationships"""
        relationships = []
        
        table_names = list(temporal_tables.keys())
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                profile1 = temporal_tables[table1]
                profile2 = temporal_tables[table2]
                
                # Check if tables represent sequential business processes
                if self._represents_temporal_sequence(profile1, profile2):
                    evidence = create_relationship_evidence(
                        method="temporal_sequence_analysis",
                        score=0.8,
                        temporal_columns_1=profile1.temporal_columns,
                        temporal_columns_2=profile2.temporal_columns
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=table1,
                        target_table=table2,
                        relationship_type=TableRelationshipType.TEMPORAL_SEQUENCE,
                        confidence=0.8,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{table1} events typically precede {table2} events",
                        bidirectional=False,
                        detection_method="temporal_analysis"
                    ))
        
        return relationships
    
    def _represents_temporal_sequence(self, profile1: TableProfile, profile2: TableProfile) -> bool:
        """Check if two tables represent a temporal sequence"""
        # Simple heuristic based on business domains and table names
        domain1 = profile1.business_domain or ""
        domain2 = profile2.business_domain or ""
        
        # Look for sequential process indicators
        sequential_patterns = [
            ("order", "ship"), ("purchase", "delivery"), ("registration", "activation"),
            ("request", "approval"), ("application", "decision")
        ]
        
        name1 = profile1.table_name.lower()
        name2 = profile2.table_name.lower()
        
        for pattern1, pattern2 in sequential_patterns:
            if pattern1 in name1 and pattern2 in name2:
                return True
            if pattern2 in name1 and pattern1 in name2:
                return True
        
        return False
    
    def _detect_workflow_relationships(self, temporal_tables: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect workflow relationships"""
        relationships = []
        
        # Look for tables that represent workflow steps
        workflow_indicators = ["status", "state", "stage", "step", "phase"]
        
        workflow_tables = []
        for name, profile in temporal_tables.items():
            if any(indicator in name.lower() for indicator in workflow_indicators):
                workflow_tables.append((name, profile))
        
        # Connect workflow tables
        for i, (table1, profile1) in enumerate(workflow_tables):
            for table2, profile2 in workflow_tables[i+1:]:
                if profile1.business_domain == profile2.business_domain:
                    evidence = create_relationship_evidence(
                        method="workflow_analysis",
                        score=0.7,
                        workflow_indicators=workflow_indicators
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=table1,
                        target_table=table2,
                        relationship_type=TableRelationshipType.WORKFLOW,
                        confidence=0.7,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{table1} and {table2} are part of the same workflow",
                        bidirectional=True,
                        detection_method="temporal_analysis"
                    ))
        
        return relationships


class BusinessProcessDetector:
    """Detect business process relationships"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BusinessProcessDetector")
        
        # Business process patterns
        self.actor_patterns = ["customer", "user", "client", "employee", "supplier", "vendor"]
        self.action_patterns = ["order", "transaction", "purchase", "sale", "payment", "activity"]
        self.resource_patterns = ["product", "inventory", "stock", "resource", "asset"]
    
    def detect_relationships(self, table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect business process relationships"""
        relationships = []
        
        # Classify tables by business role
        actors = self._find_actor_tables(table_profiles)
        actions = self._find_action_tables(table_profiles)
        resources = self._find_resource_tables(table_profiles)
        
        # Detect actor-action relationships
        actor_action_rels = self._detect_actor_action_relationships(actors, actions, table_profiles)
        relationships.extend(actor_action_rels)
        
        # Detect resource usage relationships
        resource_usage_rels = self._detect_resource_usage_relationships(actions, resources, table_profiles)
        relationships.extend(resource_usage_rels)
        
        return relationships
    
    def _find_actor_tables(self, table_profiles: Dict[str, TableProfile]) -> List[str]:
        """Find tables representing business actors"""
        actors = []
        
        for name, profile in table_profiles.items():
            name_lower = name.lower()
            concepts_lower = [c.lower() for c in profile.key_concepts]
            
            if (any(pattern in name_lower for pattern in self.actor_patterns) or
                any(any(pattern in concept for pattern in self.actor_patterns) 
                    for concept in concepts_lower)):
                actors.append(name)
        
        return actors
    
    def _find_action_tables(self, table_profiles: Dict[str, TableProfile]) -> List[str]:
        """Find tables representing business actions"""
        actions = []
        
        for name, profile in table_profiles.items():
            name_lower = name.lower()
            concepts_lower = [c.lower() for c in profile.key_concepts]
            
            # Action tables typically have temporal columns and measures
            has_temporal = len(profile.temporal_columns) > 0
            has_measures = len(profile.measure_columns) > 0
            
            if ((any(pattern in name_lower for pattern in self.action_patterns) or
                 any(any(pattern in concept for pattern in self.action_patterns) 
                     for concept in concepts_lower)) and
                (has_temporal or has_measures)):
                actions.append(name)
        
        return actions
    
    def _find_resource_tables(self, table_profiles: Dict[str, TableProfile]) -> List[str]:
        """Find tables representing business resources"""
        resources = []
        
        for name, profile in table_profiles.items():
            name_lower = name.lower()
            concepts_lower = [c.lower() for c in profile.key_concepts]
            
            if (any(pattern in name_lower for pattern in self.resource_patterns) or
                any(any(pattern in concept for pattern in self.resource_patterns) 
                    for concept in concepts_lower)):
                resources.append(name)
        
        return resources
    
    def _detect_actor_action_relationships(self, actors: List[str], actions: List[str],
                                         table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect actor-action relationships"""
        relationships = []
        
        for actor in actors:
            for action in actions:
                actor_profile = table_profiles[actor]
                action_profile = table_profiles[action]
                
                # Check if action table likely references actor
                if self._has_actor_action_relationship(actor_profile, action_profile):
                    evidence = create_relationship_evidence(
                        method="actor_action_analysis",
                        score=0.85,
                        actor_table=actor,
                        action_table=action
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=actor,
                        target_table=action,
                        relationship_type=TableRelationshipType.ACTOR_ACTION,
                        confidence=0.85,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{actor} performs actions recorded in {action}",
                        bidirectional=False,
                        business_meaning=f"Business process: {actor} → {action}",
                        detection_method="business_process_analysis"
                    ))
        
        return relationships
    
    def _has_actor_action_relationship(self, actor_profile: TableProfile, 
                                     action_profile: TableProfile) -> bool:
        """Check if actor and action tables are related"""
        # Simple heuristic: action table has identifier that might reference actor
        actor_name = actor_profile.table_name.lower()
        
        # Check if action table has columns that might reference the actor
        for col in action_profile.identifier_columns:
            if actor_name in col.lower() or any(pattern in col.lower() 
                                               for pattern in self.actor_patterns):
                return True
        
        # Check same business domain
        return actor_profile.business_domain == action_profile.business_domain
    
    def _detect_resource_usage_relationships(self, actions: List[str], resources: List[str],
                                           table_profiles: Dict[str, TableProfile]) -> List[TableRelationship]:
        """Detect resource usage relationships"""
        relationships = []
        
        for action in actions:
            for resource in resources:
                action_profile = table_profiles[action]
                resource_profile = table_profiles[resource]
                
                if self._has_resource_usage_relationship(action_profile, resource_profile):
                    evidence = create_relationship_evidence(
                        method="resource_usage_analysis",
                        score=0.8,
                        action_table=action,
                        resource_table=resource
                    )
                    
                    relationships.append(TableRelationship(
                        source_table=action,
                        target_table=resource,
                        relationship_type=TableRelationshipType.RESOURCE_USAGE,
                        confidence=0.8,
                        evidence=[evidence.to_dict()],
                        semantic_description=f"{action} uses or affects resources in {resource}",
                        bidirectional=False,
                        detection_method="business_process_analysis"
                    ))
        
        return relationships
    
    def _has_resource_usage_relationship(self, action_profile: TableProfile, 
                                       resource_profile: TableProfile) -> bool:
        """Check if action uses resources"""
        resource_name = resource_profile.table_name.lower()
        
        # Check if action table references resource
        for col in action_profile.identifier_columns:
            if resource_name in col.lower() or any(pattern in col.lower() 
                                                  for pattern in self.resource_patterns):
                return True
        
        return False


class SemanticTableGraphBuilder:
    """Build comprehensive semantic table graph using multiple detection methods"""
    
    def __init__(self, knowledge_graph: nx.MultiDiGraph, 
                 table_intelligence: TableIntelligenceLayer):
        self.kg = knowledge_graph
        self.table_intelligence = table_intelligence
        self.logger = logging.getLogger(f"{__name__}.SemanticTableGraphBuilder")
        
        # Initialize table graph
        self.table_graph = nx.MultiDiGraph()
        
        # Initialize detectors
        encoder = getattr(table_intelligence, 'encoder', None)
        self.detectors = {
            'structural': StructuralRelationshipDetector(),
            'semantic': SemanticRelationshipDetector(encoder),
            'temporal': TemporalRelationshipDetector(),
            'business': BusinessProcessDetector()
        }
        
        # Store table profiles
        self.table_profiles: Dict[str, TableProfile] = {}
        
        # Check if LLM summarizer is available
        self.llm_summarizer = getattr(table_intelligence, 'llm_summarizer', None)
        self.use_llm_descriptions = self.llm_summarizer is not None and getattr(self.llm_summarizer, 'ollama_available', False)
    
    def build_table_graph(self, tables_data: Dict[str, pd.DataFrame]) -> nx.MultiDiGraph:
        """Build comprehensive semantic table graph"""
        self.logger.info("Building semantic table graph...")
        
        # Step 1: Generate table profiles using table intelligence
        self._generate_table_profiles(tables_data)
        
        # Step 2: Add table nodes with rich metadata
        self._add_table_nodes()
        
        # Step 3: Detect relationships using all detectors
        all_relationships = []
        
        for detector_name, detector in self.detectors.items():
            self.logger.info(f"Running {detector_name} relationship detection...")
            
            if detector_name == 'structural':
                relationships = detector.detect_relationships(
                    self._create_tables_metadata(), self.kg
                )
            else:
                relationships = detector.detect_relationships(self.table_profiles)
            
            self.logger.info(f"Found {len(relationships)} {detector_name} relationships")
            all_relationships.extend(relationships)
        
        # Step 4: Add relationships to graph
        self._add_relationships_to_graph(all_relationships)
        
        # Step 5: Compute graph analytics
        self._compute_graph_analytics()
        
        self.logger.info(f"Built semantic table graph with {self.table_graph.number_of_nodes()} nodes "
                        f"and {self.table_graph.number_of_edges()} edges")
        
        return self.table_graph
    
    def _generate_table_profiles(self, tables_data: Dict[str, pd.DataFrame]):
        """Generate table profiles using table intelligence"""
        for table_name, df in tables_data.items():
            profile = self.table_intelligence.analyze_table(table_name, df)
            self.table_profiles[table_name] = profile
    
    def _create_tables_metadata(self) -> Dict[str, Any]:
        """Create tables metadata dictionary from profiles"""
        metadata = {}
        for name, profile in self.table_profiles.items():
            metadata[name] = {
                'table_name': profile.table_name,
                'row_count': profile.row_count,
                'column_count': profile.column_count,
                'table_type': profile.table_type,
                'business_domain': profile.business_domain,
                'identifier_columns': profile.identifier_columns,
                'measure_columns': profile.measure_columns,
                'dimension_columns': profile.dimension_columns,
                'temporal_columns': profile.temporal_columns,
                'key_concepts': profile.key_concepts
            }
        return metadata
    
    def _add_table_nodes(self):
        """Add table nodes with comprehensive metadata"""
        for table_name, profile in self.table_profiles.items():
            # Add node with all profile information
            self.table_graph.add_node(
                table_name,
                # Basic info
                table_name=profile.table_name,
                row_count=profile.row_count,
                column_count=profile.column_count,
                
                # Semantic info
                semantic_summary=profile.semantic_summary,
                key_concepts=profile.key_concepts,
                business_domain=profile.business_domain,
                table_type=profile.table_type,
                
                # Column categorization
                measure_columns=profile.measure_columns,
                dimension_columns=profile.dimension_columns,
                identifier_columns=profile.identifier_columns,
                temporal_columns=profile.temporal_columns,
                
                # Quality and embedding
                data_quality_score=profile.data_quality_score,
                embedding=profile.embedding,
                
                # Metadata
                profile_metadata=profile.profile_metadata,
                
                # Graph analytics (to be computed)
                centrality=0.0,
                importance_score=0.0,
                community=None
            )
    
    def _add_relationships_to_graph(self, relationships: List[TableRelationship]):
        """Add relationships to the graph"""
        for rel in relationships:
            # Check if both tables exist in graph
            if (rel.source_table in self.table_graph.nodes and 
                rel.target_table in self.table_graph.nodes):
                
                # Enhance semantic description with LLM if available
                enhanced_description = rel.semantic_description
                if self.use_llm_descriptions:
                    enhanced_description = self._enhance_relationship_description(rel)
                
                # Add edge with full relationship metadata
                self.table_graph.add_edge(
                    rel.source_table,
                    rel.target_table,
                    relationship_type=rel.relationship_type,
                    confidence=rel.confidence,
                    evidence=rel.evidence,
                    semantic_description=enhanced_description,
                    bidirectional=rel.bidirectional,
                    strength=rel.strength,
                    join_cardinality=rel.join_cardinality,
                    typical_join_columns=rel.typical_join_columns,
                    business_meaning=rel.business_meaning,
                    detection_method=rel.detection_method,
                    detection_timestamp=rel.detection_timestamp
                )
                
                # Add reverse edge if bidirectional
                if rel.bidirectional:
                    self.table_graph.add_edge(
                        rel.target_table,
                        rel.source_table,
                        relationship_type=rel.relationship_type,
                        confidence=rel.confidence,
                        evidence=rel.evidence,
                        semantic_description=enhanced_description,
                        bidirectional=rel.bidirectional,
                        strength=rel.strength,
                        join_cardinality=rel.join_cardinality,
                        typical_join_columns=[(col[1], col[0]) for col in rel.typical_join_columns],
                        business_meaning=rel.business_meaning,
                        detection_method=rel.detection_method,
                        detection_timestamp=rel.detection_timestamp
                    )
    
    def _enhance_relationship_description(self, relationship: TableRelationship) -> str:
        """Enhance relationship description using LLM"""
        try:
            # Get table profiles
            source_profile = self.table_profiles.get(relationship.source_table)
            target_profile = self.table_profiles.get(relationship.target_table)
            
            if not source_profile or not target_profile:
                return relationship.semantic_description
            
            # Get key columns from profiles
            source_columns = (
                source_profile.identifier_columns[:3] + 
                source_profile.measure_columns[:2] + 
                source_profile.dimension_columns[:2]
            )
            target_columns = (
                target_profile.identifier_columns[:3] + 
                target_profile.measure_columns[:2] + 
                target_profile.dimension_columns[:2]
            )
            
            # Generate enhanced description
            enhanced_desc = self.llm_summarizer.generate_relationship_description(
                table1_name=relationship.source_table,
                table1_summary=source_profile.semantic_summary,
                table1_columns=source_columns,
                table2_name=relationship.target_table,
                table2_summary=target_profile.semantic_summary,
                table2_columns=target_columns,
                relationship_type=relationship.relationship_type.value,
                confidence=relationship.confidence,
                linking_columns=relationship.typical_join_columns or []
            )
            
            return enhanced_desc if enhanced_desc else relationship.semantic_description
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance relationship description: {e}")
            return relationship.semantic_description
    
    def _compute_graph_analytics(self):
        """Compute graph analytics for nodes"""
        if self.table_graph.number_of_nodes() == 0:
            return
        
        # Compute centrality measures
        try:
            centrality = nx.degree_centrality(self.table_graph)
            betweenness = nx.betweenness_centrality(self.table_graph)
            
            # Update node attributes
            for node in self.table_graph.nodes():
                self.table_graph.nodes[node]['centrality'] = centrality.get(node, 0.0)
                self.table_graph.nodes[node]['betweenness'] = betweenness.get(node, 0.0)
                
                # Calculate importance score (weighted combination)
                importance = (
                    centrality.get(node, 0.0) * 0.4 +
                    betweenness.get(node, 0.0) * 0.3 +
                    self.table_graph.nodes[node].get('data_quality_score', 0.0) * 0.3
                )
                self.table_graph.nodes[node]['importance_score'] = importance
                
        except Exception as e:
            self.logger.warning(f"Failed to compute graph analytics: {e}")
    
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all relationships for a specific table"""
        relationships = []
        
        # Outgoing relationships
        for target in self.table_graph.successors(table_name):
            edge_data = self.table_graph.get_edge_data(table_name, target)
            for key, data in edge_data.items():
                relationships.append({
                    'direction': 'outgoing',
                    'source': table_name,
                    'target': target,
                    **data
                })
        
        # Incoming relationships
        for source in self.table_graph.predecessors(table_name):
            edge_data = self.table_graph.get_edge_data(source, table_name)
            for key, data in edge_data.items():
                relationships.append({
                    'direction': 'incoming',
                    'source': source,
                    'target': table_name,
                    **data
                })
        
        return relationships
    
    def export_graph_summary(self) -> Dict[str, Any]:
        """Export comprehensive graph summary"""
        return {
            'graph_stats': {
                'num_tables': self.table_graph.number_of_nodes(),
                'num_relationships': self.table_graph.number_of_edges(),
                'graph_density': nx.density(self.table_graph),
                'is_connected': nx.is_weakly_connected(self.table_graph)
            },
            'relationship_types': self._get_relationship_type_counts(),
            'business_domains': self._get_business_domain_distribution(),
            'table_types': self._get_table_type_distribution(),
            'most_important_tables': self._get_most_important_tables(5),
            'detection_methods': self._get_detection_method_counts()
        }
    
    def _get_relationship_type_counts(self) -> Dict[str, int]:
        """Get count of each relationship type"""
        counts = defaultdict(int)
        for _, _, data in self.table_graph.edges(data=True):
            rel_type = data.get('relationship_type')
            if rel_type:
                counts[rel_type.value if hasattr(rel_type, 'value') else str(rel_type)] += 1
        return dict(counts)
    
    def _get_business_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of business domains"""
        domains = defaultdict(int)
        for node, data in self.table_graph.nodes(data=True):
            domain = data.get('business_domain', 'unknown')
            domains[domain] += 1
        return dict(domains)
    
    def _get_table_type_distribution(self) -> Dict[str, int]:
        """Get distribution of table types"""
        types = defaultdict(int)
        for node, data in self.table_graph.nodes(data=True):
            table_type = data.get('table_type', 'unknown')
            types[table_type] += 1
        return dict(types)
    
    def _get_most_important_tables(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most important tables by importance score"""
        tables_with_scores = []
        for node, data in self.table_graph.nodes(data=True):
            tables_with_scores.append({
                'table': node,
                'importance_score': data.get('importance_score', 0.0),
                'centrality': data.get('centrality', 0.0),
                'business_domain': data.get('business_domain', 'unknown'),
                'table_type': data.get('table_type', 'unknown')
            })
        
        # Sort by importance score
        tables_with_scores.sort(key=lambda x: x['importance_score'], reverse=True)
        return tables_with_scores[:limit]
    
    def _get_detection_method_counts(self) -> Dict[str, int]:
        """Get count of relationships by detection method"""
        methods = defaultdict(int)
        for _, _, data in self.table_graph.edges(data=True):
            method = data.get('detection_method', 'unknown')
            methods[method] += 1
        return dict(methods)