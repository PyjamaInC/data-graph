"""
Table Relationship Types and Data Classes

This module defines the types of semantic relationships between tables
and provides data structures for representing rich table relationships.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


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
    
    # Content-based Relationships
    SEMANTIC_SIMILARITY = "semantic_similarity"    # Tables with similar semantic content
    CONCEPTUAL_OVERLAP = "conceptual_overlap"      # Tables sharing business concepts


@dataclass
class TableRelationship:
    """Rich relationship between tables with comprehensive metadata"""
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
    typical_join_columns: List[Tuple[str, str]] = field(default_factory=list)
    business_meaning: Optional[str] = None
    
    # Detection metadata
    detection_method: str = "unknown"
    detection_timestamp: Optional[str] = None
    detection_confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize relationship data"""
        # Ensure confidence is between 0 and 1
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure strength is positive
        self.strength = max(0.0, self.strength)
        
        # Generate detection timestamp if not provided
        if self.detection_timestamp is None:
            from datetime import datetime
            self.detection_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source_table': self.source_table,
            'target_table': self.target_table,
            'relationship_type': self.relationship_type.value,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'semantic_description': self.semantic_description,
            'bidirectional': self.bidirectional,
            'strength': self.strength,
            'join_cardinality': self.join_cardinality,
            'typical_join_columns': self.typical_join_columns,
            'business_meaning': self.business_meaning,
            'detection_method': self.detection_method,
            'detection_timestamp': self.detection_timestamp,
            'detection_confidence_breakdown': self.detection_confidence_breakdown
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableRelationship':
        """Create from dictionary"""
        # Convert relationship type string back to enum
        relationship_type = TableRelationshipType(data['relationship_type'])
        
        return cls(
            source_table=data['source_table'],
            target_table=data['target_table'],
            relationship_type=relationship_type,
            confidence=data['confidence'],
            evidence=data['evidence'],
            semantic_description=data['semantic_description'],
            bidirectional=data.get('bidirectional', False),
            strength=data.get('strength', 1.0),
            join_cardinality=data.get('join_cardinality'),
            typical_join_columns=data.get('typical_join_columns', []),
            business_meaning=data.get('business_meaning'),
            detection_method=data.get('detection_method', 'unknown'),
            detection_timestamp=data.get('detection_timestamp'),
            detection_confidence_breakdown=data.get('detection_confidence_breakdown', {})
        )


@dataclass
class RelationshipEvidence:
    """Evidence supporting a table relationship"""
    method: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'score': self.score,
            'metadata': self.metadata,
            'description': self.description
        }


class RelationshipStrength(Enum):
    """Strength levels for table relationships"""
    WEAK = (0.3, "Weak relationship, low confidence")
    MODERATE = (0.6, "Moderate relationship, medium confidence")
    STRONG = (0.8, "Strong relationship, high confidence")
    VERY_STRONG = (0.95, "Very strong relationship, very high confidence")
    
    def __init__(self, threshold: float, description: str):
        self.threshold = threshold
        self.description = description
    
    @classmethod
    def classify(cls, confidence: float) -> 'RelationshipStrength':
        """Classify confidence score into strength category"""
        if confidence >= cls.VERY_STRONG.threshold:
            return cls.VERY_STRONG
        elif confidence >= cls.STRONG.threshold:
            return cls.STRONG
        elif confidence >= cls.MODERATE.threshold:
            return cls.MODERATE
        else:
            return cls.WEAK


def create_relationship_evidence(method: str, score: float, **metadata) -> RelationshipEvidence:
    """Helper function to create relationship evidence"""
    return RelationshipEvidence(
        method=method,
        score=score,
        metadata=metadata,
        description=f"Evidence from {method} with score {score:.3f}"
    )


def merge_relationship_evidence(evidences: List[RelationshipEvidence]) -> Dict[str, Any]:
    """Merge multiple pieces of evidence into a summary"""
    if not evidences:
        return {}
    
    # Calculate weighted average score
    total_weight = sum(ev.score for ev in evidences)
    weighted_score = total_weight / len(evidences) if evidences else 0.0
    
    # Collect all methods
    methods = [ev.method for ev in evidences]
    
    # Merge metadata
    merged_metadata = {}
    for ev in evidences:
        merged_metadata.update(ev.metadata)
    
    return {
        'combined_score': weighted_score,
        'evidence_count': len(evidences),
        'methods_used': methods,
        'individual_scores': [ev.score for ev in evidences],
        'merged_metadata': merged_metadata
    }


# Relationship type compatibility matrix
RELATIONSHIP_COMPATIBILITY = {
    TableRelationshipType.FOREIGN_KEY: {
        'compatible_with': [
            TableRelationshipType.ACTOR_ACTION,
            TableRelationshipType.DIMENSIONAL,
            TableRelationshipType.HIERARCHICAL
        ],
        'mutually_exclusive': [
            TableRelationshipType.JUNCTION
        ]
    },
    TableRelationshipType.TEMPORAL_SEQUENCE: {
        'compatible_with': [
            TableRelationshipType.WORKFLOW,
            TableRelationshipType.ACTOR_ACTION
        ],
        'mutually_exclusive': [
            TableRelationshipType.VERSIONED
        ]
    },
    TableRelationshipType.SEMANTIC_SIMILARITY: {
        'compatible_with': [
            TableRelationshipType.CONCEPTUAL_OVERLAP,
            TableRelationshipType.COMPARATIVE
        ],
        'mutually_exclusive': []
    }
}


def are_relationships_compatible(rel1: TableRelationshipType, rel2: TableRelationshipType) -> bool:
    """Check if two relationship types are compatible"""
    if rel1 == rel2:
        return True
    
    config1 = RELATIONSHIP_COMPATIBILITY.get(rel1, {})
    config2 = RELATIONSHIP_COMPATIBILITY.get(rel2, {})
    
    # Check mutual exclusivity
    if rel2 in config1.get('mutually_exclusive', []):
        return False
    if rel1 in config2.get('mutually_exclusive', []):
        return False
    
    # Check compatibility
    if rel2 in config1.get('compatible_with', []):
        return True
    if rel1 in config2.get('compatible_with', []):
        return True
    
    # Default: compatible unless explicitly excluded
    return True