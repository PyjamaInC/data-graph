"""
Schema Management Module

Provides flexible schema discovery and management capabilities for the knowledge graph system.
"""

from .schema_manager import (
    SchemaManager,
    DatasetSchema,
    TableSchema,
    ColumnSchema,
    DataType,
    SemanticRole,
    SchemaAutoDiscovery
)

__all__ = [
    'SchemaManager',
    'DatasetSchema', 
    'TableSchema',
    'ColumnSchema',
    'DataType',
    'SemanticRole',
    'SchemaAutoDiscovery'
]