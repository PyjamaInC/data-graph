from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import pandas as pd

@dataclass
class TableMetadata:
    table_name: str
    row_count: int
    column_count: int
    columns: List[str]
    column_types: Dict[str, str]
    memory_usage: int
    
@dataclass
class ColumnProfile:
    column_name: str
    table_name: str
    data_type: str
    null_count: int
    unique_count: int
    sample_values: List[Any]
    statistics: Dict[str, Any]

@dataclass  
class Relationship:
    source: str
    target: str
    relationship_type: str
    weight: float
    confidence: float
    evidence: Dict[str, Any]