from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
from ..models.data_models import TableMetadata, ColumnProfile

class BaseConnector(ABC):
    def __init__(self):
        self.tables = {}
        self.metadata = {}
    
    @abstractmethod
    async def load_data(self, source_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from source"""
        pass
    
    @abstractmethod  
    async def analyze_schema(self) -> Dict[str, TableMetadata]:
        """Analyze loaded data schema"""
        pass
    
    def profile_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """Generate column profile"""
        series = df[column]
        
        # Basic statistics
        stats = {
            'mean': series.mean() if pd.api.types.is_numeric_dtype(series) else None,
            'std': series.std() if pd.api.types.is_numeric_dtype(series) else None,
            'min': series.min(),
            'max': series.max(),
            'median': series.median() if pd.api.types.is_numeric_dtype(series) else None,
        }
        
        return ColumnProfile(
            column_name=column,
            table_name="",  # Will be set by caller
            data_type=str(series.dtype),
            null_count=series.isnull().sum(),
            unique_count=series.nunique(),
            sample_values=series.dropna().unique()[:10].tolist(),
            statistics=stats
        )