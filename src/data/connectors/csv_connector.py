import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any
from .base_connector import BaseConnector
from ..models.data_models import TableMetadata

class CSVConnector(BaseConnector):
    async def load_data(self, source_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load CSV files from directory or file list"""
        data_path = source_config.get('data_path')
        file_list = source_config.get('files', [])
        
        tables = {}
        
        if data_path and os.path.isdir(data_path):
            # Load all CSV files from directory
            for file_path in Path(data_path).glob('*.csv'):
                table_name = file_path.stem
                try:
                    df = pd.read_csv(file_path)
                    tables[table_name] = df
                    print(f"Loaded {table_name}: {df.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        elif file_list:
            # Load specific files
            for file_info in file_list:
                file_path = file_info['path']
                table_name = file_info.get('name', Path(file_path).stem)
                
                try:
                    df = pd.read_csv(file_path)
                    tables[table_name] = df
                    print(f"Loaded {table_name}: {df.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        self.tables = tables
        return tables
    
    async def analyze_schema(self) -> Dict[str, TableMetadata]:
        """Analyze schema of loaded tables"""
        metadata = {}
        
        for table_name, df in self.tables.items():
            metadata[table_name] = TableMetadata(
                table_name=table_name,
                row_count=len(df),
                column_count=len(df.columns),
                columns=df.columns.tolist(),
                column_types={col: str(df[col].dtype) for col in df.columns},
                memory_usage=df.memory_usage(deep=True).sum()
            )
            
        self.metadata = metadata
        return metadata