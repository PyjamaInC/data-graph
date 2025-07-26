"""
Flexible Schema Manager for Data Analysis System

Supports both auto-discovery from data files and user-provided schema configurations.
Provides a unified interface for schema management across the knowledge graph system.
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    ProfileReport = None
    PROFILING_AVAILABLE = False


class DataType(Enum):
    """Standardized data types for schema management"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"  # IDs, keys, codes
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"


class SemanticRole(Enum):
    """Semantic roles that columns can play in analysis"""
    IDENTIFIER = "identifier"        # Primary/foreign keys
    MEASURE = "measure"             # Quantitative values for analysis
    DIMENSION = "dimension"         # Categorical grouping variables
    TEMPORAL = "temporal"           # Time-based columns
    GEOGRAPHICAL = "geographical"   # Location-based columns
    DESCRIPTIVE = "descriptive"    # Text descriptions, names
    DERIVED = "derived"             # Calculated/computed columns
    METADATA = "metadata"           # System columns (created_at, etc.)


@dataclass
class ColumnSchema:
    """Schema definition for a single column"""
    name: str
    data_type: DataType
    semantic_role: SemanticRole
    nullable: bool = True
    unique_ratio: float = 0.0
    null_ratio: float = 0.0
    
    # Business context
    display_name: Optional[str] = None
    description: Optional[str] = None
    business_domain: Optional[str] = None
    
    # Statistical properties
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    
    # Categorical properties
    categories: Optional[List[str]] = None
    cardinality: Optional[int] = None
    
    # Relationships
    foreign_key_to: Optional[str] = None  # table.column format
    related_columns: List[str] = None
    
    # Analysis hints
    analysis_tags: List[str] = None
    aggregation_methods: List[str] = None  # sum, avg, count, etc.
    
    def __post_init__(self):
        if self.related_columns is None:
            self.related_columns = []
        if self.analysis_tags is None:
            self.analysis_tags = []
        if self.aggregation_methods is None:
            self.aggregation_methods = []


@dataclass
class TableSchema:
    """Schema definition for a table"""
    name: str
    columns: Dict[str, ColumnSchema]
    
    # Table-level metadata
    description: Optional[str] = None
    business_domain: Optional[str] = None
    primary_key: Optional[str] = None
    
    # Relationships
    foreign_keys: Dict[str, str] = None  # column -> target_table.column
    related_tables: List[str] = None
    
    # Analysis context
    fact_or_dimension: Optional[str] = None  # "fact", "dimension", "bridge"
    analysis_tags: List[str] = None
    
    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = {}
        if self.related_tables is None:
            self.related_tables = []
        if self.analysis_tags is None:
            self.analysis_tags = []


@dataclass
class DatasetSchema:
    """Complete schema for a dataset"""
    name: str
    tables: Dict[str, TableSchema]
    
    # Dataset-level metadata
    description: Optional[str] = None
    business_domain: Optional[str] = None
    created_at: Optional[datetime] = None
    version: str = "1.0"
    
    # Global relationships and analysis context
    global_relationships: List[Dict[str, Any]] = None
    analysis_scenarios: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.global_relationships is None:
            self.global_relationships = []
        if self.analysis_scenarios is None:
            self.analysis_scenarios = []
        if self.created_at is None:
            self.created_at = datetime.now()


class SchemaAutoDiscovery:
    """Automatically discovers schema from data files using ydata-profiling and custom logic"""
    
    def __init__(self, use_profiling: bool = True):
        self.use_profiling = use_profiling and PROFILING_AVAILABLE
        if self.use_profiling:
            logging.info("Using ydata-profiling for enhanced schema discovery")
        else:
            logging.info("Using basic schema discovery (ydata-profiling not available)")
            
        self.semantic_patterns = {
            SemanticRole.IDENTIFIER: [
                r'.*_?id$', r'.*_?key$', r'.*_?code$', r'.*_?number$',
                r'^id$', r'^key$', r'^code$', r'^pk$', r'^uuid$'
            ],
            SemanticRole.TEMPORAL: [
                r'.*date.*', r'.*time.*', r'.*timestamp.*',
                r'.*created.*', r'.*updated.*', r'.*modified.*'
            ],
            SemanticRole.GEOGRAPHICAL: [
                r'.*address.*', r'.*city.*', r'.*state.*', r'.*country.*',
                r'.*zip.*', r'.*postal.*', r'.*location.*', r'.*region.*'
            ],
            SemanticRole.MEASURE: [
                r'.*amount.*', r'.*price.*', r'.*cost.*', r'.*value.*',
                r'.*total.*', r'.*sum.*', r'.*count.*', r'.*quantity.*'
            ],
            SemanticRole.DESCRIPTIVE: [
                r'.*name.*', r'.*title.*', r'.*description.*', r'.*label.*'
            ]
        }
        
        self.business_domain_patterns = {
            'sales': ['order', 'purchase', 'sale', 'revenue', 'customer'],
            'product': ['product', 'item', 'catalog', 'inventory', 'category'],
            'customer': ['customer', 'client', 'user', 'buyer'],
            'geography': ['location', 'address', 'city', 'state', 'country'],
            'temporal': ['date', 'time', 'created', 'updated']
        }
    
    def discover_column_schema(self, df: pd.DataFrame, column_name: str, 
                             table_name: str, profile_data: Optional[Dict] = None) -> ColumnSchema:
        """Auto-discover schema for a single column using ydata-profiling when available"""
        series = df[column_name]
        
        # Enhanced analysis using ydata-profiling if available
        if (self.use_profiling and 
            profile_data and 
            isinstance(profile_data, dict) and 
            'variables' in profile_data and 
            column_name in profile_data['variables']):
            return self._discover_from_profile(column_name, series, profile_data['variables'][column_name])
        else:
            # Fallback to basic analysis
            return self._discover_basic(column_name, series, table_name)
    
    def _discover_from_profile(self, column_name: str, series: pd.Series, 
                              profile_var: Dict) -> ColumnSchema:
        """Discover schema using ydata-profiling results"""
        # Extract profiling insights
        var_type = profile_var.get('type', 'Unknown')
        is_unique = profile_var.get('is_unique', False)
        n_missing = profile_var.get('n_missing', 0)
        n_distinct = profile_var.get('n_distinct', 0)
        count = profile_var.get('count', len(series))
        
        # Map profiling types to our types
        type_mapping = {
            'Numeric': DataType.FLOAT,
            'Categorical': DataType.CATEGORICAL,
            'Boolean': DataType.BOOLEAN,
            'DateTime': DataType.DATETIME,
            'Text': DataType.STRING,
            'URL': DataType.STRING,
            'Path': DataType.STRING,
            'Unknown': DataType.UNKNOWN
        }
        
        data_type = type_mapping.get(var_type, DataType.STRING)
        
        # Enhanced semantic role detection
        semantic_role = self._detect_semantic_role_enhanced(
            column_name, series, data_type, profile_var
        )
        
        # Extract statistical properties
        stats = profile_var.get('statistics', {})
        min_val = stats.get('min') if var_type == 'Numeric' else None
        max_val = stats.get('max') if var_type == 'Numeric' else None
        mean_val = stats.get('mean') if var_type == 'Numeric' else None
        std_val = stats.get('std') if var_type == 'Numeric' else None
        
        # Categorical properties
        categories = None
        if var_type == 'Categorical' and 'value_counts_without_nan' in profile_var:
            categories = list(profile_var['value_counts_without_nan'].keys())[:20]
        
        # Ratios
        null_ratio = n_missing / count if count > 0 else 0
        unique_ratio = n_distinct / count if count > 0 else 0
        
        # Business context
        business_domain = self._detect_business_domain(column_name)
        analysis_tags = self._generate_analysis_tags(column_name, data_type, semantic_role)
        aggregation_methods = self._suggest_aggregations(data_type, semantic_role)
        
        return ColumnSchema(
            name=column_name,
            data_type=data_type,
            semantic_role=semantic_role,
            nullable=n_missing > 0,
            unique_ratio=unique_ratio,
            null_ratio=null_ratio,
            display_name=column_name.replace('_', ' ').title(),
            description=f"Auto-discovered using ydata-profiling: {var_type}",
            business_domain=business_domain,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            categories=categories,
            cardinality=n_distinct,
            analysis_tags=analysis_tags,
            aggregation_methods=aggregation_methods
        )
    
    def _discover_basic(self, column_name: str, series: pd.Series, table_name: str) -> ColumnSchema:
        """Basic schema discovery without ydata-profiling"""
        # Basic type detection
        data_type = self._detect_data_type(series)
        semantic_role = self._detect_semantic_role(column_name, series, data_type)
        
        # Statistical analysis
        null_ratio = series.isnull().sum() / len(series)
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        
        # Statistical properties
        min_val = max_val = mean_val = std_val = None
        if pd.api.types.is_numeric_dtype(series):
            min_val = float(series.min()) if not series.empty else None
            max_val = float(series.max()) if not series.empty else None
            mean_val = float(series.mean()) if not series.empty else None
            std_val = float(series.std()) if not series.empty else None
        
        # Categorical analysis
        categories = None
        cardinality = series.nunique()
        if cardinality <= 50:  # Treat as categorical if low cardinality
            categories = series.dropna().unique().tolist()[:20]  # Limit for storage
        
        # Business domain detection
        business_domain = self._detect_business_domain(column_name)
        
        # Analysis tags
        analysis_tags = self._generate_analysis_tags(column_name, data_type, semantic_role)
        
        # Aggregation methods
        aggregation_methods = self._suggest_aggregations(data_type, semantic_role)
        
        return ColumnSchema(
            name=column_name,
            data_type=data_type,
            semantic_role=semantic_role,
            nullable=null_ratio > 0,
            unique_ratio=unique_ratio,
            null_ratio=null_ratio,
            display_name=column_name.replace('_', ' ').title(),
            business_domain=business_domain,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            categories=categories,
            cardinality=cardinality,
            analysis_tags=analysis_tags,
            aggregation_methods=aggregation_methods
        )
    
    def _detect_data_type(self, series: pd.Series) -> DataType:
        """Detect the most appropriate data type"""
        if pd.api.types.is_integer_dtype(series):
            return DataType.INTEGER
        elif pd.api.types.is_float_dtype(series):
            return DataType.FLOAT
        elif pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        elif pd.api.types.is_categorical_dtype(series):
            return DataType.CATEGORICAL
        elif pd.api.types.is_object_dtype(series):
            # Try to infer more specific types
            sample = series.dropna().astype(str).str.lower()
            if len(sample) == 0:
                return DataType.STRING
            
            # Check for boolean-like strings
            if sample.isin(['true', 'false', 'yes', 'no', '1', '0']).all():
                return DataType.BOOLEAN
            
            # Check for date-like strings
            if any(keyword in sample.iloc[0] if len(sample) > 0 else '' 
                  for keyword in ['2020', '2021', '2022', '2023', '2024']):
                return DataType.DATE
            
            # Check for currency
            if any('$' in str(val) or '€' in str(val) for val in sample[:10]):
                return DataType.CURRENCY
            
            # Check for percentage
            if any('%' in str(val) for val in sample[:10]):
                return DataType.PERCENTAGE
            
            return DataType.STRING
        else:
            return DataType.UNKNOWN
    
    def _detect_semantic_role_enhanced(self, column_name: str, series: pd.Series,
                                     data_type: DataType, profile_var: Dict) -> SemanticRole:
        """Enhanced semantic role detection using profiling insights"""
        # Use profiling insights
        is_unique = profile_var.get('is_unique', False)
        n_distinct = profile_var.get('n_distinct', 0)
        count = profile_var.get('count', len(series))
        unique_ratio = n_distinct / count if count > 0 else 0
        
        # Check if it's identified as a key by profiling
        if is_unique or unique_ratio > 0.95:
            return SemanticRole.IDENTIFIER
        
        # Use existing pattern matching
        return self._detect_semantic_role(column_name, series, data_type)
    
    def _detect_semantic_role(self, column_name: str, series: pd.Series, 
                            data_type: DataType) -> SemanticRole:
        """Detect the semantic role of a column"""
        import re
        
        column_lower = column_name.lower()
        
        # Check patterns
        for role, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                if re.match(pattern, column_lower):
                    return role
        
        # Context-based detection
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        
        # High uniqueness suggests identifier
        if unique_ratio > 0.95 and data_type in [DataType.INTEGER, DataType.STRING]:
            return SemanticRole.IDENTIFIER
        
        # Numeric types are likely measures
        if data_type in [DataType.INTEGER, DataType.FLOAT, DataType.CURRENCY]:
            return SemanticRole.MEASURE
        
        # Date/time types
        if data_type in [DataType.DATETIME, DataType.DATE]:
            return SemanticRole.TEMPORAL
        
        # Low cardinality suggests dimension
        if unique_ratio < 0.1:
            return SemanticRole.DIMENSION
        
        return SemanticRole.DESCRIPTIVE
    
    def _detect_business_domain(self, column_name: str) -> Optional[str]:
        """Detect business domain based on column name"""
        column_lower = column_name.lower()
        
        for domain, keywords in self.business_domain_patterns.items():
            if any(keyword in column_lower for keyword in keywords):
                return domain
        
        return None
    
    def _generate_analysis_tags(self, column_name: str, data_type: DataType, 
                              semantic_role: SemanticRole) -> List[str]:
        """Generate analysis tags for the column"""
        tags = []
        
        # Role-based tags
        if semantic_role == SemanticRole.MEASURE:
            tags.extend(['quantitative', 'aggregatable'])
        elif semantic_role == SemanticRole.DIMENSION:
            tags.extend(['categorical', 'groupable'])
        elif semantic_role == SemanticRole.TEMPORAL:
            tags.extend(['time-series', 'filterable'])
        elif semantic_role == SemanticRole.IDENTIFIER:
            tags.extend(['unique', 'joinable'])
        
        # Type-based tags
        if data_type == DataType.CURRENCY:
            tags.append('financial')
        elif data_type == DataType.PERCENTAGE:
            tags.append('ratio')
        
        # Name-based tags
        column_lower = column_name.lower()
        if 'price' in column_lower or 'cost' in column_lower:
            tags.append('financial')
        if 'count' in column_lower or 'quantity' in column_lower:
            tags.append('countable')
        
        return list(set(tags))  # Remove duplicates
    
    def _suggest_aggregations(self, data_type: DataType, 
                            semantic_role: SemanticRole) -> List[str]:
        """Suggest appropriate aggregation methods"""
        aggregations = []
        
        if semantic_role == SemanticRole.MEASURE:
            if data_type in [DataType.INTEGER, DataType.FLOAT, DataType.CURRENCY]:
                aggregations.extend(['sum', 'avg', 'min', 'max', 'count'])
        elif semantic_role == SemanticRole.DIMENSION:
            aggregations.extend(['count', 'count_distinct'])
        elif semantic_role == SemanticRole.IDENTIFIER:
            aggregations.extend(['count', 'count_distinct'])
        
        return aggregations


class SchemaManager:
    """Main schema management system with hybrid capabilities"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("./schemas")
        self.config_dir.mkdir(exist_ok=True)
        
        self.auto_discovery = SchemaAutoDiscovery()
        self.logger = logging.getLogger(__name__)
        
        # Cache for discovered schemas
        self._schema_cache: Dict[str, DatasetSchema] = {}
    
    def discover_schema_from_data(self, data: Dict[str, pd.DataFrame], 
                                dataset_name: str) -> DatasetSchema:
        """Auto-discover schema from loaded DataFrames using ydata-profiling when available"""
        self.logger.info(f"Auto-discovering schema for dataset: {dataset_name}")
        
        tables = {}
        
        for table_name, df in data.items():
            self.logger.info(f"Processing table: {table_name} ({df.shape})")
            
            # Generate profile if ydata-profiling is available
            profile_data = None
            if self.auto_discovery.use_profiling:
                try:
                    self.logger.info(f"Generating ydata-profiling report for {table_name}...")
                    profile = ProfileReport(
                        df.sample(min(1000, len(df))),  # Sample for performance
                        title=f"Profile for {table_name}",
                        explorative=True,
                        minimal=True  # Faster profiling
                    )
                    # Get description and convert to dict format
                    description = profile.get_description()
                    
                    # Access the variables from the description object
                    if hasattr(description, 'variables'):
                        profile_data = {'variables': {}}
                        
                        for col_name, var in description.variables.items():
                            try:
                                profile_data['variables'][col_name] = {
                                    'type': str(var.type) if hasattr(var, 'type') else 'Unknown',
                                    'is_unique': getattr(var, 'is_unique', False),
                                    'n_missing': getattr(var, 'n_missing', 0),
                                    'n_distinct': getattr(var, 'n_distinct', 0),
                                    'count': getattr(var, 'count', 0)
                                }
                            except Exception as var_error:
                                self.logger.warning(f"Error processing variable {col_name}: {var_error}")
                                # Skip this variable, continue with others
                                continue
                    else:
                        # Fallback: create empty structure
                        self.logger.warning("Profile description has no 'variables' attribute")
                        profile_data = {'variables': {}}
                    self.logger.info(f"✓ Profile generated for {table_name}")
                except Exception as e:
                    self.logger.warning(f"Profiling failed for {table_name}: {e}, using basic discovery")
                    profile_data = None
            
            # Discover columns
            columns = {}
            for column_name in df.columns:
                column_schema = self.auto_discovery.discover_column_schema(
                    df, column_name, table_name, profile_data
                )
                columns[column_name] = column_schema
            
            # Detect relationships (basic FK detection)
            foreign_keys = self._detect_foreign_keys(df, table_name, data)
            
            # Create table schema
            table_schema = TableSchema(
                name=table_name,
                columns=columns,
                description=f"Auto-discovered schema for {table_name}" + 
                           (" (enhanced with ydata-profiling)" if profile_data else ""),
                foreign_keys=foreign_keys,
                business_domain=self._detect_table_business_domain(columns)
            )
            
            tables[table_name] = table_schema
        
        # Create dataset schema
        dataset_schema = DatasetSchema(
            name=dataset_name,
            tables=tables,
            description=f"Auto-discovered schema for {dataset_name}" + 
                        (" using ydata-profiling" if self.auto_discovery.use_profiling else ""),
            created_at=datetime.now()
        )
        
        # Cache the schema
        self._schema_cache[dataset_name] = dataset_schema
        
        return dataset_schema
    
    def load_schema_from_config(self, config_path: Union[str, Path]) -> DatasetSchema:
        """Load schema from user-provided configuration file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Schema config file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Convert to DatasetSchema
        return self._config_to_schema(config_data)
    
    def save_schema_config(self, schema: DatasetSchema, 
                          output_path: Optional[Path] = None,
                          format: str = 'yaml') -> Path:
        """Save schema as configuration file"""
        if output_path is None:
            output_path = self.config_dir / f"{schema.name}_schema.{format}"
        
        # Convert schema to serializable format
        config_data = self._schema_to_config(schema)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
        elif format.lower() in ['yaml', 'yml']:
            with open(output_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Schema saved to: {output_path}")
        return output_path
    
    def get_or_create_schema(self, dataset_name: str, 
                           data: Optional[Dict[str, pd.DataFrame]] = None,
                           config_path: Optional[Path] = None) -> DatasetSchema:
        """
        Hybrid method: try to load from config, fallback to auto-discovery
        """
        # First, check cache
        if dataset_name in self._schema_cache:
            return self._schema_cache[dataset_name]
        
        # Try to load from config
        if config_path and config_path.exists():
            try:
                schema = self.load_schema_from_config(config_path)
                self._schema_cache[dataset_name] = schema
                return schema
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, falling back to auto-discovery")
        
        # Auto-discovery
        if data is not None:
            schema = self.discover_schema_from_data(data, dataset_name)
            
            # Auto-save discovered schema for future use
            try:
                self.save_schema_config(schema)
            except Exception as e:
                self.logger.warning(f"Failed to save auto-discovered schema: {e}")
            
            return schema
        
        raise ValueError(f"Cannot create schema for {dataset_name}: no data or config provided")
    
    def _detect_foreign_keys(self, df: pd.DataFrame, table_name: str, 
                           all_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Basic foreign key detection"""
        foreign_keys = {}
        
        for column in df.columns:
            if 'id' in column.lower() and column.lower() != 'id':
                # Look for matching tables
                potential_table = column.lower().replace('_id', '').replace('id', '')
                
                for other_table_name, other_df in all_data.items():
                    if other_table_name != table_name:
                        # Check if other table has matching column or 'id' column
                        if ('id' in other_df.columns or 
                            column in other_df.columns):
                            foreign_keys[column] = f"{other_table_name}.id"
                            break
        
        return foreign_keys
    
    def _detect_table_business_domain(self, columns: Dict[str, ColumnSchema]) -> Optional[str]:
        """Detect business domain for the entire table"""
        domain_counts = {}
        
        for column_schema in columns.values():
            if column_schema.business_domain:
                domain_counts[column_schema.business_domain] = domain_counts.get(
                    column_schema.business_domain, 0) + 1
        
        if domain_counts:
            return max(domain_counts, key=domain_counts.get)
        
        return None
    
    def _schema_to_config(self, schema: DatasetSchema) -> Dict[str, Any]:
        """Convert DatasetSchema to serializable config"""
        # This would convert the dataclass to a dictionary
        # Using asdict for simplicity, but you might want custom serialization
        return asdict(schema)
    
    def _config_to_schema(self, config_data: Dict[str, Any]) -> DatasetSchema:
        """Convert config dictionary to DatasetSchema"""
        # This would need custom deserialization logic
        # For brevity, this is a simplified implementation
        # You'd want to properly reconstruct the nested dataclasses
        
        tables = {}
        for table_name, table_data in config_data.get('tables', {}).items():
            columns = {}
            for col_name, col_data in table_data.get('columns', {}).items():
                columns[col_name] = ColumnSchema(**col_data)
            
            tables[table_name] = TableSchema(
                name=table_name,
                columns=columns,
                **{k: v for k, v in table_data.items() if k != 'columns'}
            )
        
        return DatasetSchema(
            name=config_data['name'],
            tables=tables,
            **{k: v for k, v in config_data.items() if k not in ['name', 'tables']}
        )