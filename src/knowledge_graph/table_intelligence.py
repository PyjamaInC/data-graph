"""
Table Intelligence Layer for Enhanced Knowledge Graph

This module provides semantic understanding of tables through embeddings, profiling,
and intelligent summarization. It serves as the foundation for the semantic table graph
and community detection components.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    ProfileReport = None
    PROFILING_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

# Import existing schema components
try:
    from ..schema.schema_manager import SchemaManager, DataType, SemanticRole
except ImportError:
    from schema.schema_manager import SchemaManager, DataType, SemanticRole


@dataclass
class TableProfile:
    """Rich profile of a table with semantic insights"""
    table_name: str
    row_count: int
    column_count: int
    semantic_summary: str
    key_concepts: List[str]
    business_domain: Optional[str]
    table_type: str  # 'fact', 'dimension', 'bridge', 'temporal'
    measure_columns: List[str]
    dimension_columns: List[str]
    identifier_columns: List[str]
    temporal_columns: List[str]
    data_quality_score: float
    embedding: Optional[np.ndarray] = None
    profile_metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedDataQualityProfile:
    """Comprehensive data quality profile using ydata-profiling alerts"""
    table_name: str
    overall_quality_score: float
    alert_summary: Dict[str, int]
    critical_alerts: List[Dict[str, Any]]
    warning_alerts: List[Dict[str, Any]]
    info_alerts: List[Dict[str, Any]]
    quality_recommendations: List[str]
    trend_alerts: List[Dict[str, Any]]  # Trend-based alerts
    distribution_alerts: List[Dict[str, Any]]  # Distribution-based alerts
    correlation_alerts: List[Dict[str, Any]]  # Correlation-based alerts


@dataclass
class EnhancedTableProfile(TableProfile):
    """Enhanced table profile with comprehensive ML insights"""
    # Data Quality Intelligence
    quality_profile: Optional[EnhancedDataQualityProfile] = None
    
    # Outlier Analysis
    outlier_analysis: Optional[Dict[str, Any]] = None
    
    # Advanced Correlations
    correlation_analysis: Optional[Dict[str, Any]] = None
    
    # Time-Series Intelligence (if applicable)
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # ML Classification Results
    classification_results: Optional[Dict[str, Any]] = None
    
    # Duplicate Analysis
    duplicate_analysis: Optional[Dict[str, Any]] = None
    
    # Missing Data Pattern Analysis
    missing_pattern_analysis: Optional[Dict[str, Any]] = None
    
    # Distribution Analysis
    distribution_analysis: Optional[Dict[str, Any]] = None
    
    # Interaction Analysis
    interaction_analysis: Optional[Dict[str, Any]] = None
    
    # ML Readiness Assessment
    ml_readiness_score: Optional[float] = None
    ml_readiness_factors: Optional[List[str]] = None
    
    # Key Insights - Semantic insights leveraging all profiling capabilities
    key_insights: Optional[List[str]] = None


@dataclass
class ColumnInsight:
    """Semantic insights about a column"""
    column_name: str
    data_type: DataType
    semantic_role: SemanticRole
    uniqueness_ratio: float
    completeness_ratio: float
    key_patterns: List[str]
    semantic_description: str
    statistical_summary: Optional[Dict[str, Any]] = None  # Enhanced statistics


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: str = "ollama"  # ollama, huggingface, llamacpp
    model: str = "llama3.2:latest"
    temperature: float = 0.1
    max_tokens: int = 300
    timeout: int = 30
    cache_enabled: bool = True
    fallback_enabled: bool = True


class PromptTemplateManager:
    """
    Manages prompt templates for different LLM tasks
    """
    
    def __init__(self):
        self.templates = {
            'table_summary': {
                'system': 'You are an expert data analyst. Generate concise, business-focused semantic summaries.',
                'user_template': """Analyze the following database table and provide a business-focused semantic summary.

TABLE: {table_name}
ROWS: {row_count}
COLUMNS: {column_count}

COLUMN ANALYSIS:
{column_analysis}

{business_context}

Generate a 2-3 sentence summary that explains:
1. What this table represents in business terms
2. Its primary purpose and key metrics/dimensions
3. How it likely fits into business processes

Focus on business meaning, not technical details. Be concise and insightful."""
            },
            'relationship_description': {
                'system': 'You are a data modeling expert. Describe table relationships in business terms.',
                'user_template': """Analyze the relationship between these two database tables:

TABLE 1: {table1_name}
Summary: {table1_summary}
Key columns: {table1_columns}

TABLE 2: {table2_name}
Summary: {table2_summary}
Key columns: {table2_columns}

RELATIONSHIP TYPE: {relationship_type}
CONFIDENCE: {confidence}%
LINKING COLUMNS: {linking_columns}

Describe this relationship in 1-2 sentences focusing on:
1. The business meaning of this relationship
2. How data flows between these tables
3. What insights this connection enables"""
            },
            'community_description': {
                'system': 'You are a business analyst. Name and describe groups of related tables.',
                'user_template': """Analyze this group of related database tables:

TABLES IN GROUP:
{table_list}

COMMON THEMES:
{common_themes}

RELATIONSHIPS:
{relationships}

Provide:
1. A concise name for this group (2-4 words)
2. A 1-2 sentence description of what this group represents in business terms"""
            }
        }
    
    def get_prompt(self, template_name: str, **kwargs) -> Tuple[str, str]:
        """
        Get a formatted prompt for a specific template
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        system_prompt = template['system']
        user_prompt = template['user_template'].format(**kwargs)
        
        return system_prompt, user_prompt
    
    def add_custom_template(self, name: str, system_prompt: str, user_template: str):
        """Add a custom prompt template"""
        self.templates[name] = {
            'system': system_prompt,
            'user_template': user_template
        }


class LLMSemanticSummarizer:
    """
    LLM-powered semantic summarization for tables and relationships
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {} if config.cache_enabled else None
        self.ollama_available = OLLAMA_AVAILABLE
        self.prompt_manager = PromptTemplateManager()
        
        if self.ollama_available:
            try:
                # Test connection
                models = ollama.list()
                self.logger.info(f"Ollama connected. Available models: {len(models.get('models', []))}")
            except Exception as e:
                self.logger.warning(f"Ollama connection failed: {e}")
                self.ollama_available = False
    
    def generate_table_summary(self, 
                              table_name: str,
                              table_profile: Dict[str, Any],
                              column_insights: List[ColumnInsight],
                              statistical_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an LLM-powered semantic summary of a table
        """
        if not self.ollama_available or not self.config.fallback_enabled:
            return self._generate_fallback_summary(table_name, table_profile, column_insights)
        
        # Check cache
        cache_key = f"table_summary_{table_name}_{hash(str(table_profile))}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Prepare prompt
            system_prompt, user_prompt = self._create_table_summary_prompt(
                table_name, table_profile, column_insights, statistical_data
            )
            
            # Generate summary using Ollama
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': user_prompt
                    }
                ],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens
                }
            )
            
            summary = response.get('message', {}).get('content', '').strip()
            
            # Cache result
            if self.cache and summary:
                self.cache[cache_key] = summary
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"LLM summary generation failed: {e}")
            return self._generate_fallback_summary(table_name, table_profile, column_insights)
    
    def generate_summary(self, prompt: str, max_tokens: int = 500) -> str:
        """General-purpose summary generation method"""
        if not self.ollama_available:
            return "Summary generation not available"
        
        try:
            response = ollama.chat(
                model=self.config.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': max_tokens
                }
            )
            return response.get('message', {}).get('content', '').strip()
        except Exception as e:
            self.logger.warning(f"Summary generation failed: {e}")
            return f"Unable to generate summary: {str(e)}"
    
    def _create_table_summary_prompt(self, 
                                   table_name: str,
                                   table_profile: Dict[str, Any],
                                   column_insights: List[ColumnInsight],
                                   statistical_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Create a well-structured prompt for table summarization using template manager
        """
        # Format column analysis
        column_lines = []
        for insight in column_insights[:15]:  # Limit to prevent prompt overflow
            col_desc = [
                f"- {insight.column_name}: {insight.data_type.value}",
                f"  Role: {insight.semantic_role.value}",
                f"  Uniqueness: {insight.uniqueness_ratio:.1%}",
                f"  Completeness: {insight.completeness_ratio:.1%}"
            ]
            
            if insight.key_patterns:
                col_desc.append(f"  Patterns: {', '.join(insight.key_patterns[:3])}")
            
            if insight.statistical_summary:
                stats = insight.statistical_summary
                if 'mean' in stats:
                    col_desc.append(f"  Mean: {stats['mean']:.2f}")
                if 'top_values' in stats:
                    col_desc.append(f"  Top values: {', '.join(str(v) for v in stats['top_values'][:3])}")
            
            column_lines.extend(col_desc)
        
        # Format business context
        business_context_parts = []
        if table_profile.get('business_domain'):
            business_context_parts.append(f"INFERRED BUSINESS DOMAIN: {table_profile['business_domain']}")
        if table_profile.get('table_type'):
            business_context_parts.append(f"TABLE TYPE: {table_profile['table_type']}-style table")
        
        # Get prompt from template manager
        return self.prompt_manager.get_prompt(
            'table_summary',
            table_name=table_name,
            row_count=table_profile.get('row_count', 'Unknown'),
            column_count=table_profile.get('column_count', 'Unknown'),
            column_analysis='\n'.join(column_lines),
            business_context='\n'.join(business_context_parts) if business_context_parts else ''
        )
    
    def _generate_fallback_summary(self, 
                                 table_name: str,
                                 table_profile: Dict[str, Any],
                                 column_insights: List[ColumnInsight]) -> str:
        """
        Generate a template-based summary as fallback
        """
        summary_parts = []
        
        # Basic info
        summary_parts.append(
            f"Table '{table_name}' contains {table_profile.get('row_count', 0)} rows "
            f"and {table_profile.get('column_count', 0)} columns"
        )
        
        # Column composition
        role_counts = Counter([insight.semantic_role for insight in column_insights])
        if role_counts:
            role_desc = []
            for role, count in role_counts.most_common():
                role_desc.append(f"{count} {role.value} column{'s' if count > 1 else ''}")
            summary_parts.append("Column composition: " + ", ".join(role_desc))
        
        # Key columns
        measure_cols = [i.column_name for i in column_insights if i.semantic_role == SemanticRole.MEASURE][:3]
        if measure_cols:
            summary_parts.append(f"Primary measures: {', '.join(measure_cols)}")
        
        dimension_cols = [i.column_name for i in column_insights if i.semantic_role == SemanticRole.DIMENSION][:3]
        if dimension_cols:
            summary_parts.append(f"Key dimensions: {', '.join(dimension_cols)}")
        
        return ". ".join(summary_parts) + "."
    
    def generate_relationship_description(self,
                                        table1_name: str,
                                        table1_summary: str,
                                        table1_columns: List[str],
                                        table2_name: str,
                                        table2_summary: str,
                                        table2_columns: List[str],
                                        relationship_type: str,
                                        confidence: float,
                                        linking_columns: List[Tuple[str, str]]) -> str:
        """
        Generate an LLM-powered description of a table relationship
        """
        if not self.ollama_available:
            return self._generate_fallback_relationship_description(
                relationship_type, confidence, linking_columns
            )
        
        # Check cache
        cache_key = f"rel_{table1_name}_{table2_name}_{relationship_type}_{confidence}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Get prompt from template
            system_prompt, user_prompt = self.prompt_manager.get_prompt(
                'relationship_description',
                table1_name=table1_name,
                table1_summary=table1_summary,
                table1_columns=', '.join(table1_columns[:5]),
                table2_name=table2_name,
                table2_summary=table2_summary,
                table2_columns=', '.join(table2_columns[:5]),
                relationship_type=relationship_type,
                confidence=int(confidence * 100),
                linking_columns=', '.join([f"{c1}->{c2}" for c1, c2 in linking_columns])
            )
            
            # Generate description using Ollama
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': 150  # Shorter for relationship descriptions
                }
            )
            
            description = response.get('message', {}).get('content', '').strip()
            
            # Cache result
            if self.cache and description:
                self.cache[cache_key] = description
            
            return description
            
        except Exception as e:
            self.logger.warning(f"LLM relationship description failed: {e}")
            return self._generate_fallback_relationship_description(
                relationship_type, confidence, linking_columns
            )
    
    def _generate_fallback_relationship_description(self,
                                                  relationship_type: str,
                                                  confidence: float,
                                                  linking_columns: List[Tuple[str, str]]) -> str:
        """Generate a template-based relationship description as fallback"""
        links = ', '.join([f"{c1}->{c2}" for c1, c2 in linking_columns])
        return f"{relationship_type} relationship (confidence: {confidence:.1%}) via {links}"


class EnhancedStatisticalProfiler:
    """
    Extract comprehensive column statistics for LLM consumption
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_column_statistics(self, column_name: str, col_data: pd.Series) -> Dict[str, Any]:
        """
        Extract rich statistical information from a column
        """
        stats = {
            'column_name': column_name,
            'total_count': len(col_data),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique()
        }
        
        non_null_data = col_data.dropna()
        
        if len(non_null_data) == 0:
            return stats
        
        # Boolean columns (handle before numeric to avoid quantile issues)
        if pd.api.types.is_bool_dtype(col_data):
            value_counts = non_null_data.value_counts()
            true_count = int(value_counts.get(True, 0))
            false_count = int(value_counts.get(False, 0))
            
            stats.update({
                'true_count': true_count,
                'false_count': false_count,
                'true_percentage': (true_count / len(non_null_data) * 100) if len(non_null_data) > 0 else 0,
                'false_percentage': (false_count / len(non_null_data) * 100) if len(non_null_data) > 0 else 0,
                'mode': bool(value_counts.index[0]) if len(value_counts) > 0 else None,
                'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            })
        
        # Numeric columns (excluding boolean)
        elif pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'mean': float(non_null_data.mean()),
                'median': float(non_null_data.median()),
                'std': float(non_null_data.std()),
                'min': float(non_null_data.min()),
                'max': float(non_null_data.max()),
                'q25': float(non_null_data.quantile(0.25)),
                'q75': float(non_null_data.quantile(0.75)),
                'skewness': float(non_null_data.skew()),
                'kurtosis': float(non_null_data.kurtosis())
            })
            
            # Detect outliers using IQR method
            q1, q3 = stats['q25'], stats['q75']
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = len(outliers) / len(non_null_data) * 100
        
        # Categorical columns
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            value_counts = non_null_data.value_counts()
            stats.update({
                'cardinality': len(value_counts),
                'top_values': value_counts.head(10).index.tolist(),
                'top_frequencies': value_counts.head(10).values.tolist(),
                'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            })
            
            # Text statistics for string columns
            if pd.api.types.is_object_dtype(col_data):
                str_lengths = non_null_data.astype(str).str.len()
                stats.update({
                    'avg_length': float(str_lengths.mean()),
                    'min_length': int(str_lengths.min()),
                    'max_length': int(str_lengths.max())
                })
        
        # Temporal columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            stats.update({
                'min_date': non_null_data.min().isoformat(),
                'max_date': non_null_data.max().isoformat(),
                'date_range_days': (non_null_data.max() - non_null_data.min()).days
            })
        
        return stats


class TableIntelligenceLayer:
    """
    Enhanced table intelligence using embeddings and semantic analysis
    
    This layer provides:
    1. Table embedding generation using sentence transformers
    2. Semantic table profiling and summarization
    3. Business domain classification
    4. Table type classification (fact/dimension)
    5. Column semantic role detection
    6. LLM-powered semantic summaries (NEW)
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 enable_profiling: bool = True,
                 cache_embeddings: bool = True,
                 # NEW PARAMETERS
                 use_llm_summaries: bool = False,
                 llm_config: Optional[LLMConfig] = None):
        """
        Initialize the table intelligence layer
        
        Args:
            model_name: Sentence transformer model name
            enable_profiling: Whether to use ydata-profiling for detailed analysis
            cache_embeddings: Whether to cache generated embeddings
            use_llm_summaries: Whether to use LLM for semantic summaries
            llm_config: Configuration for LLM integration
        """
        self.logger = logging.getLogger(__name__)
        self.enable_profiling = enable_profiling and PROFILING_AVAILABLE
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {}
        
        # LLM configuration
        self.use_llm_summaries = use_llm_summaries
        self.llm_config = llm_config or LLMConfig()
        self.llm_summarizer = LLMSemanticSummarizer(self.llm_config) if use_llm_summaries else None
        self.statistical_profiler = EnhancedStatisticalProfiler() if use_llm_summaries else None
        
        # Initialize sentence transformer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(model_name)
                self.logger.info(f"Loaded sentence transformer: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                self.encoder = None
        else:
            self.encoder = None
            self.logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        # Business domain keywords for classification
        self.domain_keywords = {
            'customer_management': ['customer', 'client', 'user', 'account', 'contact', 'person'],
            'order_processing': ['order', 'purchase', 'transaction', 'sale', 'payment', 'invoice'],
            'product_catalog': ['product', 'item', 'catalog', 'inventory', 'stock', 'category'],
            'financial': ['revenue', 'cost', 'profit', 'budget', 'expense', 'price', 'amount'],
            'operational': ['employee', 'department', 'location', 'facility', 'resource'],
            'analytics': ['metric', 'kpi', 'measure', 'report', 'summary', 'aggregate']
        }
        
        # Table type classification patterns
        self.table_type_patterns = {
            'fact': ['transaction', 'event', 'measurement', 'sales', 'order', 'activity'],
            'dimension': ['master', 'reference', 'lookup', 'category', 'type', 'status'],
            'bridge': ['mapping', 'link', 'association', 'junction', 'relationship'],
            'temporal': ['history', 'log', 'audit', 'snapshot', 'version', 'daily', 'monthly']
        }
    
    def analyze_table(self, 
                     table_name: str, 
                     df: pd.DataFrame,
                     schema_info: Optional[Dict] = None) -> TableProfile:
        """
        Perform comprehensive table analysis
        
        Args:
            table_name: Name of the table
            df: DataFrame containing the table data
            schema_info: Optional schema information from SchemaManager
            
        Returns:
            TableProfile with complete semantic analysis
        """
        self.logger.info(f"Analyzing table: {table_name}")
        
        # Basic statistics
        row_count = len(df)
        column_count = len(df.columns)
        
        # Column analysis
        column_insights = self._analyze_columns(df, schema_info)
        
        # Categorize columns by semantic role
        measure_columns = [col.column_name for col in column_insights 
                          if col.semantic_role == SemanticRole.MEASURE]
        dimension_columns = [col.column_name for col in column_insights 
                           if col.semantic_role == SemanticRole.DIMENSION]
        identifier_columns = [col.column_name for col in column_insights 
                            if col.semantic_role == SemanticRole.IDENTIFIER]
        temporal_columns = [col.column_name for col in column_insights 
                          if col.semantic_role == SemanticRole.TEMPORAL]
        
        # Generate semantic summary
        semantic_summary = self._generate_semantic_summary(
            table_name, df, column_insights
        )
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(table_name, df, column_insights)
        
        # Classify business domain
        business_domain = self._classify_business_domain(table_name, key_concepts)
        
        # Classify table type
        table_type = self._classify_table_type(
            table_name, measure_columns, dimension_columns, identifier_columns
        )
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(df, column_insights)
        
        # Generate table embedding
        embedding = self._generate_table_embedding(table_name, semantic_summary, key_concepts)
        
        # Generate detailed profile if enabled
        profile_metadata = None
        if self.enable_profiling:
            profile_metadata = self._generate_detailed_profile(df, table_name)
        
        return TableProfile(
            table_name=table_name,
            row_count=row_count,
            column_count=column_count,
            semantic_summary=semantic_summary,
            key_concepts=key_concepts,
            business_domain=business_domain,
            table_type=table_type,
            measure_columns=measure_columns,
            dimension_columns=dimension_columns,
            identifier_columns=identifier_columns,
            temporal_columns=temporal_columns,
            data_quality_score=data_quality_score,
            embedding=embedding,
            profile_metadata=profile_metadata
        )
    
    def _analyze_columns(self, df: pd.DataFrame, schema_info: Optional[Dict] = None) -> List[ColumnInsight]:
        """Analyze each column for semantic insights"""
        insights = []
        
        for column in df.columns:
            # Basic statistics
            col_data = df[column]
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            unique_count = col_data.nunique()
            
            # Calculate ratios
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            completeness_ratio = (total_count - null_count) / total_count if total_count > 0 else 0
            
            # Infer data type
            data_type = self._infer_data_type(col_data)
            
            # Infer semantic role
            semantic_role = self._infer_semantic_role(column, col_data, uniqueness_ratio)
            
            # Extract patterns
            key_patterns = self._extract_column_patterns(col_data)
            
            # Generate semantic description
            semantic_description = self._generate_column_description(
                column, data_type, semantic_role, uniqueness_ratio, completeness_ratio
            )
            
            # Extract enhanced statistics if LLM is enabled
            statistical_summary = None
            if self.use_llm_summaries and self.statistical_profiler:
                try:
                    statistical_summary = self.statistical_profiler.extract_column_statistics(column, col_data)
                except Exception as e:
                    self.logger.warning(f"Failed to extract statistics for column {column}: {e}")
            
            insights.append(ColumnInsight(
                column_name=column,
                data_type=data_type,
                semantic_role=semantic_role,
                uniqueness_ratio=uniqueness_ratio,
                completeness_ratio=completeness_ratio,
                key_patterns=key_patterns,
                semantic_description=semantic_description,
                statistical_summary=statistical_summary
            ))
        
        return insights
    
    def _infer_data_type(self, col_data: pd.Series) -> DataType:
        """Infer data type from column data"""
        # Get non-null data for analysis
        non_null_data = col_data.dropna()
        
        if len(non_null_data) == 0:
            return DataType.UNKNOWN
        
        # Check pandas dtype first
        if pd.api.types.is_integer_dtype(col_data):
            return DataType.INTEGER
        elif pd.api.types.is_float_dtype(col_data):
            return DataType.FLOAT
        elif pd.api.types.is_bool_dtype(col_data):
            return DataType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            return DataType.DATETIME
        
        # For object types, analyze content
        if pd.api.types.is_object_dtype(col_data):
            sample_values = non_null_data.head(100).astype(str)
            
            # Check for currency patterns
            if any('$' in str(val) or '€' in str(val) or '£' in str(val) for val in sample_values):
                return DataType.CURRENCY
            
            # Check for percentage patterns
            if any('%' in str(val) for val in sample_values):
                return DataType.PERCENTAGE
            
            # Check for date patterns
            date_patterns = ['-', '/', 'T', ':']
            if any(pattern in str(sample_values.iloc[0]) for pattern in date_patterns):
                try:
                    pd.to_datetime(sample_values.iloc[0])
                    return DataType.DATE
                except:
                    pass
            
            # Check categorical vs identifier
            unique_count = non_null_data.nunique()
            total_count = len(non_null_data)
            
            if unique_count == total_count:
                return DataType.IDENTIFIER
            elif unique_count < total_count * 0.1:  # Less than 10% unique
                return DataType.CATEGORICAL
            else:
                return DataType.STRING
        
        return DataType.UNKNOWN
    
    def _infer_semantic_role(self, column_name: str, col_data: pd.Series, uniqueness_ratio: float) -> SemanticRole:
        """Infer semantic role from column characteristics"""
        column_lower = column_name.lower()
        
        # Identifier patterns
        id_patterns = ['id', '_id', 'key', 'code', 'number', 'ref']
        if any(pattern in column_lower for pattern in id_patterns) or uniqueness_ratio > 0.95:
            return SemanticRole.IDENTIFIER
        
        # Temporal patterns
        temporal_patterns = ['date', 'time', 'created', 'updated', 'timestamp', 'year', 'month', 'day']
        if any(pattern in column_lower for pattern in temporal_patterns):
            return SemanticRole.TEMPORAL
        
        # Measure patterns (numeric columns that aren't IDs)
        measure_patterns = ['amount', 'price', 'cost', 'value', 'count', 'total', 'sum', 'avg', 'rate', 'score']
        if (pd.api.types.is_numeric_dtype(col_data) and 
            any(pattern in column_lower for pattern in measure_patterns)):
            return SemanticRole.MEASURE
        
        # Geographical patterns
        geo_patterns = ['address', 'city', 'country', 'state', 'zip', 'postal', 'location', 'lat', 'lon', 'longitude', 'latitude']
        if any(pattern in column_lower for pattern in geo_patterns):
            return SemanticRole.GEOGRAPHICAL
        
        # Dimension patterns (categorical data)
        if uniqueness_ratio < 0.1 and not pd.api.types.is_numeric_dtype(col_data):
            return SemanticRole.DIMENSION
        
        # Descriptive patterns
        desc_patterns = ['name', 'description', 'title', 'comment', 'note', 'text']
        if any(pattern in column_lower for pattern in desc_patterns):
            return SemanticRole.DESCRIPTIVE
        
        # Default fallback based on data type
        if pd.api.types.is_numeric_dtype(col_data):
            return SemanticRole.MEASURE
        else:
            return SemanticRole.DIMENSION
    
    def _extract_column_patterns(self, col_data: pd.Series) -> List[str]:
        """Extract key patterns from column data"""
        patterns = []
        non_null_data = col_data.dropna()
        
        if len(non_null_data) == 0:
            return patterns
        
        # Pattern: High cardinality
        if non_null_data.nunique() / len(non_null_data) > 0.9:
            patterns.append("high_cardinality")
        
        # Pattern: Low cardinality
        elif non_null_data.nunique() / len(non_null_data) < 0.1:
            patterns.append("low_cardinality")
        
        # Pattern: Contains nulls
        if col_data.isnull().any():
            patterns.append("contains_nulls")
        
        # Pattern: Numeric range
        if pd.api.types.is_numeric_dtype(col_data):
            if non_null_data.min() >= 0:
                patterns.append("non_negative")
            if non_null_data.std() / non_null_data.mean() > 2:  # High coefficient of variation
                patterns.append("high_variance")
        
        return patterns
    
    def _generate_column_description(self, column_name: str, data_type: DataType, 
                                   semantic_role: SemanticRole, uniqueness_ratio: float, 
                                   completeness_ratio: float) -> str:
        """Generate semantic description for a column"""
        desc_parts = []
        
        # Role and type
        desc_parts.append(f"{semantic_role.value} column of type {data_type.value}")
        
        # Uniqueness
        if uniqueness_ratio > 0.95:
            desc_parts.append("with unique values (likely identifier)")
        elif uniqueness_ratio < 0.1:
            desc_parts.append("with low cardinality (categorical)")
        
        # Completeness
        if completeness_ratio < 0.8:
            desc_parts.append(f"with {(1-completeness_ratio)*100:.1f}% missing values")
        
        return "; ".join(desc_parts)
    
    def _generate_semantic_summary(self, table_name: str, df: pd.DataFrame, 
                                 column_insights: List[ColumnInsight]) -> str:
        """Generate semantic summary of the table"""
        # Use LLM if enabled
        if self.use_llm_summaries and self.llm_summarizer:
            try:
                # Prepare table profile for LLM
                table_profile = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'business_domain': self._classify_business_domain(
                        table_name, 
                        self._extract_key_concepts(table_name, df, column_insights)
                    ),
                    'table_type': self._classify_table_type(
                        table_name,
                        [i.column_name for i in column_insights if i.semantic_role == SemanticRole.MEASURE],
                        [i.column_name for i in column_insights if i.semantic_role == SemanticRole.DIMENSION],
                        [i.column_name for i in column_insights if i.semantic_role == SemanticRole.IDENTIFIER]
                    )
                }
                
                # Generate LLM summary
                llm_summary = self.llm_summarizer.generate_table_summary(
                    table_name, table_profile, column_insights
                )
                
                if llm_summary and len(llm_summary) > 10:  # Basic validation
                    return llm_summary
                
            except Exception as e:
                self.logger.warning(f"Failed to generate LLM summary: {e}")
        
        # Fallback to template-based summary
        summary_parts = []
        
        # Table basic info
        summary_parts.append(f"Table '{table_name}' contains {len(df)} rows and {len(df.columns)} columns")
        
        # Column role distribution
        role_counts = Counter([insight.semantic_role for insight in column_insights])
        role_desc = []
        for role, count in role_counts.most_common():
            role_desc.append(f"{count} {role.value} column{'s' if count > 1 else ''}")
        
        if role_desc:
            summary_parts.append("Column composition: " + ", ".join(role_desc))
        
        # Key characteristics
        measure_cols = [insight.column_name for insight in column_insights 
                       if insight.semantic_role == SemanticRole.MEASURE]
        if measure_cols:
            summary_parts.append(f"Primary measures: {', '.join(measure_cols[:3])}")
        
        dimension_cols = [insight.column_name for insight in column_insights 
                         if insight.semantic_role == SemanticRole.DIMENSION]
        if dimension_cols:
            summary_parts.append(f"Key dimensions: {', '.join(dimension_cols[:3])}")
        
        return ". ".join(summary_parts) + "."
    
    def _extract_key_concepts(self, table_name: str, df: pd.DataFrame, 
                            column_insights: List[ColumnInsight]) -> List[str]:
        """Extract key business concepts from table"""
        concepts = []
        
        # Extract from table name
        table_words = table_name.lower().replace('_', ' ').split()
        concepts.extend(table_words)
        
        # Extract from column names
        for insight in column_insights:
            col_words = insight.column_name.lower().replace('_', ' ').split()
            concepts.extend(col_words)
        
        # Filter and deduplicate
        business_concepts = []
        common_words = {'id', 'name', 'date', 'time', 'created', 'updated', 'is', 'has', 'the', 'a', 'an'}
        
        for concept in concepts:
            if len(concept) > 2 and concept not in common_words:
                if concept not in business_concepts:
                    business_concepts.append(concept)
        
        return business_concepts[:10]  # Top 10 concepts
    
    def _classify_business_domain(self, table_name: str, key_concepts: List[str]) -> Optional[str]:
        """Classify table into business domain"""
        concept_text = f"{table_name} {' '.join(key_concepts)}".lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in concept_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _classify_table_type(self, table_name: str, measure_columns: List[str], 
                           dimension_columns: List[str], identifier_columns: List[str]) -> str:
        """Classify table type (fact/dimension/bridge/temporal)"""
        table_lower = table_name.lower()
        
        # Check patterns in table name
        for table_type, patterns in self.table_type_patterns.items():
            if any(pattern in table_lower for pattern in patterns):
                return table_type
        
        # Heuristic based on column composition
        if len(measure_columns) > len(dimension_columns):
            return 'fact'
        elif len(dimension_columns) > len(measure_columns) and len(identifier_columns) >= 1:
            return 'dimension'
        elif len(identifier_columns) >= 2 and len(measure_columns) == 0:
            return 'bridge'
        else:
            return 'fact'  # Default
    
    def _calculate_data_quality_score(self, df: pd.DataFrame, column_insights: List[ColumnInsight]) -> float:
        """Calculate overall data quality score"""
        if len(column_insights) == 0:
            return 0.0
        
        # Completeness score
        completeness_scores = [insight.completeness_ratio for insight in column_insights]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        
        # Consistency score (based on data types and patterns)
        consistency_score = 0.8  # Placeholder - could be enhanced with more analysis
        
        # Overall score
        quality_score = (avg_completeness * 0.6 + consistency_score * 0.4)
        
        return round(quality_score, 3)
    
    def _generate_table_embedding(self, table_name: str, semantic_summary: str, 
                                key_concepts: List[str]) -> Optional[np.ndarray]:
        """Generate embedding for the table"""
        if not self.encoder:
            return None
        
        # Check cache
        cache_key = f"{table_name}_{hash(semantic_summary)}"
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Create text representation
        text_repr = f"{table_name}. {semantic_summary}. Key concepts: {', '.join(key_concepts)}"
        
        try:
            # Generate embedding
            embedding = self.encoder.encode(text_repr)
            
            # Cache if enabled
            if self.cache_embeddings:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding for {table_name}: {e}")
            return None
    
    # def _generate_detailed_profile(self, df: pd.DataFrame, table_name: str) -> Optional[Dict[str, Any]]:
    #     """Generate detailed profile using ydata-profiling if available"""
    #     if not self.enable_profiling:
    #         return None
        
    #     try:
    #         # Generate profile report
    #         profile = ProfileReport(
    #             df, 
    #             title=f"Profile Report for {table_name}",
    #             explorative=True,
    #             minimal=False
    #         )
            
    #         # Extract key insights - handle different API versions
    #         profile_data = profile.get_description()
            
    #         # Handle newer ydata-profiling versions where get_description() returns BaseDescription object
    #         if hasattr(profile_data, 'correlations'):
    #             # Newer API - access attributes directly
    #             return {
    #                 'correlations': getattr(profile_data, 'correlations', {}),
    #                 'missing_values': getattr(profile_data, 'missing', {}),
    #                 'sample_data': getattr(profile_data, 'sample', {}),
    #                 'alerts': getattr(profile_data, 'alerts', [])
    #             }
    #         elif hasattr(profile_data, 'get'):
    #             # Older API - dictionary-like access
    #             return {
    #                 'correlations': profile_data.get('correlations', {}),
    #                 'missing_values': profile_data.get('missing', {}),
    #                 'sample_data': profile_data.get('sample', {}),
    #                 'alerts': profile_data.get('alerts', [])
    #             }
    #         else:
    #             # Fallback - try to extract what we can
    #             self.logger.info(f"Using fallback profile extraction for {table_name}")
    #             return {
    #                 'correlations': {},
    #                 'missing_values': {},
    #                 'sample_data': df.head(5).to_dict(),
    #                 'alerts': []
    #             }
            
    #     except Exception as e:
    #         self.logger.warning(f"Failed to generate detailed profile for {table_name}: {e}")
    #         return None

    def _generate_detailed_profile(self, df: pd.DataFrame, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive detailed profile using ydata-profiling
        Handles BaseDescription object from modern ydata-profiling versions
        """
        if not self.enable_profiling:
            return None
        
        try:
            # Generate profile report
            self.logger.info(f"Generating ydata-profiling report for {table_name}...")
            profile = ProfileReport(
                df, 
                title=f"Profile Report for {table_name}",
                explorative=True,
                minimal=False,
                progress_bar=False  # Disable progress bar
            )
            
            # Extract the description object (BaseDescription in modern versions)
            profile_data = profile.get_description()
            self.logger.info(f"Profile data type: {type(profile_data)}")
            
            comprehensive_profile = {}
            
            # ===== ANALYSIS SECTION =====
            try:
                if hasattr(profile_data, 'analysis'):
                    analysis = profile_data.analysis
                    comprehensive_profile['analysis'] = {
                        'title': getattr(analysis, 'title', table_name),
                        'date_start': str(getattr(analysis, 'date_start', '')),
                        'date_end': str(getattr(analysis, 'date_end', '')),
                        'duration': getattr(analysis, 'duration', 0),
                    }
                    self.logger.info("✓ Extracted analysis section")
            except Exception as e:
                self.logger.warning(f"Failed to extract analysis: {e}")
                comprehensive_profile['analysis'] = {'title': table_name}
            
            # ===== TABLE OVERVIEW =====
            try:
                if hasattr(profile_data, 'table'):
                    table_info = profile_data.table
                    comprehensive_profile['table_overview'] = {
                        'n': getattr(table_info, 'n', len(df)),
                        'n_var': getattr(table_info, 'n_var', len(df.columns)),
                        'memory_size': getattr(table_info, 'memory_size', df.memory_usage(deep=True).sum()),
                        'record_size': getattr(table_info, 'record_size', 0),
                        'n_cells_missing': getattr(table_info, 'n_cells_missing', df.isnull().sum().sum()),
                        'n_vars_with_missing': getattr(table_info, 'n_vars_with_missing', (df.isnull().sum() > 0).sum()),
                        'n_vars_all_missing': getattr(table_info, 'n_vars_all_missing', (df.isnull().sum() == len(df)).sum()),
                        'p_cells_missing': getattr(table_info, 'p_cells_missing', 0),
                        'types': self._convert_to_serializable(getattr(table_info, 'types', {})),
                    }
                    self.logger.info("✓ Extracted table overview")
            except Exception as e:
                self.logger.warning(f"Failed to extract table overview: {e}")
                comprehensive_profile['table_overview'] = {'n': len(df), 'n_var': len(df.columns)}
            
            # ===== VARIABLES (DETAILED COLUMN ANALYSIS) =====
            try:
                if hasattr(profile_data, 'variables'):
                    variables = profile_data.variables
                    comprehensive_profile['variables'] = {}
                    
                    # variables might be a dict-like object or have items()
                    if hasattr(variables, 'items'):
                        for var_name, var_data in variables.items():
                            try:
                                var_dict = self._extract_variable_data(var_data)
                                comprehensive_profile['variables'][var_name] = var_dict
                            except Exception as e:
                                self.logger.warning(f"Failed to extract data for variable {var_name}: {e}")
                                comprehensive_profile['variables'][var_name] = {'error': str(e)}
                    
                    self.logger.info(f"✓ Extracted {len(comprehensive_profile['variables'])} variables")
            except Exception as e:
                self.logger.warning(f"Failed to extract variables: {e}")
                comprehensive_profile['variables'] = {}
            
            # ===== CORRELATIONS =====
            try:
                if hasattr(profile_data, 'correlations'):
                    correlations = profile_data.correlations
                    comprehensive_profile['correlations'] = self._convert_to_serializable(correlations)
                    self.logger.info("✓ Extracted correlations")
            except Exception as e:
                self.logger.warning(f"Failed to extract correlations: {e}")
                comprehensive_profile['correlations'] = {}
            
            # ===== MISSING DATA =====
            try:
                if hasattr(profile_data, 'missing'):
                    missing = profile_data.missing
                    comprehensive_profile['missing_data'] = self._convert_to_serializable(missing)
                    self.logger.info("✓ Extracted missing data analysis")
            except Exception as e:
                self.logger.warning(f"Failed to extract missing data: {e}")
                comprehensive_profile['missing_data'] = {}
            
            # ===== ALERTS =====
            try:
                if hasattr(profile_data, 'alerts'):
                    alerts = profile_data.alerts
                    comprehensive_profile['alerts'] = self._convert_to_serializable(alerts)
                    self.logger.info(f"✓ Extracted {len(comprehensive_profile.get('alerts', []))} alerts")
            except Exception as e:
                self.logger.warning(f"Failed to extract alerts: {e}")
                comprehensive_profile['alerts'] = []
            
            # ===== SAMPLE DATA =====
            try:
                if hasattr(profile_data, 'sample'):
                    sample = profile_data.sample
                    comprehensive_profile['sample_data'] = self._convert_to_serializable(sample)
                    self.logger.info("✓ Extracted sample data")
            except Exception as e:
                self.logger.warning(f"Failed to extract sample data: {e}")
                comprehensive_profile['sample_data'] = {}
            
            # ===== DUPLICATES =====
            try:
                if hasattr(profile_data, 'duplicates'):
                    duplicates = profile_data.duplicates
                    comprehensive_profile['duplicates'] = self._convert_to_serializable(duplicates)
                    self.logger.info("✓ Extracted duplicates analysis")
            except Exception as e:
                self.logger.warning(f"Failed to extract duplicates: {e}")
                comprehensive_profile['duplicates'] = {}
            
            # ===== INTERACTIONS =====
            try:
                if hasattr(profile_data, 'interactions'):
                    interactions = profile_data.interactions
                    comprehensive_profile['interactions'] = self._convert_to_serializable(interactions)
                    self.logger.info("✓ Extracted interactions")
            except Exception as e:
                self.logger.warning(f"Failed to extract interactions: {e}")
                comprehensive_profile['interactions'] = {}
            
            # ===== PACKAGE INFO =====
            try:
                if hasattr(profile_data, 'package'):
                    package = profile_data.package
                    comprehensive_profile['package_info'] = self._convert_to_serializable(package)
                    self.logger.info("✓ Extracted package info")
            except Exception as e:
                self.logger.warning(f"Failed to extract package info: {e}")
                comprehensive_profile['package_info'] = {}
            
            # ===== ADD METADATA =====
            comprehensive_profile['extraction_metadata'] = {
                'extraction_timestamp': pd.Timestamp.now().isoformat(),
                'table_name': table_name,
                'extraction_method': 'ydata_profiling_BaseDescription',
                'successful_extractions': len([k for k in comprehensive_profile.keys() 
                                            if comprehensive_profile[k] and k != 'extraction_metadata']),
                'profile_data_type': str(type(profile_data)),
                'available_attributes': [attr for attr in dir(profile_data) if not attr.startswith('_')][:10]
            }
            
            self.logger.info(f"Profile extraction completed for {table_name}:")
            self.logger.info(f"  - Successful extractions: {comprehensive_profile['extraction_metadata']['successful_extractions']}")
            
            return comprehensive_profile
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive profile for {table_name}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _extract_variable_data(self, var_data):
        """
        Extract comprehensive data from a single variable analysis
        """
        var_dict = {}
        
        # Common attributes to extract
        common_attrs = [
            'type', 'n', 'count', 'n_distinct', 'p_distinct', 'is_unique', 
            'n_unique', 'p_unique', 'n_missing', 'p_missing', 'memory_size',
            'min', 'max', 'mean', 'std', 'variance', 'kurtosis', 'skewness',
            'sum', 'mad', 'range', 'iqr', 'cv', 'p_zeros', 'n_zeros',
            'monotonic', 'n_negative', 'p_negative', 'n_infinite', 'p_infinite'
        ]
        
        # Extract basic attributes
        for attr in common_attrs:
            if hasattr(var_data, attr):
                try:
                    value = getattr(var_data, attr)
                    var_dict[attr] = self._convert_to_serializable(value)
                except:
                    pass
        
        # Extract quantiles if available
        try:
            if hasattr(var_data, 'quantiles'):
                var_dict['quantiles'] = self._convert_to_serializable(var_data.quantiles)
        except:
            pass
        
        # Extract histogram if available
        try:
            if hasattr(var_data, 'histogram'):
                var_dict['histogram'] = self._convert_to_serializable(var_data.histogram)
        except:
            pass
        
        # Extract value counts if available
        try:
            if hasattr(var_data, 'value_counts'):
                var_dict['value_counts'] = self._convert_to_serializable(var_data.value_counts)
        except:
            pass
        
        # Extract frequency distribution
        try:
            if hasattr(var_data, 'value_counts_without_nan'):
                var_dict['value_counts_without_nan'] = self._convert_to_serializable(var_data.value_counts_without_nan)
        except:
            pass
        
        return var_dict

    def _convert_to_serializable(self, obj):
        """Convert complex objects to JSON-serializable format"""
        import numpy as np
        import pandas as pd
        from datetime import datetime, date
        
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._convert_to_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # For objects with attributes, extract them
                result = {}
                for attr in dir(obj):
                    if not attr.startswith('_') and not callable(getattr(obj, attr)):
                        try:
                            value = getattr(obj, attr)
                            result[attr] = self._convert_to_serializable(value)
                        except:
                            pass
                return result
            else:
                return str(obj)
        except Exception as e:
            return f"<conversion_error: {str(e)}>"
    
    def compare_table_similarity(self, profile1: TableProfile, profile2: TableProfile) -> float:
        """Calculate semantic similarity between two tables"""
        if profile1.embedding is None or profile2.embedding is None:
            # Fallback to concept-based similarity
            return self._calculate_concept_similarity(profile1.key_concepts, profile2.key_concepts)
        
        # Calculate cosine similarity between embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            profile1.embedding.reshape(1, -1),
            profile2.embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def _calculate_concept_similarity(self, concepts1: List[str], concepts2: List[str]) -> float:
        """Calculate similarity based on shared concepts"""
        if not concepts1 or not concepts2:
            return 0.0
        
        set1 = set(concepts1)
        set2 = set(concepts2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class AdvancedDataQualityAnalyzer:
    """
    Advanced data quality analyzer using ydata-profiling's ML-powered alerts
    """
    
    # Alert categorization mapping
    ALERT_CATEGORIES = {
        'critical': ['constant', 'zeros', 'infinite', 'rejected', 'unsupported', 'empty'],
        'warning': ['high_correlation', 'high_cardinality', 'imbalance', 'skewness', 'missing', 'duplicates'],
        'info': ['unique', 'uniform', 'constant_length', 'date', 'seasonal', 'non_stationary']
    }
    
    # Quality scoring weights
    ALERT_WEIGHTS = {
        'critical': -10,
        'warning': -3,
        'info': -1
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_data_quality(self, df: pd.DataFrame, table_name: str) -> EnhancedDataQualityProfile:
        """Generate comprehensive data quality analysis using ydata-profiling alerts"""
        
        if not PROFILING_AVAILABLE:
            self.logger.warning("ydata-profiling not available. Using basic quality analysis.")
            return self._generate_basic_quality_profile(df, table_name)
        
        try:
            # Generate enhanced profile with comprehensive alerts
            profile = ProfileReport(
                df,
                title=f"Quality Analysis: {table_name}",
                minimal=False,
                explorative=True,
                progress_bar=False,  # Disable progress bar
                # Enable all alert types
                vars={
                    'num': {'low_categorical_threshold': 0},  # Detect categorical in numeric
                    'cat': {'cardinality_threshold': 50}      # Adjust cardinality threshold
                },
                # Advanced correlation detection
                correlations={
                    "auto": {"calculate": True},
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True}, 
                    "kendall": {"calculate": True},
                    "phi_k": {"calculate": True},     # Categorical correlations
                    "cramers": {"calculate": True}    # Categorical associations
                },
                # Missing data pattern analysis
                missing_diagrams={
                    "heatmap": True,
                    "dendrogram": True,
                    "matrix": True
                },
                # Interaction detection
                interactions={
                    "continuous": True,
                    "targets": []
                }
            )
            
            profile_data = profile.get_description()
            alerts = self._extract_alerts(profile_data)
            
            # Categorize alerts
            categorized_alerts = self._categorize_alerts(alerts)
            
            # Calculate quality score
            quality_score = self._calculate_enhanced_quality_score(alerts, df)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(categorized_alerts, df)
            
            return EnhancedDataQualityProfile(
                table_name=table_name,
                overall_quality_score=quality_score,
                alert_summary=self._get_alert_summary(categorized_alerts),
                critical_alerts=categorized_alerts['critical'],
                warning_alerts=categorized_alerts['warning'],
                info_alerts=categorized_alerts['info'],
                quality_recommendations=recommendations,
                trend_alerts=self._extract_trend_alerts(alerts),
                distribution_alerts=self._extract_distribution_alerts(alerts),
                correlation_alerts=self._extract_correlation_alerts(alerts)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate advanced quality profile: {e}")
            return self._generate_basic_quality_profile(df, table_name)
    
    def _extract_alerts(self, profile_data: Dict) -> List[Dict]:
        """Extract alerts from profile data"""
        alerts = []
        
        # Handle different ydata-profiling API versions
        if hasattr(profile_data, 'alerts'):
            # Newer API - direct attribute access
            alerts = profile_data.alerts if hasattr(profile_data.alerts, '__iter__') else []
        elif isinstance(profile_data, dict) and 'alerts' in profile_data:
            # Dictionary-based API
            alerts = profile_data.get('alerts', [])
        
        return alerts
    
    def _categorize_alerts(self, alerts: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize alerts by severity"""
        categorized = {'critical': [], 'warning': [], 'info': []}
        
        for alert in alerts:
            alert_type = str(alert.get('alert_type', '') if isinstance(alert, dict) else getattr(alert, 'alert_type', '')).lower()
            
            # Enhanced alert categorization
            if any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['critical']):
                categorized['critical'].append(alert)
            elif any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['warning']):
                categorized['warning'].append(alert)
            else:
                categorized['info'].append(alert)
        
        return categorized
    
    def _calculate_enhanced_quality_score(self, alerts: List[Dict], df: pd.DataFrame) -> float:
        """Calculate advanced quality score using ML-detected patterns"""
        base_score = 100.0
        
        # Deduct points based on alert severity
        for alert in alerts:
            alert_type = str(alert.get('alert_type', '') if isinstance(alert, dict) else getattr(alert, 'alert_type', '')).lower()
            
            if any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['critical']):
                base_score += self.ALERT_WEIGHTS['critical']
            elif any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['warning']):
                base_score += self.ALERT_WEIGHTS['warning']
            else:
                base_score += self.ALERT_WEIGHTS['info']
        
        # Additional ML-based quality metrics
        completeness_bonus = self._calculate_completeness_bonus(df)
        consistency_bonus = self._calculate_consistency_bonus(df)
        
        final_score = max(0, min(100, base_score + completeness_bonus + consistency_bonus))
        return round(final_score, 2)
    
    def _generate_quality_recommendations(self, alerts: Dict, df: pd.DataFrame) -> List[str]:
        """Generate actionable quality improvement recommendations"""
        recommendations = []
        
        # Critical issues
        if alerts['critical']:
            recommendations.append("🚨 CRITICAL: Address constant columns, infinite values, or data type issues immediately")
        
        # High correlation recommendations
        correlation_alerts = [a for a in alerts['warning'] if 'correlation' in str(a.get('alert_type', '') if isinstance(a, dict) else getattr(a, 'alert_type', '')).lower()]
        if correlation_alerts:
            recommendations.append("⚠️ Consider feature selection to reduce high correlations between variables")
        
        # Missing data recommendations
        missing_alerts = [a for a in alerts['warning'] if 'missing' in str(a.get('alert_type', '') if isinstance(a, dict) else getattr(a, 'alert_type', '')).lower()]
        if missing_alerts:
            recommendations.append("📊 Implement data imputation strategy for missing values")
        
        # Skewness recommendations
        skew_alerts = [a for a in alerts['warning'] if 'skew' in str(a.get('alert_type', '') if isinstance(a, dict) else getattr(a, 'alert_type', '')).lower()]
        if skew_alerts:
            recommendations.append("📈 Consider data transformation to address skewed distributions")
        
        return recommendations
    
    def _get_alert_summary(self, categorized_alerts: Dict[str, List]) -> Dict[str, int]:
        """Get summary count of alerts by category"""
        return {
            'critical': len(categorized_alerts['critical']),
            'warning': len(categorized_alerts['warning']),
            'info': len(categorized_alerts['info']),
            'total': sum(len(alerts) for alerts in categorized_alerts.values())
        }
    
    def _extract_trend_alerts(self, alerts: List[Dict]) -> List[Dict[str, Any]]:
        """Extract trend-based alerts"""
        trend_keywords = ['trend', 'seasonal', 'time', 'temporal', 'pattern']
        return [a for a in alerts if any(keyword in str(a.get('alert_type', '') if isinstance(a, dict) else getattr(a, 'alert_type', '')).lower() for keyword in trend_keywords)]
    
    def _extract_distribution_alerts(self, alerts: List[Dict]) -> List[Dict[str, Any]]:
        """Extract distribution-based alerts"""
        distribution_keywords = ['skew', 'kurtosis', 'distribution', 'normal', 'uniform', 'bimodal']
        return [a for a in alerts if any(keyword in str(a.get('alert_type', '') if isinstance(a, dict) else getattr(a, 'alert_type', '')).lower() for keyword in distribution_keywords)]
    
    def _extract_correlation_alerts(self, alerts: List[Dict]) -> List[Dict[str, Any]]:
        """Extract correlation-based alerts"""
        correlation_keywords = ['correlation', 'collinear', 'association', 'relationship']
        return [a for a in alerts if any(keyword in str(a.get('alert_type', '') if isinstance(a, dict) else getattr(a, 'alert_type', '')).lower() for keyword in correlation_keywords)]
    
    def _calculate_completeness_bonus(self, df: pd.DataFrame) -> float:
        """Calculate bonus score based on data completeness"""
        completeness_ratio = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        return completeness_ratio * 5  # Max 5 bonus points
    
    def _calculate_consistency_bonus(self, df: pd.DataFrame) -> float:
        """Calculate bonus score based on data consistency"""
        # Simple consistency check - can be enhanced
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for infinite values
            inf_count = np.isinf(df[numeric_cols]).sum().sum()
            if inf_count == 0:
                return 3  # Bonus for no infinite values
        return 0
    
    def _generate_basic_quality_profile(self, df: pd.DataFrame, table_name: str) -> EnhancedDataQualityProfile:
        """Generate basic quality profile when ydata-profiling is not available"""
        # Basic quality metrics
        null_count = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        completeness_ratio = 1 - (null_count / total_cells) if total_cells > 0 else 0
        
        quality_score = completeness_ratio * 100
        
        recommendations = []
        if completeness_ratio < 0.8:
            recommendations.append("⚠️ High percentage of missing values detected")
        
        return EnhancedDataQualityProfile(
            table_name=table_name,
            overall_quality_score=round(quality_score, 2),
            alert_summary={'critical': 0, 'warning': 0, 'info': 0, 'total': 0},
            critical_alerts=[],
            warning_alerts=[],
            info_alerts=[],
            quality_recommendations=recommendations,
            trend_alerts=[],
            distribution_alerts=[],
            correlation_alerts=[]
        )


class MLOutlierAnalyzer:
    """Advanced outlier detection using ydata-profiling's ML capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_outliers_comprehensive(self, df: pd.DataFrame, column_insights: List[ColumnInsight]) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple ML methods"""
        
        if not PROFILING_AVAILABLE:
            self.logger.warning("ydata-profiling not available. Using basic outlier detection.")
            return self._basic_outlier_detection(df, column_insights)
        
        try:
            # Generate profile with outlier focus
            profile = ProfileReport(
                df,
                progress_bar=False,  # Disable progress bar
                # Enable advanced outlier detection
                vars={
                    'num': {
                        'low_categorical_threshold': 0,
                        'skewness_threshold': 0.5,  # Detect distribution skewness
                    }
                },
                # Enable comprehensive statistical analysis
                explorative=True
            )
            
            profile_data = profile.get_description()
            variables = self._extract_variables(profile_data)
            
            outlier_insights = {}
            
            for var_name, var_data in variables.items():
                if var_name in [insight.column_name for insight in column_insights if insight.semantic_role == SemanticRole.MEASURE]:
                    
                    outlier_info = {
                        'column_name': var_name,
                        'outlier_method': 'ydata_profiling_advanced',
                        'outlier_count': 0,
                        'outlier_percentage': 0,
                        'outlier_bounds': {},
                        'distribution_type': var_data.get('type', 'unknown'),
                        'skewness': var_data.get('skewness', 0),
                        'kurtosis': var_data.get('kurtosis', 0),
                        'outlier_impact': 'low'
                    }
                    
                    # Extract outlier information from ydata-profiling
                    if 'outliers' in var_data:
                        outlier_data = var_data['outliers']
                        outlier_info.update({
                            'outlier_count': outlier_data.get('count', 0),
                            'outlier_percentage': outlier_data.get('percentage', 0),
                            'outlier_bounds': {
                                'lower': outlier_data.get('lower_bound'),
                                'upper': outlier_data.get('upper_bound')
                            }
                        })
                    
                    # Determine outlier impact using ML-based analysis
                    outlier_info['outlier_impact'] = self._assess_outlier_impact(
                        outlier_info['outlier_percentage'],
                        outlier_info['skewness'],
                        var_data
                    )
                    
                    outlier_insights[var_name] = outlier_info
            
            return {
                'outlier_summary': outlier_insights,
                'high_impact_outliers': self._identify_high_impact_outliers(outlier_insights),
                'outlier_recommendations': self._generate_outlier_recommendations(outlier_insights)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect outliers: {e}")
            return self._basic_outlier_detection(df, column_insights)
    
    def _extract_variables(self, profile_data: Dict) -> Dict:
        """Extract variables from profile data"""
        if hasattr(profile_data, 'variables'):
            # Newer API
            return profile_data.variables if hasattr(profile_data.variables, 'items') else {}
        elif isinstance(profile_data, dict) and 'variables' in profile_data:
            # Dictionary-based API
            return profile_data.get('variables', {})
        return {}
    
    def _assess_outlier_impact(self, outlier_percentage: float, skewness: float, var_data: Dict) -> str:
        """Assess the impact of outliers using ML insights"""
        if outlier_percentage > 10:
            return 'high'
        elif outlier_percentage > 5 or abs(skewness) > 2:
            return 'medium'
        else:
            return 'low'
    
    def _identify_high_impact_outliers(self, outlier_insights: Dict) -> List[Dict]:
        """Identify columns with high-impact outliers"""
        high_impact = []
        for col_name, outlier_info in outlier_insights.items():
            if outlier_info['outlier_impact'] in ['high', 'medium']:
                high_impact.append({
                    'column': col_name,
                    'impact': outlier_info['outlier_impact'],
                    'percentage': outlier_info['outlier_percentage'],
                    'recommendation': self._get_outlier_treatment_recommendation(outlier_info)
                })
        return high_impact
    
    def _generate_outlier_recommendations(self, outlier_insights: Dict) -> List[str]:
        """Generate recommendations for handling outliers"""
        recommendations = []
        
        high_impact_count = sum(1 for info in outlier_insights.values() if info['outlier_impact'] == 'high')
        medium_impact_count = sum(1 for info in outlier_insights.values() if info['outlier_impact'] == 'medium')
        
        if high_impact_count > 0:
            recommendations.append(f"🚨 {high_impact_count} columns have high-impact outliers requiring immediate attention")
        
        if medium_impact_count > 0:
            recommendations.append(f"⚠️ {medium_impact_count} columns have moderate outliers that may affect analysis")
        
        # Specific treatment recommendations
        for col_name, info in outlier_insights.items():
            if info['outlier_impact'] == 'high':
                if abs(info['skewness']) > 3:
                    recommendations.append(f"📊 Consider log transformation for '{col_name}' due to extreme skewness")
                elif info['outlier_percentage'] > 15:
                    recommendations.append(f"🔍 Investigate data quality issues in '{col_name}' - unusually high outlier rate")
        
        return recommendations
    
    def _get_outlier_treatment_recommendation(self, outlier_info: Dict) -> str:
        """Get specific treatment recommendation for outliers"""
        if outlier_info['outlier_percentage'] > 20:
            return "Investigate data collection issues"
        elif abs(outlier_info['skewness']) > 3:
            return "Apply transformation (log, square root)"
        elif outlier_info['outlier_impact'] == 'high':
            return "Consider winsorization or trimming"
        else:
            return "Monitor but may not require treatment"
    
    def _basic_outlier_detection(self, df: pd.DataFrame, column_insights: List[ColumnInsight]) -> Dict[str, Any]:
        """Basic outlier detection using IQR method"""
        outlier_insights = {}
        
        for insight in column_insights:
            if insight.semantic_role == SemanticRole.MEASURE and insight.column_name in df.columns:
                col_data = df[insight.column_name].dropna()
                
                if pd.api.types.is_numeric_dtype(col_data) and len(col_data) > 0:
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percentage = (outlier_count / len(col_data)) * 100
                    
                    outlier_insights[insight.column_name] = {
                        'column_name': insight.column_name,
                        'outlier_method': 'iqr',
                        'outlier_count': outlier_count,
                        'outlier_percentage': outlier_percentage,
                        'outlier_bounds': {
                            'lower': lower_bound,
                            'upper': upper_bound
                        },
                        'distribution_type': 'numeric',
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'outlier_impact': 'high' if outlier_percentage > 10 else 'medium' if outlier_percentage > 5 else 'low'
                    }
        
        return {
            'outlier_summary': outlier_insights,
            'high_impact_outliers': self._identify_high_impact_outliers(outlier_insights),
            'outlier_recommendations': self._generate_outlier_recommendations(outlier_insights)
        }


class AdvancedCorrelationAnalyzer:
    """Advanced correlation analysis using ydata-profiling's multiple correlation methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_advanced_correlations(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Comprehensive correlation analysis using all available methods"""
        
        if not PROFILING_AVAILABLE:
            self.logger.warning("ydata-profiling not available. Using basic correlation analysis.")
            return self._basic_correlation_analysis(df)
        
        try:
            profile = ProfileReport(
                df,
                title=f"Correlation Analysis: {table_name}",
                progress_bar=False,  # Disable progress bar
                correlations={
                    "auto": {"calculate": True},
                    "pearson": {"calculate": True},       # Linear relationships
                    "spearman": {"calculate": True},      # Monotonic relationships  
                    "kendall": {"calculate": True},       # Ordinal relationships
                    "phi_k": {"calculate": True},         # Categorical correlations
                    "cramers": {"calculate": True}        # Categorical associations
                },
                interactions={
                    "continuous": True,                   # Detect non-linear interactions
                    "targets": []
                }
            )
            
            profile_data = profile.get_description()
            correlations = self._extract_correlations(profile_data)
            interactions = self._extract_interactions(profile_data)
            
            # Analyze correlation patterns
            correlation_insights = {
                'linear_relationships': self._analyze_linear_correlations(correlations.get('pearson', {})),
                'monotonic_relationships': self._analyze_monotonic_correlations(correlations.get('spearman', {})),
                'categorical_associations': self._analyze_categorical_correlations(
                    correlations.get('phi_k', {}), 
                    correlations.get('cramers', {})
                ),
                'non_linear_interactions': self._analyze_interactions(interactions),
                'correlation_strength_matrix': self._build_correlation_matrix(correlations),
                'feature_redundancy': self._detect_feature_redundancy(correlations),
                'relationship_recommendations': []
            }
            
            # Generate insights and recommendations
            correlation_insights['relationship_recommendations'] = self._generate_correlation_recommendations(
                correlation_insights
            )
            
            return correlation_insights
            
        except Exception as e:
            self.logger.error(f"Failed to analyze correlations: {e}")
            return self._basic_correlation_analysis(df)
    
    def _extract_correlations(self, profile_data: Dict) -> Dict:
        """Extract correlations from profile data"""
        if hasattr(profile_data, 'correlations'):
            return profile_data.correlations if hasattr(profile_data.correlations, 'items') else {}
        elif isinstance(profile_data, dict) and 'correlations' in profile_data:
            return profile_data.get('correlations', {})
        return {}
    
    def _extract_interactions(self, profile_data: Dict) -> Dict:
        """Extract interactions from profile data"""
        if hasattr(profile_data, 'interactions'):
            return profile_data.interactions if hasattr(profile_data.interactions, 'items') else {}
        elif isinstance(profile_data, dict) and 'interactions' in profile_data:
            return profile_data.get('interactions', {})
        return {}
    
    def _analyze_linear_correlations(self, pearson_corr: Dict) -> Dict[str, Any]:
        """Analyze linear relationships using Pearson correlations"""
        strong_correlations = []
        moderate_correlations = []
        
        for pair, corr_value in pearson_corr.items():
            if isinstance(corr_value, (int, float)) and not np.isnan(corr_value):
                abs_corr = abs(corr_value)
                
                if abs_corr >= 0.8:
                    strong_correlations.append({
                        'variables': pair,
                        'correlation': corr_value,
                        'strength': 'strong',
                        'type': 'positive' if corr_value > 0 else 'negative'
                    })
                elif abs_corr >= 0.5:
                    moderate_correlations.append({
                        'variables': pair,
                        'correlation': corr_value,
                        'strength': 'moderate',
                        'type': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return {
            'strong_linear': strong_correlations,
            'moderate_linear': moderate_correlations,
            'linear_relationship_count': len(strong_correlations) + len(moderate_correlations)
        }
    
    def _analyze_monotonic_correlations(self, spearman_corr: Dict) -> Dict[str, Any]:
        """Analyze monotonic relationships using Spearman correlations"""
        strong_monotonic = []
        moderate_monotonic = []
        
        for pair, corr_value in spearman_corr.items():
            if isinstance(corr_value, (int, float)) and not np.isnan(corr_value):
                abs_corr = abs(corr_value)
                
                if abs_corr >= 0.8:
                    strong_monotonic.append({
                        'variables': pair,
                        'correlation': corr_value,
                        'strength': 'strong',
                        'type': 'monotonic_increasing' if corr_value > 0 else 'monotonic_decreasing'
                    })
                elif abs_corr >= 0.5:
                    moderate_monotonic.append({
                        'variables': pair,
                        'correlation': corr_value,
                        'strength': 'moderate',
                        'type': 'monotonic_increasing' if corr_value > 0 else 'monotonic_decreasing'
                    })
        
        return {
            'strong_monotonic': strong_monotonic,
            'moderate_monotonic': moderate_monotonic,
            'monotonic_relationship_count': len(strong_monotonic) + len(moderate_monotonic)
        }
    
    def _analyze_categorical_correlations(self, phi_k_corr: Dict, cramers_corr: Dict) -> Dict[str, Any]:
        """Analyze categorical associations"""
        categorical_associations = []
        
        # Combine phi_k and cramers correlations
        all_categorical = {}
        all_categorical.update(phi_k_corr)
        all_categorical.update(cramers_corr)
        
        for pair, corr_value in all_categorical.items():
            if isinstance(corr_value, (int, float)) and not np.isnan(corr_value):
                if corr_value >= 0.3:  # Lower threshold for categorical
                    categorical_associations.append({
                        'variables': pair,
                        'association_strength': corr_value,
                        'strength': 'strong' if corr_value >= 0.5 else 'moderate',
                        'type': 'categorical_association'
                    })
        
        return {
            'categorical_associations': categorical_associations,
            'association_count': len(categorical_associations)
        }
    
    def _analyze_interactions(self, interactions: Dict) -> Dict[str, Any]:
        """Analyze non-linear interactions"""
        if not interactions:
            return {'interaction_count': 0, 'significant_interactions': []}
        
        significant_interactions = []
        
        # Extract significant interactions
        for interaction_type, interaction_data in interactions.items():
            if isinstance(interaction_data, dict):
                for pair, strength in interaction_data.items():
                    if isinstance(strength, (int, float)) and strength > 0.3:
                        significant_interactions.append({
                            'variables': pair,
                            'interaction_strength': strength,
                            'interaction_type': interaction_type
                        })
        
        return {
            'interaction_count': len(significant_interactions),
            'significant_interactions': significant_interactions
        }
    
    def _build_correlation_matrix(self, correlations: Dict) -> Dict[str, Dict[str, float]]:
        """Build a comprehensive correlation matrix"""
        matrix = {}
        
        # Use Pearson as primary, supplement with others
        primary_corr = correlations.get('pearson', correlations.get('auto', {}))
        
        for pair, value in primary_corr.items():
            if isinstance(pair, tuple) and len(pair) == 2:
                var1, var2 = pair
                if var1 not in matrix:
                    matrix[var1] = {}
                matrix[var1][var2] = value
        
        return matrix
    
    def _detect_feature_redundancy(self, correlations: Dict) -> List[Dict[str, Any]]:
        """Detect redundant features using correlation analysis"""
        redundant_groups = []
        
        # Check Pearson correlations for high linear redundancy
        pearson_corr = correlations.get('pearson', {})
        
        for pair, corr_value in pearson_corr.items():
            if isinstance(corr_value, (int, float)) and abs(corr_value) >= 0.95:
                redundant_groups.append({
                    'variables': pair,
                    'correlation_type': 'pearson',
                    'correlation_value': corr_value,
                    'redundancy_level': 'high',
                    'recommendation': 'Consider removing one variable to reduce multicollinearity'
                })
        
        return redundant_groups
    
    def _generate_correlation_recommendations(self, insights: Dict) -> List[str]:
        """Generate recommendations based on correlation analysis"""
        recommendations = []
        
        # Feature redundancy recommendations
        if insights['feature_redundancy']:
            recommendations.append(f"🔄 Found {len(insights['feature_redundancy'])} pairs of highly redundant features")
        
        # Strong correlation recommendations
        strong_linear = insights['linear_relationships']['strong_linear']
        if strong_linear:
            recommendations.append(f"📊 {len(strong_linear)} strong linear relationships detected - consider for predictive modeling")
        
        # Non-linear interaction recommendations
        interactions = insights['non_linear_interactions']['significant_interactions']
        if interactions:
            recommendations.append(f"🔀 {len(interactions)} non-linear interactions found - consider polynomial features or tree-based models")
        
        # Categorical association recommendations
        cat_associations = insights['categorical_associations']['categorical_associations']
        if cat_associations:
            recommendations.append(f"🏷️ {len(cat_associations)} significant categorical associations detected")
        
        return recommendations
    
    def _basic_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic correlation analysis using pandas"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {
                'linear_relationships': {'strong_linear': [], 'moderate_linear': [], 'linear_relationship_count': 0},
                'monotonic_relationships': {'strong_monotonic': [], 'moderate_monotonic': [], 'monotonic_relationship_count': 0},
                'categorical_associations': {'categorical_associations': [], 'association_count': 0},
                'non_linear_interactions': {'interaction_count': 0, 'significant_interactions': []},
                'correlation_strength_matrix': {},
                'feature_redundancy': [],
                'relationship_recommendations': []
            }
        
        # Calculate Pearson correlations
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Extract significant correlations
        strong_correlations = []
        moderate_correlations = []
        redundant_features = []
        
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_value = pearson_corr.iloc[i, j]
                if not np.isnan(corr_value):
                    abs_corr = abs(corr_value)
                    pair = (pearson_corr.columns[i], pearson_corr.columns[j])
                    
                    if abs_corr >= 0.95:
                        redundant_features.append({
                            'variables': pair,
                            'correlation_type': 'pearson',
                            'correlation_value': corr_value,
                            'redundancy_level': 'high',
                            'recommendation': 'Consider removing one variable'
                        })
                    elif abs_corr >= 0.8:
                        strong_correlations.append({
                            'variables': pair,
                            'correlation': corr_value,
                            'strength': 'strong',
                            'type': 'positive' if corr_value > 0 else 'negative'
                        })
                    elif abs_corr >= 0.5:
                        moderate_correlations.append({
                            'variables': pair,
                            'correlation': corr_value,
                            'strength': 'moderate',
                            'type': 'positive' if corr_value > 0 else 'negative'
                        })
        
        return {
            'linear_relationships': {
                'strong_linear': strong_correlations,
                'moderate_linear': moderate_correlations,
                'linear_relationship_count': len(strong_correlations) + len(moderate_correlations)
            },
            'monotonic_relationships': {'strong_monotonic': [], 'moderate_monotonic': [], 'monotonic_relationship_count': 0},
            'categorical_associations': {'categorical_associations': [], 'association_count': 0},
            'non_linear_interactions': {'interaction_count': 0, 'significant_interactions': []},
            'correlation_strength_matrix': pearson_corr.to_dict(),
            'feature_redundancy': redundant_features,
            'relationship_recommendations': self._generate_correlation_recommendations({
                'feature_redundancy': redundant_features,
                'linear_relationships': {'strong_linear': strong_correlations},
                'non_linear_interactions': {'significant_interactions': []},
                'categorical_associations': {'categorical_associations': []}
            })
        }


class EnhancedTableIntelligenceLayer(TableIntelligenceLayer):
    """Enhanced table intelligence with comprehensive ML capabilities"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 enable_profiling: bool = True,
                 cache_embeddings: bool = True,
                 use_llm_summaries: bool = False,
                 llm_config: Optional[LLMConfig] = None,
                 # NEW PARAMETERS
                 enable_advanced_quality: bool = True,
                 enable_outlier_detection: bool = True,
                 enable_correlation_analysis: bool = True,
                 enable_temporal_analysis: bool = True,
                 enable_ml_classification: bool = True):
        
        super().__init__(model_name, enable_profiling, cache_embeddings, use_llm_summaries, llm_config)
        
        # Initialize new analyzers
        self.quality_analyzer = AdvancedDataQualityAnalyzer() if enable_advanced_quality else None
        self.outlier_analyzer = MLOutlierAnalyzer() if enable_outlier_detection else None
        self.correlation_analyzer = AdvancedCorrelationAnalyzer() if enable_correlation_analysis else None
        self.temporal_analyzer = None  # TimeSeriesIntelligenceLayer() if enable_temporal_analysis else None
        self.ml_classifier = None  # MLTableClassifier() if enable_ml_classification else None
    
    def analyze_table_comprehensive(self, 
                                  table_name: str, 
                                  df: pd.DataFrame,
                                  schema_info: Optional[Dict] = None) -> EnhancedTableProfile:
        """Comprehensive table analysis with all ML capabilities"""
        
        self.logger.info(f"Starting comprehensive analysis for table: {table_name}")
        
        # Perform base analysis
        base_profile = super().analyze_table(table_name, df, schema_info)
        
        # Get column insights for reuse
        column_insights = self._analyze_columns(df, schema_info)
        
        # Initialize enhanced profile
        enhanced_profile = EnhancedTableProfile(
            **asdict(base_profile)
        )
        
        # Advanced Data Quality Analysis
        if self.quality_analyzer:
            self.logger.info("Performing advanced data quality analysis...")
            enhanced_profile.quality_profile = self.quality_analyzer.analyze_data_quality(df, table_name)
        
        # Outlier Detection
        if self.outlier_analyzer:
            self.logger.info("Performing ML-based outlier detection...")
            enhanced_profile.outlier_analysis = self.outlier_analyzer.detect_outliers_comprehensive(
                df, column_insights
            )
        
        # Advanced Correlation Analysis
        if self.correlation_analyzer:
            self.logger.info("Performing advanced correlation analysis...")
            enhanced_profile.correlation_analysis = self.correlation_analyzer.analyze_advanced_correlations(
                df, table_name
            )
        
        # Temporal Analysis (if applicable and implemented)
        if self.temporal_analyzer and enhanced_profile.temporal_columns:
            self.logger.info("Performing time-series analysis...")
            # enhanced_profile.temporal_analysis = self.temporal_analyzer.analyze_temporal_patterns(
            #     df, enhanced_profile.temporal_columns, table_name
            # )
        
        # ML Classification (if implemented)
        if self.ml_classifier:
            self.logger.info("Performing ML-based table classification...")
            # enhanced_profile.classification_results = self.ml_classifier.classify_table_advanced(
            #     df, table_name, column_insights
            # )
        
        # ML Readiness Assessment
        enhanced_profile.ml_readiness_score, enhanced_profile.ml_readiness_factors = self._assess_ml_readiness(
            enhanced_profile
        )
        
        # Generate key insights leveraging all profiling capabilities
        enhanced_profile.key_insights = self._generate_key_insights(enhanced_profile)
        
        self.logger.info(f"Comprehensive analysis completed for table: {table_name}")
        return enhanced_profile
    
    def _assess_ml_readiness(self, profile: EnhancedTableProfile) -> Tuple[float, List[str]]:
        """Assess ML readiness based on comprehensive analysis"""
        readiness_score = 0.0
        factors = []
        
        # Data quality factor (30% max)
        if profile.quality_profile and profile.quality_profile.overall_quality_score:
            quality_score = profile.quality_profile.overall_quality_score
            # Ensure quality_score is treated as a percentage (0-100)
            if quality_score > 1:  # It's already a percentage
                readiness_score += (quality_score / 100) * 30
            else:  # It's a ratio (0-1), convert to percentage
                quality_score = quality_score * 100
                readiness_score += quality_score * 0.30
            
            if quality_score >= 80:
                factors.append("✅ High data quality score")
            elif quality_score >= 60:
                factors.append("⚠️ Moderate data quality - may need cleaning")
            else:
                factors.append("❌ Low data quality - requires significant cleaning")
        
        # Missing data factor (20%)
        if profile.data_quality_score >= 0.9:
            readiness_score += 20
            factors.append("✅ Low missing data percentage")
        elif profile.data_quality_score >= 0.7:
            readiness_score += 10
            factors.append("⚠️ Moderate missing data")
        else:
            factors.append("❌ High missing data percentage")
        
        # Feature diversity factor (25%)
        if len(profile.measure_columns) >= 3 and len(profile.dimension_columns) >= 2:
            readiness_score += 25
            factors.append("✅ Good feature diversity")
        elif len(profile.measure_columns) >= 1 and len(profile.dimension_columns) >= 1:
            readiness_score += 15
            factors.append("⚠️ Moderate feature diversity")
        else:
            factors.append("❌ Limited feature diversity")
        
        # Data volume factor (15%)
        if profile.row_count >= 10000:
            readiness_score += 15
            factors.append("✅ Sufficient data volume for ML")
        elif profile.row_count >= 1000:
            readiness_score += 10
            factors.append("⚠️ Moderate data volume")
        else:
            factors.append("❌ Limited data volume for ML")
        
        # Correlation structure factor (10%)
        if profile.correlation_analysis:
            moderate_corr_count = len(profile.correlation_analysis.get('linear_relationships', {}).get('moderate_linear', []))
            if moderate_corr_count >= 3:
                readiness_score += 10
                factors.append("✅ Good feature relationships detected")
            elif moderate_corr_count >= 1:
                readiness_score += 5
                factors.append("⚠️ Some feature relationships detected")
            else:
                factors.append("❌ Limited feature relationships")
        
        # Ensure the score doesn't exceed 100%
        readiness_score = min(readiness_score, 100.0)
        
        return round(readiness_score, 2), factors
    
    def _generate_key_insights(self, profile: EnhancedTableProfile) -> List[str]:
        """Generate actionable key insights from comprehensive analysis"""
        insights = []
        
        # Data volume and scale insights
        if profile.row_count > 1000000:
            insights.append(f"Large-scale dataset with {profile.row_count:,} records - consider partitioning for optimal performance")
        elif profile.row_count > 100000:
            insights.append(f"Substantial dataset with {profile.row_count:,} records - suitable for advanced analytics and ML")
        elif profile.row_count < 1000:
            insights.append(f"Small dataset ({profile.row_count:,} records) - may limit statistical significance")
        
        # Data quality insights based on comprehensive analysis
        if profile.quality_profile:
            quality_score = profile.quality_profile.overall_quality_score
            if quality_score >= 95:
                insights.append("Exceptional data quality (95%+) - ready for immediate analysis")
            elif quality_score >= 85:
                insights.append(f"Good data quality ({quality_score:.1f}%) with {len(profile.quality_profile.warning_alerts)} minor issues")
            elif quality_score < 70:
                critical_count = len(profile.quality_profile.critical_alerts)
                insights.append(f"Data quality needs attention ({quality_score:.1f}%) - {critical_count} critical issues detected")
        
        # Column structure and feature insights
        measure_ratio = len(profile.measure_columns) / profile.column_count if profile.column_count > 0 else 0
        dimension_ratio = len(profile.dimension_columns) / profile.column_count if profile.column_count > 0 else 0
        
        if measure_ratio > 0.6:
            insights.append(f"Measure-heavy table ({len(profile.measure_columns)} metrics) - ideal for quantitative analysis and KPI tracking")
        elif dimension_ratio > 0.6:
            insights.append(f"Dimension-rich table ({len(profile.dimension_columns)} attributes) - excellent for segmentation and filtering")
        elif profile.temporal_columns:
            insights.append(f"Time-series capable with {len(profile.temporal_columns)} temporal columns - suitable for trend analysis")
        
        # Business domain and purpose insights
        if profile.business_domain:
            domain_insight = f"Classified as {profile.business_domain} domain"
            if profile.table_type == 'fact':
                domain_insight += " fact table - tracks business events and transactions"
            elif profile.table_type == 'dimension':
                domain_insight += " dimension table - provides context for analysis"
            insights.append(domain_insight)
        
        # ML readiness insights with specific recommendations
        if profile.ml_readiness_score:
            if profile.ml_readiness_score >= 80:
                insights.append(f"High ML readiness ({profile.ml_readiness_score}%) - ready for predictive modeling")
            elif profile.ml_readiness_score >= 60:
                insights.append(f"Moderate ML readiness ({profile.ml_readiness_score}%) - feature engineering recommended")
            else:
                insights.append(f"Limited ML readiness ({profile.ml_readiness_score}%) - data preparation required")
        
        # Outlier analysis insights
        if profile.outlier_analysis and profile.outlier_analysis.get('high_impact_outliers'):
            outlier_count = len(profile.outlier_analysis['high_impact_outliers'])
            if outlier_count > 0:
                insights.append(f"⚠️ {outlier_count} columns have significant outliers affecting data distribution")
        
        # Correlation insights for feature relationships
        if profile.correlation_analysis:
            redundant_count = len(profile.correlation_analysis.get('feature_redundancy', []))
            strong_corr_count = len(profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', []))
            
            if redundant_count > 0:
                insights.append(f"🔄 {redundant_count} redundant features detected - consider dimensionality reduction")
            elif strong_corr_count > 0:
                insights.append(f"📊 {strong_corr_count} strong feature relationships identified - useful for predictive modeling")
        
        # Data freshness and temporal insights
        if profile.temporal_columns and hasattr(profile, 'temporal_analysis'):
            temporal_info = profile.temporal_analysis
            if temporal_info and 'data_freshness' in temporal_info:
                insights.append(f"📅 Data freshness: {temporal_info['data_freshness']}")
        
        # Specific quality alerts
        if profile.quality_profile and profile.quality_profile.critical_alerts:
            most_critical = profile.quality_profile.critical_alerts[0]
            alert_type = most_critical.get('alert_type', 'Unknown')
            insights.append(f"🚨 Critical quality alert: {alert_type}")
        
        # Return top 5 most relevant insights
        return insights[:5]