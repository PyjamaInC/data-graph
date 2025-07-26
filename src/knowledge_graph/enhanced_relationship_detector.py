"""
Enhanced Relationship Detector using ydata-profiling

This module leverages ydata-profiling's comprehensive analysis capabilities
to detect relationships between columns more efficiently.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ydata_profiling import ProfileReport
from ydata_profiling.model.correlations import calculate_correlation
import logging


class ProfilingBasedRelationshipDetector:
    """
    Detects relationships between columns using ydata-profiling's
    built-in correlation and association analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._profile_cache = {}
    
    def detect_dataset_relationships(self, data: Dict[str, pd.DataFrame], 
                                   sample_size: int = 5000) -> Dict[str, Any]:
        """
        Detect all relationships in a dataset using profiling
        
        Args:
            data: Dictionary of table_name -> DataFrame
            sample_size: Number of rows to sample for profiling
            
        Returns:
            Dictionary containing discovered relationships and insights
        """
        # Combine all tables for comprehensive profiling
        combined_df = self._prepare_combined_dataset(data, sample_size)
        
        # Generate comprehensive profile
        self.logger.info("Generating comprehensive profile for relationship detection...")
        profile = ProfileReport(
            combined_df,
            title="Relationship Discovery Profile",
            correlations={
                "auto": {"calculate": True},  # Auto-detect correlation method
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},  # Categorical correlations
                "cramers": {"calculate": True}  # Categorical associations
            },
            interactions={"continuous": True},  # Detect interactions
            missing_diagrams={"heatmap": True},  # Missing value patterns
            explorative=True
        )
        
        # Extract relationships from profile
        profile_data = profile.get_description()
        relationships = self._extract_relationships_from_profile(profile_data, data)
        
        return relationships
    
    def _prepare_combined_dataset(self, data: Dict[str, pd.DataFrame], 
                                sample_size: int) -> pd.DataFrame:
        """Prepare a combined dataset for profiling with table prefixes"""
        combined_dfs = []
        
        for table_name, df in data.items():
            # Sample if needed
            sampled_df = df.sample(min(sample_size, len(df)), random_state=42)
            
            # Prefix columns with table name
            sampled_df = sampled_df.add_prefix(f"{table_name}.")
            combined_dfs.append(sampled_df)
        
        # Combine all tables
        if len(combined_dfs) == 1:
            return combined_dfs[0]
        
        # For multiple tables, concatenate along columns
        # This allows correlation analysis across tables
        combined_df = pd.concat(combined_dfs, axis=1)
        
        return combined_df
    
    def _extract_relationships_from_profile(self, profile_data: Dict, 
                                          original_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract relationships from profiling results"""
        relationships = {
            'correlations': {},
            'associations': {},
            'foreign_keys': [],
            'patterns': [],
            'insights': []
        }
        
        # Extract correlations
        correlations = profile_data.get('correlations', {})
        for corr_type, corr_data in correlations.items():
            if isinstance(corr_data, dict) and 'matrix' in corr_data:
                matrix = corr_data['matrix']
                relationships['correlations'][corr_type] = self._process_correlation_matrix(
                    matrix, threshold=0.5
                )
        
        # Extract variable insights
        variables = profile_data.get('variables', {})
        relationships['variable_insights'] = self._extract_variable_insights(variables)
        
        # Detect foreign key relationships
        relationships['foreign_keys'] = self._detect_foreign_keys_from_profile(
            variables, original_data
        )
        
        # Extract interaction insights
        if 'interactions' in profile_data:
            relationships['interactions'] = profile_data['interactions']
        
        # Extract missing value patterns
        if 'missing' in profile_data:
            relationships['missing_patterns'] = self._analyze_missing_patterns(
                profile_data['missing']
            )
        
        return relationships
    
    def _process_correlation_matrix(self, matrix: pd.DataFrame, 
                                  threshold: float = 0.5) -> List[Dict]:
        """Process correlation matrix to extract significant relationships"""
        relationships = []
        
        # Get upper triangle to avoid duplicates
        for i in range(len(matrix.columns)):
            for j in range(i + 1, len(matrix.columns)):
                col1, col2 = matrix.columns[i], matrix.columns[j]
                corr_value = matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    relationships.append({
                        'source': col1,
                        'target': col2,
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate',
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return relationships
    
    def _extract_variable_insights(self, variables: Dict) -> Dict[str, Any]:
        """Extract insights about each variable from profiling"""
        insights = {}
        
        for var_name, var_data in variables.items():
            insights[var_name] = {
                'type': var_data.get('type', 'Unknown'),
                'distinct_count': var_data.get('n_distinct', 0),
                'missing_count': var_data.get('n_missing', 0),
                'is_unique': var_data.get('is_unique', False),
                'memory_size': var_data.get('memory_size', 0)
            }
            
            # Add type-specific insights
            if var_data.get('type') == 'Numeric':
                stats = var_data.get('statistics', {})
                insights[var_name]['numeric_stats'] = {
                    'mean': stats.get('mean'),
                    'std': stats.get('std'),
                    'skewness': stats.get('skewness'),
                    'kurtosis': stats.get('kurtosis')
                }
            elif var_data.get('type') == 'Categorical':
                insights[var_name]['top_categories'] = list(
                    var_data.get('value_counts_index_sorted', [])[:5]
                )
        
        return insights
    
    def _detect_foreign_keys_from_profile(self, variables: Dict, 
                                        original_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Detect potential foreign key relationships using profiling insights"""
        foreign_keys = []
        
        # Group variables by table
        table_columns = {}
        for var_name in variables:
            if '.' in var_name:
                table, column = var_name.split('.', 1)
                if table not in table_columns:
                    table_columns[table] = []
                table_columns[table].append(column)
        
        # Look for ID-like columns and their references
        for table1, columns1 in table_columns.items():
            for col1 in columns1:
                var1_data = variables.get(f"{table1}.{col1}", {})
                
                # Check if it's an ID-like column
                if (col1.lower().endswith('_id') or col1.lower() == 'id') and \
                   var1_data.get('is_unique', False):
                    
                    # Look for references in other tables
                    for table2, columns2 in table_columns.items():
                        if table1 != table2:
                            for col2 in columns2:
                                # Check if column name suggests FK
                                if col2.lower() == f"{table1}_id" or \
                                   col2.lower() == f"{table1.rstrip('s')}_id":
                                    
                                    var2_data = variables.get(f"{table2}.{col2}", {})
                                    
                                    # Verify with value overlap
                                    overlap_ratio = self._calculate_value_overlap(
                                        original_data.get(table1, pd.DataFrame()),
                                        original_data.get(table2, pd.DataFrame()),
                                        col1, col2
                                    )
                                    
                                    if overlap_ratio > 0.8:
                                        foreign_keys.append({
                                            'source_table': table2,
                                            'source_column': col2,
                                            'target_table': table1,
                                            'target_column': col1,
                                            'confidence': overlap_ratio,
                                            'type': 'foreign_key'
                                        })
        
        return foreign_keys
    
    def _calculate_value_overlap(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               col1: str, col2: str) -> float:
        """Calculate value overlap between two columns"""
        if df1.empty or df2.empty or col1 not in df1.columns or col2 not in df2.columns:
            return 0.0
        
        values1 = set(df1[col1].dropna().unique())
        values2 = set(df2[col2].dropna().unique())
        
        if not values1 or not values2:
            return 0.0
        
        intersection = values1.intersection(values2)
        return len(intersection) / len(values2)  # Ratio of df2 values found in df1
    
    def _analyze_missing_patterns(self, missing_data: Dict) -> Dict[str, Any]:
        """Analyze missing value patterns for insights"""
        patterns = {
            'correlated_missing': [],
            'high_missing_columns': []
        }
        
        # Extract columns with high missing rates
        if 'count' in missing_data:
            for col, count in missing_data['count'].items():
                if count > 0:
                    patterns['high_missing_columns'].append({
                        'column': col,
                        'missing_count': count,
                        'missing_percentage': missing_data.get('percentage', {}).get(col, 0)
                    })
        
        # Look for correlated missing patterns
        if 'matrix' in missing_data:
            # This would contain missing value correlations
            patterns['missing_correlations'] = missing_data['matrix']
        
        return patterns
    
    def generate_analysis_recommendations(self, relationships: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on discovered relationships"""
        recommendations = []
        
        # Check correlations
        for corr_type, correlations in relationships.get('correlations', {}).items():
            if correlations:
                recommendations.append(
                    f"Found {len(correlations)} significant {corr_type} correlations. "
                    f"Consider using these for feature engineering or predictive modeling."
                )
        
        # Check foreign keys
        fks = relationships.get('foreign_keys', [])
        if fks:
            recommendations.append(
                f"Discovered {len(fks)} potential foreign key relationships. "
                f"These can be used for joining tables in analysis."
            )
        
        # Check missing patterns
        missing = relationships.get('missing_patterns', {})
        high_missing = missing.get('high_missing_columns', [])
        if high_missing:
            cols_over_50 = [col['column'] for col in high_missing 
                          if col['missing_percentage'] > 50]
            if cols_over_50:
                recommendations.append(
                    f"Columns {cols_over_50[:3]} have >50% missing values. "
                    f"Consider imputation or exclusion strategies."
                )
        
        return recommendations