"""
Enhanced Data Exploration ReAct Agent - Comprehensive Intelligence Integration

This agent provides intelligent, context-aware data exploration with:
- Question-specific analysis strategies
- Context preservation between iterations
- Intelligent result interpretation
- Business-relevant insight generation
"""

import pandas as pd
import numpy as np
import json
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, SystemMessage
import networkx as nx
from datetime import datetime, timedelta

from .data_exploration_agent import DataExplorationReActAgent, DataExplorationToolkit, ExplorationState


@dataclass
class IntelligentExplorationState(ExplorationState):
    """Enhanced exploration state with comprehensive intelligence context"""
    
    # Intelligence components
    enhanced_profiles: Dict[str, Any] = field(default_factory=dict)
    table_relationships: Dict[str, Any] = field(default_factory=dict)
    semantic_graph: Optional[nx.MultiDiGraph] = None
    
    # Question-specific context
    question_analysis: Dict[str, Any] = field(default_factory=dict)
    analysis_strategy: Dict[str, Any] = field(default_factory=dict)
    target_insights: List[str] = field(default_factory=list)
    
    # Iteration context preservation
    previous_operations: List[Dict[str, Any]] = field(default_factory=list)
    accumulated_insights: List[Dict[str, Any]] = field(default_factory=list)
    context_summary: str = ""
    
    # Quality and intelligence tracking
    quality_concerns: List[str] = field(default_factory=list)
    intelligence_flags: Dict[str, Any] = field(default_factory=dict)
    data_limitations: List[str] = field(default_factory=list)
    
    # Business context
    business_domain: str = "general"
    key_metrics: List[str] = field(default_factory=list)
    stakeholder_needs: List[str] = field(default_factory=list)


class QuestionAnalyzer:
    """Analyzes user questions to determine analysis strategy"""
    
    def __init__(self):
        self.intent_patterns = {
            'temporal': {
                'keywords': ['seasonal', 'trend', 'over time', 'monthly', 'daily', 'pattern', 'when', 'delivery', 'time'],
                'required_data': ['temporal_columns'],
                'analysis_type': 'time_series'
            },
            'correlation': {
                'keywords': ['correlation', 'relationship', 'impact', 'influence', 'factor', 'affect', 'between'],
                'required_data': ['numeric_columns', 'multiple_tables'],
                'analysis_type': 'correlation'
            },
            'segmentation': {
                'keywords': ['segment', 'group', 'category', 'type', 'which', 'breakdown', 'customer', 'behavior'],
                'required_data': ['categorical_columns'],
                'analysis_type': 'segmentation'
            },
            'quality': {
                'keywords': ['quality', 'missing', 'complete', 'clean', 'issues', 'problems'],
                'required_data': ['all_columns'],
                'analysis_type': 'quality_assessment'
            },
            'outlier': {
                'keywords': ['unusual', 'outlier', 'abnormal', 'unexpected', 'strange', 'anomaly', 'pricing'],
                'required_data': ['numeric_columns'],
                'analysis_type': 'outlier_detection'
            },
            'aggregation': {
                'keywords': ['total', 'sum', 'average', 'count', 'distribution', 'how many', 'performance'],
                'required_data': ['numeric_columns'],
                'analysis_type': 'aggregation'
            }
        }
    
    def analyze_question(self, question: str, profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive question analysis"""
        
        question_lower = question.lower()
        
        # Detect primary and secondary intents
        detected_intents = []
        for intent, pattern in self.intent_patterns.items():
            if any(keyword in question_lower for keyword in pattern['keywords']):
                detected_intents.append({
                    'intent': intent,
                    'confidence': self._calculate_intent_confidence(question_lower, pattern['keywords']),
                    'pattern': pattern
                })
        
        # Sort by confidence
        detected_intents.sort(key=lambda x: x['confidence'], reverse=True)
        
        primary_intent = detected_intents[0] if detected_intents else {
            'intent': 'exploration',
            'confidence': 0.5,
            'pattern': {'analysis_type': 'general_discovery'}
        }
        
        # Assess data requirements and viability
        data_requirements = self._assess_data_requirements(primary_intent, profiles)
        
        # Generate analysis strategy
        strategy = self._generate_analysis_strategy(primary_intent, data_requirements, profiles)
        
        return {
            'primary_intent': primary_intent,
            'secondary_intents': detected_intents[1:3],
            'data_requirements': data_requirements,
            'analysis_strategy': strategy,
            'question_complexity': self._assess_complexity(question, detected_intents),
            'expected_insights': self._predict_expected_insights(primary_intent, question)
        }
    
    def _calculate_intent_confidence(self, question: str, keywords: List[str]) -> float:
        """Calculate confidence in intent detection"""
        matches = sum(1 for keyword in keywords if keyword in question)
        return min(1.0, matches / len(keywords) * 2)  # Normalize to 0-1
    
    def _assess_data_requirements(self, primary_intent: Dict[str, Any], profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Assess what data is needed and available"""
        
        required_data = primary_intent['pattern']['required_data']
        available_data = {}
        
        for table_name, profile in profiles.items():
            table_capabilities = {
                'temporal_columns': hasattr(profile, 'temporal_columns') and profile.temporal_columns,
                'numeric_columns': hasattr(profile, 'measure_columns') and profile.measure_columns,
                'categorical_columns': hasattr(profile, 'dimension_columns') and profile.dimension_columns,
                'all_columns': True
            }
            
            available_data[table_name] = {
                capability: table_capabilities.get(capability, False)
                for capability in required_data
            }
        
        # Check if requirements are met
        requirements_met = True
        for req in required_data:
            if req == 'multiple_tables':
                # Special case: check if we have multiple tables
                requirements_met = requirements_met and len(profiles) >= 2
            else:
                # Regular capability check
                requirements_met = requirements_met and any(
                    table_data.get(req, False) for table_data in available_data.values()
                )
        
        return {
            'required': required_data,
            'available': available_data,
            'met': requirements_met,
            'gaps': self._identify_data_gaps(required_data, available_data)
        }
    
    def _identify_data_gaps(self, required: List[str], available: Dict[str, Any]) -> List[str]:
        """Identify missing data requirements"""
        gaps = []
        
        for req in required:
            if req == 'multiple_tables':
                # Special case: check if we have multiple tables
                if len(available) < 2:
                    gaps.append(f"Missing {req}")
            else:
                # Regular capability check
                if not any(table_data.get(req, False) for table_data in available.values()):
                    gaps.append(f"Missing {req}")
        
        return gaps
    
    def _generate_analysis_strategy(self, primary_intent: Dict[str, Any], 
                                  data_requirements: Dict[str, Any], 
                                  profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific analysis strategy"""
        
        analysis_type = primary_intent['pattern']['analysis_type']
        
        strategies = {
            'time_series': {
                'approach': 'temporal_analysis',
                'steps': [
                    'identify_temporal_columns',
                    'check_temporal_coverage',
                    'aggregate_by_time_periods', 
                    'detect_seasonal_patterns',
                    'analyze_trends'
                ],
                'priority_tables': self._find_tables_with_temporal_data(profiles),
                'key_operations': [
                    'temporal_column_identification',
                    'date_range_analysis',
                    'time_series_aggregation',
                    'pattern_detection'
                ]
            },
            'correlation': {
                'approach': 'correlation_analysis',
                'steps': [
                    'identify_numeric_columns',
                    'calculate_correlation_matrix',
                    'identify_strong_relationships',
                    'validate_correlations',
                    'interpret_business_impact'
                ],
                'priority_tables': self._find_tables_with_numeric_data(profiles),
                'key_operations': [
                    'correlation_matrix',
                    'relationship_analysis',
                    'statistical_validation',
                    'business_interpretation'
                ]
            },
            'segmentation': {
                'approach': 'segmentation_analysis',
                'steps': [
                    'identify_categorical_columns',
                    'calculate_segment_metrics',
                    'compare_segments',
                    'identify_key_differentiators',
                    'analyze_segment_behavior'
                ],
                'priority_tables': self._find_tables_with_categorical_data(profiles),
                'key_operations': [
                    'categorical_analysis',
                    'segment_metrics',
                    'comparative_analysis',
                    'behavior_patterns'
                ]
            },
            'quality_assessment': {
                'approach': 'quality_analysis',
                'steps': [
                    'assess_data_completeness',
                    'identify_missing_values',
                    'check_data_types',
                    'detect_anomalies',
                    'generate_quality_report'
                ],
                'priority_tables': list(profiles.keys()),
                'key_operations': [
                    'completeness_check',
                    'missing_value_analysis',
                    'data_type_validation',
                    'quality_scoring'
                ]
            },
            'outlier_detection': {
                'approach': 'outlier_analysis',
                'steps': [
                    'identify_numeric_columns',
                    'calculate_statistical_bounds',
                    'detect_outliers',
                    'analyze_outlier_patterns',
                    'assess_business_impact'
                ],
                'priority_tables': self._find_tables_with_numeric_data(profiles),
                'key_operations': [
                    'statistical_analysis',
                    'outlier_detection',
                    'pattern_analysis',
                    'impact_assessment'
                ]
            },
            'aggregation': {
                'approach': 'aggregation_analysis',
                'steps': [
                    'identify_aggregation_columns',
                    'calculate_summary_statistics',
                    'analyze_distributions',
                    'identify_performance_metrics',
                    'generate_insights'
                ],
                'priority_tables': self._find_tables_with_numeric_data(profiles),
                'key_operations': [
                    'summary_statistics',
                    'distribution_analysis',
                    'performance_metrics',
                    'insight_generation'
                ]
            }
        }
        
        strategy = strategies.get(analysis_type, strategies['aggregation'])
        
        # Add data requirement validation
        if not data_requirements['met']:
            strategy['limitations'] = data_requirements['gaps']
            strategy['fallback_approach'] = 'general_discovery'
        
        return strategy
    
    def _find_tables_with_temporal_data(self, profiles: Dict[str, Any]) -> List[str]:
        """Find tables with temporal columns"""
        return [
            table_name for table_name, profile in profiles.items()
            if hasattr(profile, 'temporal_columns') and profile.temporal_columns
        ]
    
    def _find_tables_with_numeric_data(self, profiles: Dict[str, Any]) -> List[str]:
        """Find tables with numeric columns"""
        return [
            table_name for table_name, profile in profiles.items()
            if hasattr(profile, 'measure_columns') and profile.measure_columns
        ]
    
    def _find_tables_with_categorical_data(self, profiles: Dict[str, Any]) -> List[str]:
        """Find tables with categorical columns"""
        return [
            table_name for table_name, profile in profiles.items()
            if hasattr(profile, 'dimension_columns') and profile.dimension_columns
        ]
    
    def _assess_complexity(self, question: str, intents: List[Dict[str, Any]]) -> str:
        """Assess question complexity"""
        if len(intents) > 2:
            return 'high'
        elif any(intent['intent'] in ['correlation', 'outlier'] for intent in intents):
            return 'medium'
        else:
            return 'low'
    
    def _predict_expected_insights(self, primary_intent: Dict[str, Any], question: str) -> List[str]:
        """Predict what insights the user expects"""
        
        intent = primary_intent['intent']
        insights = {
            'temporal': [
                'Seasonal patterns and trends',
                'Time-based performance metrics',
                'Temporal correlations and cycles'
            ],
            'correlation': [
                'Relationships between variables',
                'Impact factors and drivers',
                'Statistical associations'
            ],
            'segmentation': [
                'Segment performance differences',
                'Behavioral patterns by group',
                'Key segment characteristics'
            ],
            'quality': [
                'Data completeness issues',
                'Quality problems and gaps',
                'Data reliability assessment'
            ],
            'outlier': [
                'Unusual data points',
                'Anomaly patterns',
                'Outlier impact analysis'
            ],
            'aggregation': [
                'Summary statistics',
                'Performance metrics',
                'Distribution patterns'
            ]
        }
        
        return insights.get(intent, ['General data patterns'])


class IntelligentOperationGenerator:
    """Generates context-aware operations based on analysis strategy"""
    
    def __init__(self, profiles: Dict[str, Any], question_analysis: Dict[str, Any], 
                 catalog_entries: Optional[Dict[str, Any]] = None):
        self.profiles = profiles
        self.question_analysis = question_analysis
        self.strategy = question_analysis['analysis_strategy']
        self.operation_history = []
        self.catalog_entries = catalog_entries or {}
    
    def generate_next_operation(self, iteration: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the next operation based on strategy and context"""
        
        current_step = self._get_current_step(iteration)
        target_tables = self._select_target_tables(context)
        
        # Consider semantic relationships when generating operations
        operation = self._generate_operation_for_step(current_step, target_tables, context)
        
        # Add relationship context
        relevant_relationships = self._get_relevant_relationships(target_tables, context)
        
        # Get catalog recommendations if available
        catalog_recommendations = self._get_catalog_recommendations(target_tables, current_step)
        
        return {
            'operation': operation,
            'step': current_step,
            'target_tables': target_tables,
            'expected_insight': self._predict_insight(current_step),
            'reasoning': self._explain_operation_choice(current_step, context),
            'relationship_context': relevant_relationships,
            'catalog_hints': catalog_recommendations
        }
    
    def _get_current_step(self, iteration: int) -> str:
        """Get the current step based on iteration"""
        steps = self.strategy.get('steps', [])
        if iteration - 1 < len(steps):
            return steps[iteration - 1]
        return 'continue_analysis'
    
    def _select_target_tables(self, context: Dict[str, Any]) -> List[str]:
        """Select target tables based on context and strategy"""
        
        priority_tables = self.strategy.get('priority_tables', [])
        
        # If we have previous operations, consider their results
        if context.get('previous_operations'):
            last_operation = context['previous_operations'][-1]
            if last_operation.get('success') and last_operation.get('target_tables'):
                # Continue with the same table if it was successful
                return last_operation['target_tables']
        
        # Return priority tables or all tables if none specified
        return priority_tables if priority_tables else list(self.profiles.keys())
    
    def _generate_operation_for_step(self, step: str, target_tables: List[str], 
                                   context: Dict[str, Any]) -> str:
        """Generate specific operation for the step"""
        
        if not target_tables:
            return "print('No suitable tables found for analysis')"
        
        table = target_tables[0]  # Focus on primary table
        
        operations = {
            'identify_temporal_columns': self._generate_temporal_column_operation(table),
            'check_temporal_coverage': self._generate_temporal_coverage_operation(table),
            'aggregate_by_time_periods': self._generate_temporal_aggregation_operation(table),
            'detect_seasonal_patterns': self._generate_seasonal_pattern_operation(table),
            'analyze_trends': self._generate_trend_analysis_operation(table),
            
            'identify_numeric_columns': self._generate_numeric_column_operation(table),
            'calculate_correlation_matrix': self._generate_correlation_operation(table),
            'identify_strong_relationships': self._generate_relationship_analysis_operation(table),
            'validate_correlations': self._generate_correlation_validation_operation(table),
            'interpret_business_impact': self._generate_business_interpretation_operation(table),
            
            'identify_categorical_columns': self._generate_categorical_column_operation(table),
            'calculate_segment_metrics': self._generate_segment_metrics_operation(table),
            'compare_segments': self._generate_segment_comparison_operation(table),
            'identify_key_differentiators': self._generate_differentiator_operation(table),
            'analyze_segment_behavior': self._generate_behavior_analysis_operation(table),
            
            'assess_data_completeness': self._generate_completeness_operation(table),
            'identify_missing_values': self._generate_missing_value_operation(table),
            'check_data_types': self._generate_data_type_operation(table),
            'detect_anomalies': self._generate_anomaly_detection_operation(table),
            'generate_quality_report': self._generate_quality_report_operation(table),
            
            'calculate_statistical_bounds': self._generate_statistical_bounds_operation(table),
            'detect_outliers': self._generate_outlier_detection_operation(table),
            'analyze_outlier_patterns': self._generate_outlier_pattern_operation(table),
            'assess_business_impact': self._generate_impact_assessment_operation(table),
            
            'identify_aggregation_columns': self._generate_aggregation_column_operation(table),
            'calculate_summary_statistics': self._generate_summary_statistics_operation(table),
            'analyze_distributions': self._generate_distribution_analysis_operation(table),
            'identify_performance_metrics': self._generate_performance_metrics_operation(table),
            'generate_insights': self._generate_insight_generation_operation(table),
            
            'continue_analysis': self._generate_continuation_operation(table, context)
        }
        
        return operations.get(step, f"tables['{table}'].info()")
    
    def _generate_temporal_column_operation(self, table: str) -> str:
        """Generate operation to identify temporal columns"""
        profile = self.profiles.get(table)
        if profile and hasattr(profile, 'temporal_columns') and profile.temporal_columns:
            cols = profile.temporal_columns
            return f"pd.DataFrame({{'temporal_columns': {cols}, 'table': '{table}'}})"
        return f"tables['{table}'].select_dtypes(include=['datetime64']).columns.tolist()"
    
    def _generate_temporal_coverage_operation(self, table: str) -> str:
        """Generate operation to check temporal coverage"""
        profile = self.profiles.get(table)
        if profile and hasattr(profile, 'temporal_columns') and profile.temporal_columns:
            col = profile.temporal_columns[0]
            return f"""temp_df = tables['{table}']
if '{col}' in temp_df.columns:
    date_range = pd.DataFrame({{
        'column': ['{col}'],
        'min_date': [temp_df['{col}'].min()],
        'max_date': [temp_df['{col}'].max()],
        'date_range_days': [(temp_df['{col}'].max() - temp_df['{col}'].min()).days],
        'total_records': [len(temp_df)]
    }})
    date_range
else:
    pd.DataFrame({{'error': ['No temporal column found']}})"""
        return f"tables['{table}'].info()"
    
    def _generate_temporal_aggregation_operation(self, table: str) -> str:
        """Generate temporal aggregation operation"""
        profile = self.profiles.get(table)
        if profile and hasattr(profile, 'temporal_columns') and profile.temporal_columns:
            date_col = profile.temporal_columns[0]
            # Find a value column for aggregation
            value_col = None
            if hasattr(profile, 'measure_columns') and profile.measure_columns:
                value_col = profile.measure_columns[0]
            
            if value_col:
                return f"""temp_df = tables['{table}']
if '{date_col}' in temp_df.columns and '{value_col}' in temp_df.columns:
    # Convert to datetime if needed
    temp_df['{date_col}'] = pd.to_datetime(temp_df['{date_col}'])
    # Aggregate by month
    monthly_agg = temp_df.groupby(temp_df['{date_col}'].dt.to_period('M'))['{value_col}'].agg(['count', 'sum', 'mean', 'std'])
    monthly_agg.reset_index()
else:
    pd.DataFrame({{'error': ['Missing required columns']}})"""
            else:
                return f"""temp_df = tables['{table}']
if '{date_col}' in temp_df.columns:
    temp_df['{date_col}'] = pd.to_datetime(temp_df['{date_col}'])
    temp_df.groupby(temp_df['{date_col}'].dt.to_period('M')).size().reset_index(name='count')
else:
    pd.DataFrame({{'error': ['No temporal column found']}})"""
        return f"tables['{table}'].describe()"
    
    def _generate_seasonal_pattern_operation(self, table: str) -> str:
        """Generate seasonal pattern detection operation"""
        return self._generate_temporal_aggregation_operation(table)
    
    def _generate_trend_analysis_operation(self, table: str) -> str:
        """Generate trend analysis operation"""
        return self._generate_temporal_aggregation_operation(table)
    
    def _generate_correlation_operation(self, table: str) -> str:
        """Generate correlation analysis operation with relationship awareness"""
        profile = self.profiles.get(table)
        
        # Check catalog for column insights
        if table in self.catalog_entries:
            catalog = self.catalog_entries[table]
            # Use catalog to identify important columns
            important_cols = []
            for term in catalog.business_glossary:
                if term['type'] == 'column' and 'value' in term['term'].lower():
                    important_cols.append(term['term'])
        
        if profile and hasattr(profile, 'measure_columns') and profile.measure_columns:
            # Prioritize columns mentioned in the question analysis
            question_keywords = self.question_analysis.get('primary_intent', {}).get('pattern', {}).get('keywords', [])
            
            # Smart column selection based on question context
            relevant_cols = []
            for col in profile.measure_columns:
                col_lower = col.lower()
                # Check if column relates to order values, shipping, or delivery
                if any(keyword in col_lower for keyword in ['price', 'value', 'cost', 'freight', 'shipping', 'delivery', 'total']):
                    relevant_cols.append(col)
            
            if relevant_cols:
                return f"tables['{table}'][{relevant_cols}].corr()"
            else:
                numeric_cols = profile.measure_columns[:5]
                return f"tables['{table}'][{numeric_cols}].corr()"
        
        return f"tables['{table}'].select_dtypes(include=[np.number]).corr()"
    
    def _generate_relationship_analysis_operation(self, table: str) -> str:
        """Generate relationship analysis operation"""
        return self._generate_correlation_operation(table)
    
    def _generate_correlation_validation_operation(self, table: str) -> str:
        """Generate correlation validation operation"""
        return self._generate_correlation_operation(table)
    
    def _generate_business_interpretation_operation(self, table: str) -> str:
        """Generate business interpretation operation"""
        return self._generate_correlation_operation(table)
    
    def _generate_numeric_column_operation(self, table: str) -> str:
        """Generate numeric column identification operation"""
        return f"tables['{table}'].select_dtypes(include=[np.number]).columns.tolist()"
    
    def _generate_categorical_column_operation(self, table: str) -> str:
        """Generate categorical column identification operation"""
        return f"tables['{table}'].select_dtypes(include=['object', 'category']).columns.tolist()"
    
    def _generate_segment_metrics_operation(self, table: str) -> str:
        """Generate segment metrics operation"""
        profile = self.profiles.get(table)
        if profile:
            cat_col = None
            num_col = None
            
            if hasattr(profile, 'dimension_columns') and profile.dimension_columns:
                cat_col = profile.dimension_columns[0]
            if hasattr(profile, 'measure_columns') and profile.measure_columns:
                num_col = profile.measure_columns[0]
            
            if cat_col and num_col:
                return f"tables['{table}'].groupby('{cat_col}')['{num_col}'].agg(['count', 'mean', 'sum', 'std', 'min', 'max'])"
            elif cat_col:
                return f"tables['{table}']['{cat_col}'].value_counts()"
        
        return f"tables['{table}'].describe(include='all')"
    
    def _generate_segment_comparison_operation(self, table: str) -> str:
        """Generate segment comparison operation"""
        return self._generate_segment_metrics_operation(table)
    
    def _generate_differentiator_operation(self, table: str) -> str:
        """Generate key differentiator operation"""
        return self._generate_segment_metrics_operation(table)
    
    def _generate_behavior_analysis_operation(self, table: str) -> str:
        """Generate behavior analysis operation"""
        return self._generate_segment_metrics_operation(table)
    
    def _generate_completeness_operation(self, table: str) -> str:
        """Generate completeness check operation"""
        return self._generate_missing_value_operation(table)
    
    def _generate_missing_value_operation(self, table: str) -> str:
        """Generate missing value analysis operation"""
        return f"""missing_analysis = pd.DataFrame({{
    'column': tables['{table}'].columns,
    'null_count': tables['{table}'].isnull().sum(),
    'null_pct': tables['{table}'].isnull().sum() / len(tables['{table}']) * 100,
    'data_type': tables['{table}'].dtypes
}}).sort_values('null_pct', ascending=False)
missing_analysis"""
    
    def _generate_data_type_operation(self, table: str) -> str:
        """Generate data type check operation"""
        return f"tables['{table}'].dtypes"
    
    def _generate_anomaly_detection_operation(self, table: str) -> str:
        """Generate anomaly detection operation"""
        return self._generate_outlier_detection_operation(table)
    
    def _generate_quality_report_operation(self, table: str) -> str:
        """Generate quality report operation"""
        return self._generate_missing_value_operation(table)
    
    def _generate_statistical_bounds_operation(self, table: str) -> str:
        """Generate statistical bounds operation"""
        return f"tables['{table}'].describe()"
    
    def _generate_outlier_detection_operation(self, table: str) -> str:
        """Generate outlier detection operation with context awareness"""
        profile = self.profiles.get(table)
        
        if profile and hasattr(profile, 'measure_columns') and profile.measure_columns:
            # Select column based on question context
            target_col = None
            
            # Look for pricing-related columns if that's the focus
            question_keywords = self.question_analysis.get('primary_intent', {}).get('pattern', {}).get('keywords', [])
            if any(keyword in ['pricing', 'price', 'cost'] for keyword in question_keywords):
                for col in profile.measure_columns:
                    if 'price' in col.lower() or 'cost' in col.lower() or 'value' in col.lower():
                        target_col = col
                        break
            
            # Fallback to first measure column
            if not target_col:
                target_col = profile.measure_columns[0]
            
            return f"""# Context-aware outlier detection for {target_col}
col_data = tables['{table}']['{target_col}'].dropna()
Q1 = col_data.quantile(0.25)
Q3 = col_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
outlier_analysis = pd.DataFrame({{
    'metric': ['total_count', 'outlier_count', 'outlier_pct', 'lower_bound', 'upper_bound', 'mean_value', 'outlier_mean'],
    'value': [len(col_data), len(outliers), len(outliers)/len(col_data)*100, lower_bound, upper_bound, col_data.mean(), outliers.mean() if len(outliers) > 0 else 0]
}})
outlier_analysis"""
        
        return f"tables['{table}'].describe()"
    
    def _generate_outlier_pattern_operation(self, table: str) -> str:
        """Generate outlier pattern analysis operation"""
        return self._generate_outlier_detection_operation(table)
    
    def _generate_impact_assessment_operation(self, table: str) -> str:
        """Generate impact assessment operation"""
        return self._generate_outlier_detection_operation(table)
    
    def _generate_aggregation_column_operation(self, table: str) -> str:
        """Generate aggregation column operation"""
        return self._generate_numeric_column_operation(table)
    
    def _generate_summary_statistics_operation(self, table: str) -> str:
        """Generate summary statistics operation"""
        return f"tables['{table}'].describe(include='all')"
    
    def _generate_distribution_analysis_operation(self, table: str) -> str:
        """Generate distribution analysis operation"""
        return f"tables['{table}'].describe()"
    
    def _generate_performance_metrics_operation(self, table: str) -> str:
        """Generate performance metrics operation"""
        return f"tables['{table}'].describe()"
    
    def _generate_insight_generation_operation(self, table: str) -> str:
        """Generate insight generation operation"""
        return f"tables['{table}'].info()"
    
    def _generate_continuation_operation(self, table: str, context: Dict[str, Any]) -> str:
        """Generate continuation operation based on context"""
        if context.get('previous_operations'):
            last_op = context['previous_operations'][-1]
            if last_op.get('success'):
                # Build on successful previous operation
                return f"tables['{table}'].describe(include='all')"
        
        return f"tables['{table}'].info()"
    
    def _predict_insight(self, step: str) -> str:
        """Predict what insight this step will provide"""
        predictions = {
            'identify_temporal_columns': 'Finding time-based columns for analysis',
            'check_temporal_coverage': 'Understanding the time span of available data',
            'aggregate_by_time_periods': 'Revealing patterns and trends over time',
            'detect_seasonal_patterns': 'Identifying recurring patterns and seasonality',
            'analyze_trends': 'Understanding long-term trends and changes',
            'identify_numeric_columns': 'Finding quantitative variables for analysis',
            'calculate_correlation_matrix': 'Discovering relationships between variables',
            'identify_strong_relationships': 'Finding the most important correlations',
            'validate_correlations': 'Ensuring correlation findings are reliable',
            'interpret_business_impact': 'Understanding business implications of relationships',
            'identify_categorical_columns': 'Finding dimensions for segmentation',
            'calculate_segment_metrics': 'Understanding performance by segment',
            'compare_segments': 'Identifying differences between segments',
            'identify_key_differentiators': 'Finding what makes segments unique',
            'analyze_segment_behavior': 'Understanding segment-specific patterns',
            'assess_data_completeness': 'Understanding data quality and completeness',
            'identify_missing_values': 'Finding data gaps and quality issues',
            'check_data_types': 'Validating data structure and types',
            'detect_anomalies': 'Finding unusual patterns in the data',
            'generate_quality_report': 'Comprehensive data quality assessment',
            'calculate_statistical_bounds': 'Understanding data distributions',
            'detect_outliers': 'Finding unusual data points',
            'analyze_outlier_patterns': 'Understanding outlier characteristics',
            'assess_business_impact': 'Evaluating business implications of findings',
            'identify_aggregation_columns': 'Finding metrics for aggregation',
            'calculate_summary_statistics': 'Understanding overall data characteristics',
            'analyze_distributions': 'Understanding data spread and patterns',
            'identify_performance_metrics': 'Finding key performance indicators',
            'generate_insights': 'Synthesizing findings into actionable insights'
        }
        return predictions.get(step, 'Gathering data characteristics')
    
    def _explain_operation_choice(self, step: str, context: Dict[str, Any]) -> str:
        """Explain why this operation was chosen"""
        return f"Executing {step} as part of {self.strategy.get('approach', 'analysis')} strategy"
    
    def _get_relevant_relationships(self, target_tables: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant relationships for target tables"""
        
        relationships = context.get('table_relationships', {})
        if not relationships:
            return {}
        
        relevant = {
            'inter_table': [],
            'intra_table': [],
            'key_relationships': []
        }
        
        # Get relationships from context
        if 'top_relationships' in relationships:
            for rel in relationships['top_relationships']:
                # Check if relationship involves our target tables
                for table in target_tables:
                    if table in rel.get('source', '') or table in rel.get('target', ''):
                        relevant['key_relationships'].append({
                            'type': rel.get('type'),
                            'source': rel.get('source'),
                            'target': rel.get('target'),
                            'confidence': rel.get('weight', 0)
                        })
        
        return relevant
    
    def _get_catalog_recommendations(self, target_tables: List[str], current_step: str) -> Dict[str, Any]:
        """Get catalog-based recommendations for the operation"""
        
        recommendations = {}
        
        for table in target_tables:
            if table in self.catalog_entries:
                catalog = self.catalog_entries[table]
                
                # Extract relevant recommendations based on step
                if 'quality' in current_step and catalog.recommended_queries:
                    # Find quality-related queries
                    quality_queries = [q for q in catalog.recommended_queries 
                                     if 'quality' in q.get('title', '').lower()]
                    if quality_queries:
                        recommendations[table] = {
                            'quality_score': catalog.quality_badge['score'],
                            'quality_insights': catalog.quality_badge['insights'],
                            'suggested_analysis': quality_queries[0]
                        }
                
                elif 'correlation' in current_step or 'relationship' in current_step:
                    # Use business glossary to understand columns better
                    key_terms = [term for term in catalog.business_glossary 
                               if term['type'] == 'column']
                    if key_terms:
                        recommendations[table] = {
                            'column_meanings': key_terms,
                            'usage_guide': catalog.usage_guide[:200]  # First 200 chars
                        }
        
        return recommendations


class IntelligentInsightSynthesizer:
    """Synthesizes insights with full context awareness"""
    
    def __init__(self, llm, question_analysis: Dict[str, Any]):
        self.llm = llm
        self.question_analysis = question_analysis
        self.primary_intent = question_analysis['primary_intent']
    
    def synthesize_insights(self, operation_result: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights with full context"""
        
        if not operation_result.get('success'):
            return self._synthesize_error_insight(operation_result)
        
        # Extract actual data from result
        result_data = self._extract_result_data(operation_result)
        
        # Extract concrete findings based on the step and data
        concrete_findings = self._extract_concrete_findings(result_data, context)
        
        # If we have concrete findings, use them directly
        if concrete_findings:
            insights = {
                'findings': concrete_findings,
                'confidence': 0.8 if len(concrete_findings) >= 2 else 0.6,
                'business_relevance': self._determine_relevance(concrete_findings, context),
                'next_steps': self._suggest_next_steps(context.get('step', '')),
                'data_quality_notes': []
            }
        else:
            # Try LLM synthesis as fallback
            prompt = self._create_synthesis_prompt(result_data, context)
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                insights = self._parse_llm_response(response.content)
            except Exception as e:
                insights = self._fallback_synthesis(result_data, context)
        
        # Add context metadata
        insights['context_metadata'] = {
            'iteration': context.get('iteration', 1),
            'step': context.get('step', 'unknown'),
            'intent': self.primary_intent['intent'],
            'question_focus': self.question_analysis.get('expected_insights', []),
            'concrete_findings_count': len(concrete_findings) if concrete_findings else 0
        }
        
        return insights
    
    def _extract_concrete_findings(self, result_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract concrete, specific findings from operation results"""
        
        findings = []
        step = context.get('step', '')
        intent = self.primary_intent['intent']
        
        if result_data['type'] == 'dataframe':
            raw_data = result_data.get('raw_data', [])
            
            # Handle missing value analysis
            if 'missing' in step or 'quality' in intent:
                for row in raw_data[:5]:
                    if isinstance(row, dict) and 'null_pct' in row and 'column' in row:
                        if row['null_pct'] > 0:
                            findings.append(f"Column '{row['column']}' has {row['null_pct']:.1f}% missing values ({row.get('null_count', 'N/A')} nulls)")
                if not any('missing' in f for f in findings):
                    findings.append("No missing values detected - all columns have 100% data completeness")
            
            # Handle correlation analysis
            elif 'correlation' in step or 'correlation' in intent:
                if raw_data and isinstance(raw_data[0], dict):
                    for i, row in enumerate(raw_data):
                        for j, (col, value) in enumerate(row.items()):
                            if i != j and isinstance(value, (int, float)) and abs(value) > 0.7 and abs(value) < 1.0:
                                col1 = result_data.get('columns', [])[i] if i < len(result_data.get('columns', [])) else f"var{i}"
                                findings.append(f"Strong correlation found: {col1} ↔ {col} (r={value:.3f})")
                    if not findings:
                        findings.append("No strong correlations (|r| > 0.7) found between variables")
            
            # Handle outlier analysis
            elif 'outlier' in step or 'outlier' in intent:
                for row in raw_data:
                    if isinstance(row, dict) and 'metric' in row and 'value' in row:
                        metric = row['metric']
                        value = row['value']
                        if metric == 'outlier_count':
                            findings.append(f"Found {int(value)} outliers in the analyzed column")
                        elif metric == 'outlier_pct':
                            findings.append(f"{value:.1f}% of values are outliers")
                        elif metric == 'lower_bound':
                            findings.append(f"Normal range lower bound: {value:.2f}")
                        elif metric == 'upper_bound':
                            findings.append(f"Normal range upper bound: {value:.2f}")
            
            # Handle temporal analysis
            elif 'temporal' in step or 'temporal' in intent:
                for row in raw_data[:1]:  # Usually temporal coverage is one row
                    if isinstance(row, dict):
                        if 'min_date' in row and 'max_date' in row:
                            findings.append(f"Date range: {row['min_date']} to {row['max_date']} ({row.get('date_range_days', 'N/A')} days)")
                        if 'count' in row:
                            findings.append(f"Time period contains {row['count']} records")
            
            # Handle general statistics
            elif 'describe' in step or 'statistics' in step:
                if raw_data:
                    findings.append(f"Analyzed {result_data['shape'][0]} rows across {result_data['shape'][1]} columns")
        
        elif result_data['type'] == 'series':
            series_data = result_data.get('raw_data', {})
            if series_data:
                # For value counts
                sorted_items = sorted(series_data.items(), key=lambda x: x[1], reverse=True)[:3]
                for name, count in sorted_items:
                    findings.append(f"'{name}': {count} occurrences")
        
        elif result_data['type'] == 'collection':
            data = result_data.get('raw_data', [])
            if isinstance(data, list) and data:
                if 'numeric_columns' in step:
                    findings.append(f"Found {len(data)} numeric columns: {', '.join(data[:5])}")
                elif 'temporal_columns' in step:
                    findings.append(f"Found {len(data)} date/time columns: {', '.join(data[:5])}")
                else:
                    findings.append(f"Found {len(data)} items")
        
        return findings
    
    def _determine_relevance(self, findings: List[str], context: Dict[str, Any]) -> str:
        """Determine business relevance based on findings"""
        
        intent = self.primary_intent['intent']
        
        if not findings:
            return "Analysis in progress"
        
        relevance_map = {
            'quality': "Data quality assessment complete - findings highlight completeness and reliability issues",
            'correlation': "Relationship analysis reveals statistical dependencies between variables",
            'outlier': "Anomaly detection identifies data points requiring attention",
            'temporal': "Time-based analysis shows data coverage and patterns",
            'segmentation': "Segment analysis reveals group-specific characteristics",
            'aggregation': "Summary statistics provide data overview"
        }
        
        return relevance_map.get(intent, "Analysis provides data-driven insights")
    
    def _suggest_next_steps(self, current_step: str) -> List[str]:
        """Suggest concrete next steps based on current analysis"""
        
        step_suggestions = {
            'identify_missing_values': ["Investigate root causes of missing data", "Consider imputation strategies"],
            'calculate_correlation_matrix': ["Deep dive into strong correlations", "Test statistical significance"],
            'detect_outliers': ["Investigate outlier causes", "Decide on outlier treatment"],
            'aggregate_by_time_periods': ["Analyze seasonal patterns", "Create time series forecasts"],
            'calculate_segment_metrics': ["Compare segment performance", "Identify optimization opportunities"]
        }
        
        return step_suggestions.get(current_step, ["Continue with detailed analysis"])
    
    def _extract_result_data(self, operation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaningful data from operation result"""
        
        # Fix: Get output directly from operation_result, not from raw_result
        output = operation_result.get('output', {})
        
        if output.get('type') == 'dataframe':
            # Extract DataFrame information with actual data
            shape = output.get('shape', [0, 0])
            columns = output.get('columns', [])
            # Get actual data rows - could be in 'data' or 'head'
            data_rows = output.get('data', output.get('head', []))
            
            return {
                'type': 'dataframe',
                'shape': shape,
                'columns': columns,
                'sample_data': data_rows[:10],  # Get more rows for better analysis
                'summary': self._summarize_dataframe(output),
                'raw_data': data_rows  # Keep all data for concrete findings
            }
        elif output.get('type') == 'series':
            # Extract Series information
            series_data = output.get('data', {})
            return {
                'type': 'series',
                'length': len(series_data),
                'data': series_data,
                'summary': self._summarize_series(series_data),
                'raw_data': series_data
            }
        elif output.get('type') == 'scalar':
            # Extract scalar information
            return {
                'type': 'scalar',
                'value': output.get('value', 'unknown'),
                'summary': f"Value: {output.get('value', 'unknown')}",
                'raw_data': output.get('value')
            }
        elif output.get('type') == 'collection':
            # Handle lists and dicts
            return {
                'type': 'collection',
                'data': output.get('data', []),
                'summary': f"Collection with {len(output.get('data', []))} items",
                'raw_data': output.get('data', [])
            }
        else:
            return {
                'type': 'unknown',
                'raw_output': output,
                'summary': str(output) if output else 'No output',
                'raw_data': output
            }
    
    def _summarize_dataframe(self, df_info: Dict[str, Any]) -> str:
        """Summarize DataFrame content"""
        shape = df_info.get('shape', [0, 0])
        columns = df_info.get('columns', [])
        data = df_info.get('data', [])
        
        summary = f"DataFrame with {shape[0]} rows and {shape[1]} columns"
        if columns:
            summary += f" (columns: {', '.join(columns[:3])}{'...' if len(columns) > 3 else ''})"
        
        if data:
            summary += f". Sample data shows {len(data)} rows."
        
        return summary
    
    def _summarize_series(self, series_data: Dict[str, Any]) -> str:
        """Summarize Series content"""
        length = len(series_data)
        if length == 0:
            return "Empty series"
        
        # Get first few values
        first_values = list(series_data.values())[:3]
        summary = f"Series with {length} values"
        if first_values:
            summary += f". Sample values: {first_values}"
        
        return summary
    
    def _create_synthesis_prompt(self, result_data: Dict[str, Any], 
                               context: Dict[str, Any]) -> str:
        """Create context-aware synthesis prompt"""
        
        user_question = context.get('user_question', '')
        step = context.get('step', '')
        iteration = context.get('iteration', 1)
        previous_insights = context.get('previous_insights', [])
        relationship_context = context.get('relationship_context', {})
        catalog_hints = context.get('catalog_hints', {})
        
        # Build relationship insights
        relationship_insights = ""
        if relationship_context and relationship_context.get('key_relationships'):
            relationship_insights = "\n\nDISCOVERED RELATIONSHIPS:"
            for rel in relationship_context['key_relationships'][:3]:
                relationship_insights += f"\n- {rel['type']}: {rel['source']} ↔ {rel['target']} (confidence: {rel['confidence']:.2f})"
        
        # Build catalog insights
        catalog_insights = ""
        if catalog_hints:
            catalog_insights = "\n\nCATALOG INSIGHTS:"
            for table, hints in catalog_hints.items():
                if 'quality_score' in hints:
                    catalog_insights += f"\n- {table} quality: {hints['quality_score']:.1f}%"
                    if hints.get('quality_insights'):
                        catalog_insights += f" ({', '.join(hints['quality_insights'][:2])})"
                if 'column_meanings' in hints:
                    catalog_insights += f"\n- Column definitions available for better interpretation"
        
        prompt = f"""You are an expert data analyst synthesizing insights from a data exploration operation.

CONTEXT:
- User Question: "{user_question}"
- Analysis Intent: {self.primary_intent['intent']}
- Current Step: {step}
- Iteration: {iteration}
- Previous Insights: {previous_insights[:3] if previous_insights else 'None'}
{relationship_insights}
{catalog_insights}

OPERATION RESULT:
{result_data['summary']}

Raw Data Sample:
{self._safe_json_serialize(result_data.get('sample_data', result_data.get('raw_data', 'No data'))[:5])}

TASK:
Analyze this result considering the discovered relationships and catalog insights to provide:
1. Specific findings that directly answer the user's question
2. Leverage relationship context to explain connections
3. Use catalog quality insights to assess reliability
4. Include exact numbers and patterns from the data
5. Consider the business context from catalog descriptions

RESPONSE FORMAT (JSON):
{{
    "findings": [
        "Specific finding 1 with exact numbers from the data",
        "Specific finding 2 explaining relationships discovered",
        "Specific finding 3 with business implications"
    ],
    "confidence": 0.85,
    "business_relevance": "How this answers the question using relationship context",
    "next_steps": [
        "Specific next step based on relationships found",
        "Investigation guided by catalog recommendations"
    ],
    "data_quality_notes": [
        "Quality observations from catalog insights"
    ]
}}

Focus on specific, data-driven insights that leverage the semantic understanding of the data."""
        
        return prompt
    
    def _safe_json_serialize(self, data: Any) -> str:
        """Safely serialize data to JSON, handling datetime objects"""
        try:
            import pandas as pd
            
            def convert_item(item):
                if isinstance(item, dict):
                    return {k: convert_item(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [convert_item(i) for i in item]
                elif pd.isna(item):
                    return None
                elif hasattr(item, 'isoformat'):  # datetime objects
                    return item.isoformat()
                elif hasattr(item, 'item'):  # numpy types
                    return item.item()
                else:
                    return str(item) if not isinstance(item, (int, float, bool, str, type(None))) else item
            
            converted_data = convert_item(data)
            return json.dumps(converted_data, indent=2)
        except Exception as e:
            return f"Data serialization error: {str(e)}"
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured insights"""
        
        try:
            # Extract JSON from response
            if '```json' in response_content:
                json_str = response_content.split('```json')[1].split('```')[0]
            elif '{' in response_content and '}' in response_content:
                start = response_content.find('{')
                end = response_content.rfind('}') + 1
                json_str = response_content[start:end]
            else:
                raise ValueError("No JSON found in response")
            
            insights = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['findings', 'confidence', 'business_relevance']
            for field in required_fields:
                if field not in insights:
                    insights[field] = []
            
            return insights
            
        except Exception as e:
            # Fallback parsing
            return self._fallback_parsing(response_content)
    
    def _fallback_parsing(self, response_content: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        
        # Extract key phrases
        findings = []
        if 'finding' in response_content.lower():
            lines = response_content.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['found', 'discovered', 'identified', 'shows']):
                    findings.append(line.strip())
        
        return {
            'findings': findings[:3] if findings else ['Analysis completed successfully'],
            'confidence': 0.7,
            'business_relevance': 'Analysis provides insights relevant to the question',
            'next_steps': ['Continue with deeper analysis'],
            'data_quality_notes': []
        }
    
    def _fallback_synthesis(self, result_data: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback synthesis when LLM fails - extract concrete data"""
        
        step = context.get('step', '')
        intent = self.primary_intent['intent']
        
        # Try to extract concrete findings first
        concrete_findings = self._extract_concrete_findings(result_data, context)
        
        if concrete_findings:
            return {
                'findings': concrete_findings,
                'confidence': 0.7,
                'business_relevance': self._determine_relevance(concrete_findings, context),
                'next_steps': self._suggest_next_steps(step),
                'data_quality_notes': []
            }
        
        # If no concrete findings, provide data-aware generic findings
        findings = []
        
        if result_data['type'] == 'dataframe':
            shape = result_data.get('shape', [0, 0])
            findings.append(f"Analyzed DataFrame with {shape[0]} rows and {shape[1]} columns")
            if result_data.get('raw_data'):
                findings.append(f"Data sample contains {len(result_data['raw_data'])} records")
        elif result_data['type'] == 'series':
            length = result_data.get('length', 0)
            findings.append(f"Analyzed Series with {length} values")
        elif result_data['type'] == 'collection':
            data = result_data.get('raw_data', [])
            findings.append(f"Found collection with {len(data)} items")
        else:
            findings.append(f"Operation completed: {result_data['summary']}")
        
        return {
            'findings': findings,
            'confidence': 0.5,
            'business_relevance': f'Analysis step completed for {intent} investigation',
            'next_steps': ['Continue with deeper analysis'],
            'data_quality_notes': []
        }
    
    def _synthesize_error_insight(self, operation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insight when operation fails"""
        
        error_msg = operation_result.get('error', {}).get('message', 'Unknown error')
        
        return {
            'findings': [f"Operation failed: {error_msg}"],
            'confidence': 0.1,
            'business_relevance': 'Unable to provide insights due to operation failure',
            'next_steps': ['Fix the operation and retry'],
            'data_quality_notes': [f'Error: {error_msg}'],
            'context_metadata': {
                'error': True,
                'error_message': error_msg
            }
        }


class EnhancedDataExplorationReActAgent(DataExplorationReActAgent):
    """Enhanced agent with comprehensive intelligence integration"""
    
    def __init__(self, 
                 enhanced_intelligence: Any,
                 semantic_graph_builder: Any,
                 enhanced_summarizer: Optional[Any] = None,
                 intelligent_catalog: Optional[Any] = None,
                 llm_model: str = "gpt-4"):
        
        super().__init__(enhanced_intelligence, semantic_graph_builder, llm_model)
        
        self.enhanced_summarizer = enhanced_summarizer
        self.intelligent_catalog = intelligent_catalog
        self.question_analyzer = QuestionAnalyzer()
        self.operation_generator = None
        self.insight_synthesizer = None
        self._current_exploration_state = None
        self.catalog_entries = {}
    
    def explore_for_insights(self, user_question: str, 
                           tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced exploration with comprehensive intelligence"""
        
        print("🧠 Initializing comprehensive intelligence analysis...")
        
        # Step 1: Generate enhanced profiles with intelligent catalog
        enhanced_profiles = {}
        for name, df in tables.items():
            print(f"📊 Profiling {name}...")
            try:
                profile = self.enhanced_intelligence.analyze_table_comprehensive(name, df)
                enhanced_profiles[name] = profile
                
                # Generate catalog entry if catalog is available
                if self.intelligent_catalog:
                    try:
                        catalog_entry = self.intelligent_catalog.generate_catalog_entry(profile)
                        self.catalog_entries[name] = catalog_entry
                        print(f"📚 Generated catalog: {catalog_entry.title}")
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to generate catalog for {name}: {e}")
                        
            except Exception as e:
                print(f"⚠️ Warning: Failed to profile {name}: {e}")
                enhanced_profiles[name] = self._basic_profile(name, df)
        
        # Step 2: Analyze the question
        print("🎯 Analyzing question and creating strategy...")
        question_analysis = self.question_analyzer.analyze_question(user_question, enhanced_profiles)
        
        # Step 3: Build semantic understanding
        print("🕸️ Building semantic relationships...")
        try:
            # First add the dataset to the graph builder
            self.semantic_graph_builder.add_dataset(tables, "analysis_dataset")
            
            # Get relationships summary
            relationships = self.semantic_graph_builder.get_relationship_summary()
            
            # Store the graph
            table_graph = self.semantic_graph_builder.graph
        except Exception as e:
            print(f"⚠️ Warning: Failed to build relationships: {e}")
            relationships = {}
            table_graph = None
        
        # Step 4: Initialize intelligent components with catalog
        self.operation_generator = IntelligentOperationGenerator(
            enhanced_profiles, 
            question_analysis,
            self.catalog_entries
        )
        self.insight_synthesizer = IntelligentInsightSynthesizer(self.llm, question_analysis)
        
        # Step 5: Initialize enhanced exploration state
        self._current_exploration_state = IntelligentExplorationState(
            user_question=user_question,
            tables=tables,
            enhanced_profiles=enhanced_profiles,
            table_relationships=relationships,
            semantic_graph=table_graph if 'table_graph' in locals() else None,
            question_analysis=question_analysis,
            analysis_strategy=question_analysis['analysis_strategy']
        )
        
        # Step 6: Execute intelligent exploration
        return self._execute_intelligent_exploration()
    
    def _basic_profile(self, name: str, df: pd.DataFrame) -> Any:
        """Create basic profile if enhanced profiling fails"""
        from dataclasses import dataclass
        
        @dataclass
        class MinimalTableProfile:
            table_name: str
            row_count: int
            column_count: int
            measure_columns: List[str]
            dimension_columns: List[str]
            identifier_columns: List[str]
            temporal_columns: List[str]
            data_quality_score: float
        
        # Detect column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Add columns with date in name
        for col in df.columns:
            if 'date' in col.lower() and col not in datetime_cols:
                datetime_cols.append(col)
        
        return MinimalTableProfile(
            table_name=name,
            row_count=len(df),
            column_count=len(df.columns),
            measure_columns=numeric_cols,
            dimension_columns=object_cols,
            identifier_columns=[col for col in df.columns if 'id' in col.lower()],
            temporal_columns=datetime_cols,
            data_quality_score=1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        )
    
    def _execute_intelligent_exploration(self) -> Dict[str, Any]:
        """Execute exploration with comprehensive intelligence guidance"""
        
        state = self._current_exploration_state
        strategy = state.analysis_strategy
        
        print(f"🔍 Starting intelligent exploration: '{state.user_question}'")
        print(f"📋 Strategy: {strategy.get('approach', 'discovery')}")
        print(f"🎯 Intent: {state.question_analysis['primary_intent']['intent']}")
        print("=" * 80)
        
        # Check for limitations
        if strategy.get('limitations'):
            print(f"⚠️ Limitations detected: {strategy['limitations']}")
            return self._handle_limitations(strategy)
        
        # Iterative exploration loop
        while (state.iteration_count < state.max_iterations and 
               state.confidence_level < 0.85):
            
            state.iteration_count += 1
            print(f"\n🔄 Intelligence Cycle {state.iteration_count}")
            
            # Execute intelligent cycle
            cycle_result = self._execute_intelligent_cycle()
            
            # Update state - ensure we accumulate actual findings as strings
            if cycle_result.get('insights'):
                insights = cycle_result['insights']
                # Store findings as individual strings
                if isinstance(insights.get('findings'), list):
                    for finding in insights['findings']:
                        if isinstance(finding, str):
                            state.accumulated_insights.append(finding)
                # Also store the entire insights dict for context
                state.accumulated_insights.append(insights)
            
            # Check if we have sufficient insights
            if cycle_result.get('sufficient_insights', False):
                print("✅ Sufficient insights achieved")
                break
        
        # Generate final comprehensive insights
        final_insights = self._generate_final_comprehensive_insights()
        
        # Add catalog insights if available
        catalog_insights = {}
        if self.catalog_entries:
            for name, catalog in self.catalog_entries.items():
                catalog_insights[name] = {
                    'title': catalog.title,
                    'quality_badge': catalog.quality_badge,
                    'recommended_queries': catalog.recommended_queries[:2] if catalog.recommended_queries else []
                }
        
        return {
            'user_question': state.user_question,
            'exploration_summary': {
                'iterations_used': state.iteration_count,
                'confidence_level': state.confidence_level,
                'total_findings': len(state.accumulated_insights),
                'strategy_used': strategy.get('approach', 'discovery'),
                'intelligence_driven': True,
                'operations_executed': len(state.previous_operations)
            },
            'insights': final_insights,
            'intelligence_context': {
                'profiles_generated': len(state.enhanced_profiles),
                'analysis_plans': {name: 1 for name in state.enhanced_profiles.keys()},
                'question_analysis': state.question_analysis,
                'strategy_applied': strategy,
                'catalog_insights': catalog_insights
            },
            'exploration_history': state.exploration_history,
            'recommendations': self._generate_intelligent_recommendations(),
            'data_quality_summary': self._summarize_data_quality()
        }
    
    def _execute_intelligent_cycle(self) -> Dict[str, Any]:
        """Execute one intelligent exploration cycle"""
        
        state = self._current_exploration_state
        
        # Generate next operation based on strategy with full context
        operation_info = self.operation_generator.generate_next_operation(
            state.iteration_count,
            {
                'previous_operations': state.previous_operations,
                'user_question': state.user_question,
                'iteration': state.iteration_count,
                'table_relationships': state.table_relationships,
                'semantic_graph': state.semantic_graph
            }
        )
        
        print(f"🎯 Table: {operation_info['target_tables'][0] if operation_info['target_tables'] else 'unknown'}")
        print(f"🔧 Operation: {operation_info['step']}")
        
        # Display relationship context if available
        if operation_info.get('relationship_context') and operation_info['relationship_context'].get('key_relationships'):
            rel_count = len(operation_info['relationship_context']['key_relationships'])
            print(f"🔗 Using {rel_count} semantic relationships")
        
        # Display catalog hints if available 
        if operation_info.get('catalog_hints'):
            hint_tables = list(operation_info['catalog_hints'].keys())
            print(f"📚 Leveraging catalog insights for: {', '.join(hint_tables)}")
        
        # Execute operation
        result = self.toolkit.execute_pandas_operation(operation_info['operation'], state.tables)
        
        # Record operation
        operation_record = {
            'iteration': state.iteration_count,
            'step': operation_info['step'],
            'operation': operation_info['operation'],
            'target_tables': operation_info['target_tables'],
            'success': result['success']
        }
        
        if result['success']:
            print(f"✅ {operation_info['expected_insight']}")
        else:
            print(f"❌ Error: {result['error']['message']}")
        
        state.previous_operations.append(operation_record)
        
        # Synthesize insights with full context including relationships and catalog
        insights = self.insight_synthesizer.synthesize_insights(
            result,
            {
                'user_question': state.user_question,
                'step': operation_info['step'],
                'iteration': state.iteration_count,
                'previous_insights': [insight for insight in state.accumulated_insights[-3:]],
                'relationship_context': operation_info.get('relationship_context', {}),
                'catalog_hints': operation_info.get('catalog_hints', {})
            }
        )
        
        # Update confidence
        state.confidence_level = insights.get('confidence', 0.5)
        
        # Record in history
        state.exploration_history.append({
            'iteration': state.iteration_count,
            'step': operation_info['step'],
            'operation': operation_info['operation'],
            'insights': insights,
            'confidence': state.confidence_level
        })
        
        return {
            'insights': insights,
            'sufficient_insights': len(state.accumulated_insights) >= 3 or state.confidence_level >= 0.8
        }
    
    def _generate_final_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate final comprehensive insights"""
        
        state = self._current_exploration_state
        
        # Compile all concrete findings (strings only)
        all_findings = []
        for item in state.accumulated_insights:
            if isinstance(item, str):
                all_findings.append(item)
            elif isinstance(item, dict) and 'findings' in item:
                # Skip - already added as individual strings
                pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_findings = []
        for finding in all_findings:
            if finding not in seen:
                seen.add(finding)
                unique_findings.append(finding)
        
        # Generate intent-specific answer
        intent = state.question_analysis['primary_intent']['intent']
        
        # Create direct answer based on findings
        if intent == 'quality':
            missing_findings = [f for f in unique_findings if 'missing' in f.lower() or 'null' in f.lower()]
            if missing_findings:
                direct_answer = f"Data quality issues found: {'; '.join(missing_findings[:2])}"
            else:
                direct_answer = "Data quality is excellent with no missing values detected across all columns"
        
        elif intent == 'correlation':
            corr_findings = [f for f in unique_findings if 'correlation' in f.lower() or '↔' in f]
            if corr_findings:
                direct_answer = f"Found relationships: {'; '.join(corr_findings[:2])}"
            else:
                direct_answer = "No strong correlations found between the analyzed variables"
        
        elif intent == 'outlier':
            outlier_findings = [f for f in unique_findings if 'outlier' in f.lower() or 'bound' in f.lower()]
            if outlier_findings:
                direct_answer = f"Outlier analysis: {'; '.join(outlier_findings[:2])}"
            else:
                direct_answer = "No significant outliers detected in the analyzed data"
        
        else:
            direct_answer = f"Analysis complete: {'; '.join(unique_findings[:2])}" if unique_findings else "Analysis completed"
        
        # If we have enough concrete findings, use them directly
        if len(unique_findings) >= 3:
            return {
                "direct_answer": direct_answer,
                "key_insights": unique_findings[:5],
                "supporting_evidence": unique_findings[5:8] if len(unique_findings) > 5 else [],
                "confidence_score": 0.9 if len(unique_findings) >= 5 else 0.7,
                "business_implications": self._derive_business_implications(intent, unique_findings)
            }
        
        # Otherwise try LLM synthesis with the findings we have
        prompt = f"""Based on this data exploration, provide a final answer to the user's question.

USER QUESTION: "{state.user_question}"
ANALYSIS INTENT: {intent}

CONCRETE FINDINGS:
{chr(10).join(f'- {finding}' for finding in unique_findings)}

Provide a JSON response that directly answers the question using the findings above.
Include the actual numbers and specific data points from the findings.

{{
    "direct_answer": "Specific answer using the findings",
    "key_insights": ["Use the findings above"],
    "supporting_evidence": ["Additional details from findings"],
    "confidence_score": 0.8,
    "business_implications": ["Based on the findings"]
}}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                content = content[start:end]
            
            insights = json.loads(content)
            
        except Exception as e:
            # Fallback with actual findings
            insights = {
                "direct_answer": direct_answer,
                "key_insights": unique_findings if unique_findings else ["Analysis completed but no specific findings extracted"],
                "supporting_evidence": [f"Executed {state.iteration_count} analysis iterations"],
                "confidence_score": 0.6 if unique_findings else 0.3,
                "business_implications": ["Further analysis recommended for actionable insights"]
            }
        
        return insights
    
    def _derive_business_implications(self, intent: str, findings: List[str]) -> List[str]:
        """Derive business implications from concrete findings"""
        
        implications = []
        
        if intent == 'quality' and any('missing' in f for f in findings):
            implications.append("Data completeness issues may impact analysis accuracy")
            implications.append("Implement data validation at collection point")
        elif intent == 'correlation' and any('correlation' in f for f in findings):
            implications.append("Use identified relationships for predictive modeling")
            implications.append("Optimize correlated variables together")
        elif intent == 'outlier' and any('outlier' in f for f in findings):
            implications.append("Investigate outliers for data quality or special cases")
            implications.append("Consider separate handling for outlier segments")
        else:
            implications.append("Use findings to guide business decisions")
            implications.append("Monitor identified patterns over time")
        
        return implications[:2]
    
    def _handle_limitations(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cases where analysis has limitations"""
        
        limitations = strategy.get('limitations', [])
        
        return {
            'user_question': self._current_exploration_state.user_question,
            'exploration_summary': {
                'iterations_used': 0,
                'confidence_level': 0.9,
                'total_findings': 1,
                'strategy_used': 'limitation_detection',
                'intelligence_driven': True,
                'operations_executed': 0
            },
            'insights': {
                'direct_answer': f"Analysis not possible due to data limitations: {', '.join(limitations)}",
                'key_insights': [
                    f"Data limitation detected: {limitations[0] if limitations else 'Insufficient data'}",
                    "Alternative analysis approaches are recommended",
                    "Consider collecting additional data for comprehensive analysis"
                ],
                'supporting_evidence': ["Intelligence analysis detected data gaps"],
                'confidence_score': 0.9,
                'business_implications': ["Data collection strategy needed for future analysis"]
            },
            'intelligence_context': {
                'profiles_generated': len(self._current_exploration_state.enhanced_profiles),
                'limitation_detected': True,
                'limitations': limitations
            },
            'exploration_history': [],
            'recommendations': [
                "Collect additional data to meet analysis requirements",
                "Consider alternative analysis approaches",
                "Focus on available data patterns in the meantime"
            ],
            'data_quality_summary': {'limitations_detected': True, 'gaps': limitations}
        }
    
    def _generate_intelligent_recommendations(self) -> List[str]:
        """Generate intelligent recommendations"""
        
        state = self._current_exploration_state
        recommendations = []
        
        # Strategy-specific recommendations
        if state.confidence_level < 0.7:
            recommendations.append("Consider collecting additional data to increase analysis confidence")
        
        if len(state.quality_concerns) > 0:
            recommendations.append("Address data quality issues before production use")
        
        if state.iteration_count >= state.max_iterations:
            recommendations.append("Consider focused analysis on specific aspects for deeper insights")
        
        # Add general recommendations
        recommendations.extend([
            "Validate findings with domain experts",
            "Monitor data patterns over time for trend confirmation",
            "Consider implementing automated monitoring for key metrics"
        ])
        
        return recommendations[:5]
    
    def _summarize_data_quality(self) -> Dict[str, Any]:
        """Summarize data quality findings"""
        
        state = self._current_exploration_state
        
        # Calculate overall quality score
        quality_scores = []
        for profile in state.enhanced_profiles.values():
            if hasattr(profile, 'data_quality_score'):
                quality_scores.append(profile.data_quality_score)
        
        overall_score = np.mean(quality_scores) if quality_scores else 0.8
        
        return {
            'overall_score': overall_score,
            'critical_issues': len(state.quality_concerns),
            'tables_analyzed': len(state.enhanced_profiles),
            'quality_concerns': state.quality_concerns[:3]
        }