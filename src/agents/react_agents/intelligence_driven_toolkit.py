"""
Intelligence-Driven Toolkit for Data Exploration

This toolkit leverages the full power of the knowledge graph infrastructure including:
- EnhancedTableIntelligenceLayer with ML-powered analysis
- Advanced data quality analysis with alerts
- Correlation analysis with multiple methods
- Outlier detection with impact assessment
- Key insights generation
- Intelligent data catalog
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from knowledge_graph.table_intelligence import (
    EnhancedTableIntelligenceLayer, 
    EnhancedTableProfile, 
    EnhancedDataQualityProfile
)


class IntelligenceDrivenToolkit:
    """Toolkit that leverages table intelligence for smart operations"""
    
    def __init__(self, enhanced_intelligence: EnhancedTableIntelligenceLayer):
        self.enhanced_intelligence = enhanced_intelligence
        self.profiles: Dict[str, EnhancedTableProfile] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize intelligence analyzers if available
        self.quality_analyzer = getattr(enhanced_intelligence, 'quality_analyzer', None)
        self.correlation_analyzer = getattr(enhanced_intelligence, 'correlation_analyzer', None)
        self.outlier_analyzer = getattr(enhanced_intelligence, 'outlier_analyzer', None)
    
    def register_table_profile(self, table_name: str, profile: EnhancedTableProfile):
        """Register a table profile for intelligence-driven operations"""
        self.profiles[table_name] = profile
        self.logger.info(f"Registered profile for {table_name} with {len(profile.key_insights or [])} insights")
    
    def get_intelligence_summary(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive intelligence summary for a table"""
        profile = self.profiles.get(table_name)
        if not profile:
            return {"error": f"No profile found for {table_name}"}
        
        summary = {
            "table_name": table_name,
            "data_quality_score": profile.data_quality_score,
            "ml_readiness_score": profile.ml_readiness_score,
            "key_insights": profile.key_insights or [],
            "measure_columns": profile.measure_columns,
            "dimension_columns": profile.dimension_columns,
            "temporal_columns": profile.temporal_columns,
            "identifier_columns": profile.identifier_columns
        }
        
        # Add quality intelligence
        if profile.quality_profile:
            summary["quality_insights"] = {
                "overall_score": profile.quality_profile.overall_quality_score,
                "critical_alerts": len(profile.quality_profile.critical_alerts),
                "warning_alerts": len(profile.quality_profile.warning_alerts),
                "recommendations": profile.quality_profile.quality_recommendations
            }
        
        # Add correlation intelligence
        if profile.correlation_analysis:
            summary["correlation_insights"] = {
                "strong_correlations": len(profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', [])),
                "feature_redundancy": len(profile.correlation_analysis.get('feature_redundancy', [])),
                "has_correlations": bool(profile.correlation_analysis)
            }
        
        # Add outlier intelligence
        if profile.outlier_analysis:
            summary["outlier_insights"] = {
                "high_impact_outliers": len(profile.outlier_analysis.get('high_impact_outliers', [])),
                "outlier_recommendations": profile.outlier_analysis.get('outlier_recommendations', []),
                "has_outliers": bool(profile.outlier_analysis.get('high_impact_outliers', []))
            }
        
        return summary
    
    # TOOL 1: Quality-Aware Operations
    def get_quality_aware_operation(self, table_name: str, operation_type: str = "investigate") -> str:
        """Generate operations that account for data quality issues"""
        profile = self.profiles.get(table_name)
        if not profile or not profile.quality_profile:
            return f"tables['{table_name}'].describe()"
        
        quality_profile = profile.quality_profile
        
        # Handle critical alerts first
        if quality_profile.critical_alerts:
            alert = quality_profile.critical_alerts[0]
            alert_type = alert.get('alert_type', '') if isinstance(alert, dict) else str(alert)
            
            if 'missing' in alert_type.lower():
                # Find column with missing values
                missing_cols = [col for col in profile.measure_columns + profile.dimension_columns 
                               if col in f"tables['{table_name}']"]
                if missing_cols:
                    return f"tables['{table_name}'][{missing_cols[:3]}].isnull().sum().sort_values(ascending=False)"
                return f"tables['{table_name}'].isnull().sum().sort_values(ascending=False)"
            
            elif 'constant' in alert_type.lower():
                return f"tables['{table_name}'].nunique().sort_values()"
            
            elif 'correlation' in alert_type.lower():
                if profile.measure_columns:
                    return f"tables['{table_name}'][{profile.measure_columns[:5]}].corr()"
        
        # Handle warning alerts
        elif quality_profile.warning_alerts:
            if operation_type == "missing_analysis":
                return f"tables['{table_name}'].isnull().sum().sort_values(ascending=False)"
            elif operation_type == "distribution_analysis":
                if profile.measure_columns:
                    return f"tables['{table_name}'][{profile.measure_columns[:3]}].describe()"
        
        # Default quality check
        return f"pd.DataFrame({{'column': tables['{table_name}'].columns, 'null_count': tables['{table_name}'].isnull().sum(), 'null_pct': tables['{table_name}'].isnull().sum() / len(tables['{table_name}']) * 100}}).sort_values('null_pct', ascending=False)"
    
    # TOOL 2: Correlation-Driven Operations
    def get_correlation_operation(self, table_name: str, focus: str = "strong") -> str:
        """Generate operations based on correlation insights"""
        profile = self.profiles.get(table_name)
        if not profile:
            return f"tables['{table_name}'].corr()"
        
        # Use correlation analysis if available
        if profile.correlation_analysis:
            linear_rels = profile.correlation_analysis.get('linear_relationships', {})
            strong_correlations = linear_rels.get('strong_linear', [])
            
            if strong_correlations and focus == "strong":
                # Get variables from first strong correlation
                corr = strong_correlations[0]
                if isinstance(corr, dict) and 'variables' in corr:
                    vars_tuple = corr['variables']
                    if isinstance(vars_tuple, (list, tuple)) and len(vars_tuple) >= 2:
                        return f"tables['{table_name}'][{list(vars_tuple)}].corr()"
            
            # Look for feature redundancy
            redundancy = profile.correlation_analysis.get('feature_redundancy', [])
            if redundancy and focus == "redundancy":
                redundant_features = redundancy[0].get('redundant_features', [])
                if redundant_features:
                    return f"tables['{table_name}'][{redundant_features[:4]}].corr()"
        
        # Fallback to measure columns correlation
        if profile.measure_columns and len(profile.measure_columns) > 1:
            return f"tables['{table_name}'][{profile.measure_columns[:5]}].corr()"
        
        # Final fallback
        return f"tables['{table_name}'].select_dtypes(include=[np.number]).corr()"
    
    # TOOL 3: Outlier-Aware Operations
    def get_outlier_operation(self, table_name: str, focus: str = "investigate") -> str:
        """Generate operations to investigate outliers"""
        profile = self.profiles.get(table_name)
        if not profile:
            return f"tables['{table_name}'].describe()"
        
        # Use outlier analysis if available
        if profile.outlier_analysis:
            high_impact = profile.outlier_analysis.get('high_impact_outliers', [])
            
            if high_impact:
                outlier_info = high_impact[0]
                outlier_col = outlier_info.get('column')
                
                if outlier_col and focus == "investigate":
                    return f"pd.DataFrame({{'statistic': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'outliers_pct'], 'value': [tables['{table_name}']['{outlier_col}'].count(), tables['{table_name}']['{outlier_col}'].mean(), tables['{table_name}']['{outlier_col}'].std(), tables['{table_name}']['{outlier_col}'].min(), tables['{table_name}']['{outlier_col}'].quantile(0.25), tables['{table_name}']['{outlier_col}'].median(), tables['{table_name}']['{outlier_col}'].quantile(0.75), tables['{table_name}']['{outlier_col}'].max(), {outlier_info.get('percentage', 0):.1f}]}})"
                
                elif outlier_col and focus == "bounds":
                    # Calculate outlier bounds
                    return f"tables['{table_name}']['{outlier_col}'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])"
        
        # Fallback to general outlier detection on measure columns
        if profile.measure_columns:
            col = profile.measure_columns[0]
            return f"tables['{table_name}']['{col}'].describe()"
        
        return f"tables['{table_name}'].describe()"
    
    # TOOL 4: Temporal-Aware Operations
    def get_temporal_operation(self, table_name: str, analysis_type: str = "range") -> str:
        """Generate temporal analysis operations"""
        profile = self.profiles.get(table_name)
        if not profile or not profile.temporal_columns:
            return f"tables['{table_name}'].dtypes"
        
        temporal_col = profile.temporal_columns[0]
        
        if analysis_type == "range":
            return f"pd.DataFrame({{'metric': ['min_date', 'max_date', 'date_range_days', 'null_count'], 'value': [tables['{table_name}']['{temporal_col}'].min(), tables['{table_name}']['{temporal_col}'].max(), (tables['{table_name}']['{temporal_col}'].max() - tables['{table_name}']['{temporal_col}'].min()).days, tables['{table_name}']['{temporal_col}'].isnull().sum()]}})"
        
        elif analysis_type == "patterns" and profile.measure_columns:
            measure_col = profile.measure_columns[0]
            return f"tables['{table_name}'].groupby(tables['{table_name}']['{temporal_col}'].dt.date)['{measure_col}'].agg(['count', 'sum', 'mean'])"
        
        elif analysis_type == "frequency":
            return f"tables['{table_name}']['{temporal_col}'].dt.dayofweek.value_counts().sort_index()"
        
        return f"tables['{table_name}']['{temporal_col}'].describe()"
    
    # TOOL 5: Segmentation Operations
    def get_segmentation_operation(self, table_name: str, analysis_type: str = "overview") -> str:
        """Generate segmentation analysis operations"""
        profile = self.profiles.get(table_name)
        if not profile:
            return f"tables['{table_name}'].info()"
        
        if not profile.dimension_columns:
            return f"tables['{table_name}'].select_dtypes(include=['object']).describe()"
        
        dim_col = profile.dimension_columns[0]
        
        if analysis_type == "overview":
            return f"tables['{table_name}']['{dim_col}'].value_counts()"
        
        elif analysis_type == "metrics" and profile.measure_columns:
            measure_col = profile.measure_columns[0]
            return f"tables['{table_name}'].groupby('{dim_col}')['{measure_col}'].agg(['count', 'mean', 'sum', 'std']).round(2)"
        
        elif analysis_type == "distribution":
            return f"pd.DataFrame({{'segment': tables['{table_name}']['{dim_col}'].value_counts().index, 'count': tables['{table_name}']['{dim_col}'].value_counts().values, 'percentage': tables['{table_name}']['{dim_col}'].value_counts(normalize=True).values * 100}})"
        
        return f"tables['{table_name}']['{dim_col}'].value_counts()"
    
    # TOOL 6: Intelligence Methods as Tools
    def analyze_correlations_on_demand(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run correlation analysis on demand"""
        if self.correlation_analyzer:
            return self.correlation_analyzer.analyze_advanced_correlations(df, table_name)
        else:
            # Fallback correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_correlations.append({
                                'variables': (corr_matrix.columns[i], corr_matrix.columns[j]),
                                'correlation': corr_val
                            })
                return {'strong_correlations': strong_correlations}
            return {}
    
    def detect_outliers_on_demand(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run outlier detection on demand"""
        if self.outlier_analyzer:
            column_insights = self.enhanced_intelligence._analyze_columns(df)
            return self.outlier_analyzer.detect_outliers_comprehensive(df, column_insights)
        else:
            # Fallback outlier detection
            outliers = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    q1, q3 = data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
                    outliers[col] = {
                        'outlier_count': outlier_count,
                        'outlier_percentage': (outlier_count / len(data)) * 100
                    }
            return {'outlier_summary': outliers}
    
    def assess_quality_on_demand(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run quality assessment on demand"""
        if self.quality_analyzer:
            return self.quality_analyzer.analyze_data_quality(df, table_name)
        else:
            # Fallback quality assessment
            null_counts = df.isnull().sum()
            quality_score = (1 - null_counts.sum() / (len(df) * len(df.columns))) * 100
            return {
                'overall_quality_score': quality_score,
                'null_counts': null_counts.to_dict(),
                'recommendations': ['Handle missing values'] if null_counts.sum() > 0 else []
            }
    
    # TOOL 7: Smart Operation Generator
    def generate_intelligent_operation(self, table_name: str, question_intent: str, iteration: int = 1) -> str:
        """Generate operations based on intelligence and question intent"""
        profile = self.profiles.get(table_name)
        if not profile:
            return f"tables['{table_name}'].info()"
        
        # Use key insights to guide operation generation
        if profile.key_insights:
            for insight in profile.key_insights:
                insight_lower = insight.lower()
                
                # Outlier-focused operations
                if "outlier" in insight_lower and any(intent in question_intent.lower() for intent in ['pattern', 'anomal', 'unusual']):
                    return self.get_outlier_operation(table_name)
                
                # Correlation-focused operations
                elif "correlation" in insight_lower and any(intent in question_intent.lower() for intent in ['relationship', 'correlation', 'factor']):
                    return self.get_correlation_operation(table_name)
                
                # Quality-focused operations
                elif "quality" in insight_lower or "missing" in insight_lower:
                    if any(intent in question_intent.lower() for intent in ['quality', 'missing', 'complete']):
                        return self.get_quality_aware_operation(table_name)
                
                # Temporal-focused operations
                elif "temporal" in insight_lower or "time" in insight_lower:
                    if any(intent in question_intent.lower() for intent in ['time', 'temporal', 'seasonal', 'trend']):
                        return self.get_temporal_operation(table_name, "patterns")
        
        # Use question intent analysis
        intent_lower = question_intent.lower()
        
        # Temporal analysis
        if any(keyword in intent_lower for keyword in ['seasonal', 'trend', 'time', 'temporal', 'monthly', 'daily']):
            if profile.temporal_columns:
                return self.get_temporal_operation(table_name, "range" if iteration == 1 else "patterns")
            else:
                available_cols = profile.measure_columns + profile.dimension_columns
                return f"print(f'No temporal columns in {table_name}. Available: {available_cols}')"
        
        # Segmentation analysis
        elif any(keyword in intent_lower for keyword in ['segment', 'group', 'category', 'which']):
            return self.get_segmentation_operation(table_name, "overview" if iteration == 1 else "metrics")
        
        # Correlation analysis
        elif any(keyword in intent_lower for keyword in ['correlation', 'relationship', 'factor', 'influence']):
            return self.get_correlation_operation(table_name)
        
        # Outlier/anomaly analysis
        elif any(keyword in intent_lower for keyword in ['outlier', 'anomaly', 'unusual', 'abnormal']):
            return self.get_outlier_operation(table_name)
        
        # Quality analysis
        elif any(keyword in intent_lower for keyword in ['quality', 'missing', 'complete', 'clean']):
            return self.get_quality_aware_operation(table_name)
        
        # Use ML readiness to suggest operations
        if profile.ml_readiness_score and profile.ml_readiness_score > 80:
            if iteration == 1:
                return f"tables['{table_name}'].select_dtypes(include=[np.number]).describe()"
            else:
                return self.get_correlation_operation(table_name)
        
        # Data quality score guidance
        if profile.data_quality_score < 0.8:
            return self.get_quality_aware_operation(table_name)
        
        # Iteration-based progression
        if iteration == 1:
            return f"tables['{table_name}'].shape, tables['{table_name}'].dtypes.value_counts(), tables['{table_name}'].isnull().sum().sum()"
        elif iteration == 2:
            if profile.measure_columns:
                return f"tables['{table_name}'][{profile.measure_columns[:3]}].describe()"
            else:
                return f"tables['{table_name}'].describe(include='all')"
        elif iteration == 3:
            if profile.dimension_columns:
                return self.get_segmentation_operation(table_name)
            else:
                return self.get_correlation_operation(table_name)
        else:
            # Advanced analysis for later iterations
            return self.get_correlation_operation(table_name)
    
    # TOOL 8: Insight-Based Operation Suggestions
    def get_insight_based_suggestions(self, table_name: str, user_question: str) -> List[str]:
        """Get operation suggestions based on intelligence insights"""
        profile = self.profiles.get(table_name)
        if not profile:
            return [f"tables['{table_name}'].info()"]
        
        suggestions = []
        
        # Quality-based suggestions
        if profile.quality_profile and profile.quality_profile.critical_alerts:
            suggestions.append(self.get_quality_aware_operation(table_name))
        
        # Outlier-based suggestions
        if (profile.outlier_analysis and 
            profile.outlier_analysis.get('high_impact_outliers')):
            suggestions.append(self.get_outlier_operation(table_name))
        
        # Correlation-based suggestions
        if (profile.correlation_analysis and 
            profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear')):
            suggestions.append(self.get_correlation_operation(table_name))
        
        # Temporal suggestions
        if profile.temporal_columns:
            suggestions.append(self.get_temporal_operation(table_name))
        
        # Segmentation suggestions
        if profile.dimension_columns:
            suggestions.append(self.get_segmentation_operation(table_name))
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def get_comprehensive_analysis_plan(self, table_name: str, user_question: str) -> Dict[str, Any]:
        """Create a comprehensive analysis plan based on intelligence"""
        profile = self.profiles.get(table_name)
        if not profile:
            return {"error": f"No profile found for {table_name}"}
        
        plan = {
            "table_name": table_name,
            "analysis_phases": [],
            "intelligence_context": self.get_intelligence_summary(table_name),
            "recommendations": []
        }
        
        # Phase 1: Data Overview
        plan["analysis_phases"].append({
            "phase": "data_overview",
            "description": "Basic data structure and quality assessment",
            "operations": [
                f"tables['{table_name}'].shape",
                f"tables['{table_name}'].info()",
                self.get_quality_aware_operation(table_name)
            ]
        })
        
        # Phase 2: Domain-specific analysis based on question
        if any(keyword in user_question.lower() for keyword in ['correlation', 'relationship']):
            plan["analysis_phases"].append({
                "phase": "relationship_analysis",
                "description": "Correlation and relationship investigation",
                "operations": [self.get_correlation_operation(table_name)]
            })
        
        if any(keyword in user_question.lower() for keyword in ['segment', 'group']):
            plan["analysis_phases"].append({
                "phase": "segmentation_analysis",
                "description": "Segment-based analysis",
                "operations": [
                    self.get_segmentation_operation(table_name, "overview"),
                    self.get_segmentation_operation(table_name, "metrics")
                ]
            })
        
        # Phase 3: Advanced analysis based on intelligence
        if profile.outlier_analysis and profile.outlier_analysis.get('high_impact_outliers'):
            plan["analysis_phases"].append({
                "phase": "outlier_investigation",
                "description": "Outlier and anomaly analysis",
                "operations": [self.get_outlier_operation(table_name)]
            })
        
        # Add recommendations
        if profile.quality_profile:
            plan["recommendations"].extend(profile.quality_profile.quality_recommendations)
        
        if profile.outlier_analysis:
            plan["recommendations"].extend(profile.outlier_analysis.get('outlier_recommendations', []))
        
        return plan