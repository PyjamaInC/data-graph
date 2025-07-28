"""
Enhanced LLM Semantic Summarizer with Rich Metadata Generation

This module extends the basic LLM summarizer to generate comprehensive, 
business-focused metadata using detailed profiling insights from ydata-profiling.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .llm_config import LLMSemanticSummarizer

logger = logging.getLogger(__name__)


@dataclass
class RichMetadata:
    """Comprehensive metadata structure"""
    business_summary: str
    data_quality_narrative: str
    ml_readiness_assessment: str
    relationship_insights: str
    anomaly_explanation: str
    usage_recommendations: List[str]
    technical_documentation: str
    executive_summary: Dict[str, Any]
    ml_strategy_guide: Dict[str, Any]
    data_quality_report: Dict[str, Any]


class EnhancedLLMSemanticSummarizer:
    """Enhanced LLM summarizer using comprehensive profiling insights"""
    
    def __init__(self, base_summarizer: 'LLMSemanticSummarizer'):
        """Initialize with base LLM summarizer"""
        self.base_summarizer = base_summarizer
        self.logger = logging.getLogger(__name__)
    
    def generate_rich_metadata(self, 
                              table_name: str,
                              enhanced_profile: 'EnhancedTableProfile') -> RichMetadata:
        """Generate comprehensive metadata using all profiling insights"""
        
        self.logger.info(f"Generating rich metadata for {table_name}")
        
        # Create rich context from profiling data
        profiling_context = self._build_profiling_context(enhanced_profile)
        
        # Generate different types of metadata
        try:
            business_summary = self._generate_business_summary(table_name, profiling_context)
            quality_narrative = self._generate_quality_narrative(enhanced_profile.quality_profile)
            ml_assessment = self._generate_ml_assessment(enhanced_profile)
            relationship_insights = self._generate_relationship_insights(enhanced_profile.correlation_analysis)
            anomaly_explanation = self._generate_anomaly_explanation(enhanced_profile.outlier_analysis)
            usage_recommendations = self._generate_usage_recommendations(enhanced_profile)
            technical_docs = self._generate_technical_docs(enhanced_profile)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(enhanced_profile)
            
            # Generate ML strategy guide
            ml_strategy_guide = self._generate_ml_strategy_guide(enhanced_profile)
            
            # Generate data quality report
            data_quality_report = self._generate_data_quality_report(enhanced_profile)
            
            return RichMetadata(
                business_summary=business_summary,
                data_quality_narrative=quality_narrative,
                ml_readiness_assessment=ml_assessment,
                relationship_insights=relationship_insights,
                anomaly_explanation=anomaly_explanation,
                usage_recommendations=usage_recommendations,
                technical_documentation=technical_docs,
                executive_summary=executive_summary,
                ml_strategy_guide=ml_strategy_guide,
                data_quality_report=data_quality_report
            )
            
        except Exception as e:
            self.logger.error(f"Error generating rich metadata: {e}")
            # Return basic metadata on error
            return self._generate_fallback_metadata(table_name, enhanced_profile)
    
    def _build_profiling_context(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Build comprehensive context from profiling data"""
        context = {
            'basic_info': {
                'table_name': profile.table_name,
                'rows': f"{profile.row_count:,}",
                'columns': profile.column_count,
                'business_domain': profile.business_domain,
                'table_type': profile.table_type
            },
            
            'data_quality': {
                'overall_score': profile.quality_profile.overall_quality_score if profile.quality_profile else None,
                'critical_issues': len(profile.quality_profile.critical_alerts) if profile.quality_profile else 0,
                'warning_issues': len(profile.quality_profile.warning_alerts) if profile.quality_profile else 0,
                'recommendations': profile.quality_profile.quality_recommendations if profile.quality_profile else []
            },
            
            'column_composition': {
                'measures': profile.measure_columns,
                'dimensions': profile.dimension_columns,
                'identifiers': profile.identifier_columns,
                'temporal': profile.temporal_columns
            },
            
            'correlation_insights': {
                'strong_relationships': len(profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', [])) if profile.correlation_analysis else 0,
                'redundant_features': len(profile.correlation_analysis.get('feature_redundancy', [])) if profile.correlation_analysis else 0
            },
            
            'outlier_insights': {
                'high_impact_outliers': len(profile.outlier_analysis.get('high_impact_outliers', [])) if profile.outlier_analysis else 0,
                'outlier_recommendations': profile.outlier_analysis.get('outlier_recommendations', []) if profile.outlier_analysis else []
            },
            
            'ml_readiness': {
                'score': profile.ml_readiness_score,
                'factors': profile.ml_readiness_factors
            }
        }
        
        return context
    
    def _generate_business_summary(self, table_name: str, profiling_context: Dict[str, Any]) -> str:
        """Generate comprehensive business summary"""
        
        prompt = f"""
        Analyze this database table and create a comprehensive business summary.

        TABLE: {table_name}
        BASIC INFO: {profiling_context['basic_info']}
        DATA QUALITY: {profiling_context['data_quality']}
        COLUMNS: {profiling_context['column_composition']}
        RELATIONSHIPS: {profiling_context['correlation_insights']}
        ANOMALIES: {profiling_context['outlier_insights']}
        ML READINESS: {profiling_context['ml_readiness']}

        Generate a comprehensive business summary that includes:
        1. Business Purpose: What this table represents in business terms
        2. Data Quality Assessment: Current state and improvement opportunities
        3. Key Insights: Important patterns, relationships, and anomalies discovered
        4. Business Value: How this data can drive business decisions
        5. Risk Assessment: Data quality risks and mitigation strategies
        6. Recommended Actions: Specific steps to improve data utility

        Write in executive-friendly language, focusing on business impact rather than technical details.
        """
        
        try:
            return self.base_summarizer.generate_summary(prompt, max_tokens=1200)  # 🚀 INCREASED from 500 to 1200!
        except Exception as e:
            self.logger.warning(f"Failed to generate business summary: {e}")
            return f"Business summary for {table_name} - comprehensive analysis of {profiling_context['basic_info']['rows']} records across {profiling_context['basic_info']['columns']} columns."
    
    def _generate_quality_narrative(self, quality_profile: Optional[Any]) -> str:
        """Generate comprehensive data quality narrative"""
        
        if not quality_profile:
            return "No quality profile available for detailed analysis."
        
        prompt = f"""
        Create a compelling data quality story based on this analysis:

        OVERALL QUALITY SCORE: {quality_profile.overall_quality_score}/100
        CRITICAL ISSUES: {len(quality_profile.critical_alerts)}
        WARNING ISSUES: {len(quality_profile.warning_alerts)}
        RECOMMENDATIONS: {quality_profile.quality_recommendations}

        Create a narrative that explains:
        1. The current state of data quality
        2. Key quality issues and their business impact
        3. Quality improvement opportunities
        4. Risk mitigation strategies
        5. Expected outcomes from quality improvements

        Write as an engaging story that makes data quality insights accessible to business stakeholders.
        """
        
        try:
            return self.base_summarizer.generate_summary(prompt, max_tokens=400)
        except Exception as e:
            self.logger.warning(f"Failed to generate quality narrative: {e}")
            return f"Data quality assessment shows score of {quality_profile.overall_quality_score}/100 with opportunities for improvement."
    
    def _generate_ml_assessment(self, profile: 'EnhancedTableProfile') -> str:
        """Generate ML readiness assessment"""
        
        prompt = f"""
        Assess the machine learning potential of this dataset:

        TABLE: {profile.table_name}
        RECORDS: {profile.row_count:,}
        QUALITY: {profile.data_quality_score * 100:.1f}%
        ML READINESS: {profile.ml_readiness_score or 0:.1f}%
        MEASURES: {len(profile.measure_columns)} numerical columns
        DIMENSIONS: {len(profile.dimension_columns)} categorical columns
        TEMPORAL: {len(profile.temporal_columns)} time-based columns

        Provide detailed ML assessment covering:
        1. Overall ML Suitability Score and reasoning
        2. Recommended ML Use Cases (classification, regression, clustering, etc.)
        3. Feature Engineering Opportunities
        4. Data Preprocessing Requirements
        5. Expected Model Performance Factors
        6. Implementation Timeline and Resource Requirements
        7. Business Value Potential from ML Applications

        Be specific about algorithms, techniques, and expected outcomes.
        """
        
        try:
            return self.base_summarizer.generate_summary(prompt, max_tokens=1000)  # 🚀 INCREASED from 400 to 1000!
        except Exception as e:
            self.logger.warning(f"Failed to generate ML assessment: {e}")
            return f"ML assessment for {profile.table_name} - {profile.ml_readiness_score or 0:.1f}% readiness score based on data quality and feature composition."
    
    def _generate_relationship_insights(self, correlation_analysis: Optional[Dict]) -> str:
        """Generate relationship and correlation insights"""
        
        if not correlation_analysis:
            return "No advanced correlation analysis available. Consider enabling correlation analysis for deeper insights."
        
        strong_corr = correlation_analysis.get('strong_correlations', [])
        interactions = correlation_analysis.get('interaction_effects', {})
        
        prompt = f"""
        Analyze these data relationships and correlations:

        STRONG CORRELATIONS: {len(strong_corr)} significant relationships found
        INTERACTION EFFECTS: {len(interactions)} complex interactions detected

        CORRELATION DETAILS:
        {str(strong_corr)}

        INTERACTION DETAILS:
        {str(interactions)}

        Provide comprehensive relationship analysis including:
        1. Key Relationship Patterns: What the strongest correlations reveal
        2. Business Implications: How these relationships impact business decisions
        3. Causal vs Correlation Analysis: Which relationships suggest causation
        4. Feature Redundancy Assessment: Overlapping or duplicate information
        5. Data Integration Opportunities: How to leverage relationships for insights
        6. Predictive Modeling Implications: Which relationships enable forecasting
        7. Risk Factors: Relationships that could indicate data quality issues

        Focus on actionable business insights rather than statistical details.
        """
        
        try:
            return self.base_summarizer.generate_summary(prompt, max_tokens=1000)  # 🚀 INCREASED from 400 to 1000!
        except Exception as e:
            self.logger.warning(f"Failed to generate relationship insights: {e}")
            return f"Relationship analysis found {len(strong_corr)} strong correlations and {len(interactions)} interaction effects requiring further investigation."
    
    def _generate_anomaly_explanation(self, outlier_analysis: Optional[Dict]) -> str:
        """Generate anomaly explanation from outlier analysis"""
        
        if not outlier_analysis:
            return "No outlier analysis available for anomaly insights."
        
        high_impact_outliers = outlier_analysis.get('high_impact_outliers', [])
        
        prompt = f"""
        Explain the anomalies and outliers detected in this data:

        HIGH IMPACT OUTLIERS: {len(high_impact_outliers)} detected
        OUTLIER PATTERNS: {outlier_analysis.get('outlier_recommendations', [])}

        Provide an explanation covering:
        1. Types of anomalies detected
        2. Potential business causes
        3. Impact on analysis and reporting
        4. Investigation recommendations
        5. Treatment strategies

        Write for business analysts who need to understand and act on these anomalies.
        """
        
        try:
            return self.base_summarizer.generate_summary(prompt, max_tokens=300)
        except Exception as e:
            self.logger.warning(f"Failed to generate anomaly explanation: {e}")
            return f"Outlier analysis detected {len(high_impact_outliers)} high-impact anomalies requiring investigation."
    
    def _generate_usage_recommendations(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Generate usage recommendations based on profile"""
        
        recommendations = []
        
        # Data quality recommendations
        if profile.quality_profile and profile.quality_profile.overall_quality_score < 80:
            recommendations.append("Implement data quality monitoring and validation processes")
        
        # ML recommendations
        if profile.ml_readiness_score and profile.ml_readiness_score > 70:
            recommendations.append("Consider implementing predictive analytics and ML models")
        
        # Performance recommendations
        if profile.row_count > 1000000:
            recommendations.append("Implement data partitioning and indexing strategies for performance")
        
        # Temporal recommendations
        if profile.temporal_columns:
            recommendations.append("Leverage temporal data for trend analysis and forecasting")
        
        # Correlation recommendations
        if profile.correlation_analysis:
            redundant_count = len(profile.correlation_analysis.get('feature_redundancy', []))
            if redundant_count > 0:
                recommendations.append("Review redundant features for potential consolidation")
        
        return recommendations
    
    def _generate_technical_docs(self, profile: 'EnhancedTableProfile') -> str:
        """Generate technical documentation"""
        
        return f"""
        # Technical Documentation: {profile.table_name}
        
        ## Overview
        - Records: {profile.row_count:,}
        - Columns: {profile.column_count}
        - Business Domain: {profile.business_domain or 'Not specified'}
        - Table Type: {profile.table_type}
        
        ## Column Classification
        - Measures: {len(profile.measure_columns)} ({', '.join(profile.measure_columns[:5])})
        - Dimensions: {len(profile.dimension_columns)} ({', '.join(profile.dimension_columns[:5])})
        - Identifiers: {len(profile.identifier_columns)} ({', '.join(profile.identifier_columns[:3])})
        - Temporal: {len(profile.temporal_columns)} ({', '.join(profile.temporal_columns)})
        
        ## Quality Metrics
        - Data Quality Score: {profile.data_quality_score:.2f}
        - ML Readiness Score: {profile.ml_readiness_score or 'Not assessed'}
        
        ## Key Insights
        {', '.join(profile.key_concepts)}
        """
    
    def _generate_executive_summary(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate executive-level summary"""
        
        return {
            'key_metrics': {
                'data_volume': f"{profile.row_count:,} records",
                'data_quality': f"{profile.data_quality_score:.1f}%",
                'ml_readiness': f"{profile.ml_readiness_score or 0:.1f}%",
                'business_value': self._assess_business_value(profile)
            },
            'risk_indicators': self._identify_risk_indicators(profile),
            'opportunity_score': self._calculate_opportunity_score(profile),
            'strategic_recommendations': self._generate_strategic_recommendations(profile)
        }
    
    def _generate_ml_strategy_guide(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate comprehensive ML strategy guide"""
        
        return {
            'readiness_assessment': {
                'score': profile.ml_readiness_score or 0,
                'factors': profile.ml_readiness_factors or [],
                'recommendations': self._get_ml_readiness_recommendations(profile)
            },
            'use_case_recommendations': self._recommend_ml_use_cases(profile),
            'preprocessing_requirements': self._identify_preprocessing_needs(profile),
            'feature_engineering_opportunities': self._identify_feature_opportunities(profile)
        }
    
    def _generate_data_quality_report(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        if not profile.quality_profile:
            return {'status': 'No quality profile available'}
        
        return {
            'overall_score': profile.quality_profile.overall_quality_score,
            'issue_breakdown': {
                'critical': len(profile.quality_profile.critical_alerts),
                'warning': len(profile.quality_profile.warning_alerts),
                'info': len(profile.quality_profile.info_alerts)
            },
            'improvement_roadmap': profile.quality_profile.quality_recommendations,
            'business_impact': self._assess_quality_business_impact(profile.quality_profile)
        }
    
    def _generate_fallback_metadata(self, table_name: str, profile: 'EnhancedTableProfile') -> RichMetadata:
        """Generate fallback metadata when LLM generation fails"""
        
        return RichMetadata(
            business_summary=f"Table {table_name} contains {profile.row_count:,} records with {profile.column_count} columns.",
            data_quality_narrative=f"Data quality score: {profile.data_quality_score:.2f}",
            ml_readiness_assessment=f"ML readiness score: {profile.ml_readiness_score or 0:.2f}",
            relationship_insights="Correlation analysis completed",
            anomaly_explanation="Outlier detection completed",
            usage_recommendations=["Review data quality", "Consider analytics applications"],
            technical_documentation=self._generate_technical_docs(profile),
            executive_summary={'status': 'Basic summary available'},
            ml_strategy_guide={'status': 'Basic strategy available'},
            data_quality_report={'score': profile.data_quality_score}
        )
    
    # Helper methods
    def _assess_business_value(self, profile: 'EnhancedTableProfile') -> str:
        """Assess business value of the dataset"""
        if profile.row_count > 100000 and profile.data_quality_score > 0.8:
            return "High"
        elif profile.row_count > 10000 and profile.data_quality_score > 0.6:
            return "Medium"
        else:
            return "Low"
    
    def _identify_risk_indicators(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Identify key risk indicators"""
        risks = []
        
        if profile.data_quality_score < 0.7:
            risks.append("Low data quality")
        
        if profile.quality_profile and len(profile.quality_profile.critical_alerts) > 0:
            risks.append("Critical data issues detected")
        
        if profile.row_count < 1000:
            risks.append("Insufficient data volume")
        
        return risks
    
    def _calculate_opportunity_score(self, profile: 'EnhancedTableProfile') -> float:
        """Calculate opportunity score"""
        score = 0
        
        # Volume factor
        if profile.row_count > 100000:
            score += 30
        elif profile.row_count > 10000:
            score += 20
        elif profile.row_count > 1000:
            score += 10
        
        # Quality factor
        score += profile.data_quality_score * 0.4
        
        # ML readiness factor
        if profile.ml_readiness_score:
            score += profile.ml_readiness_score * 0.3
        
        return min(100, score)
    
    def _generate_strategic_recommendations(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        if profile.ml_readiness_score and profile.ml_readiness_score > 70:
            recommendations.append("Prioritize for AI/ML initiatives")
        
        if profile.data_quality_score < 0.8:
            recommendations.append("Invest in data quality improvements")
        
        if len(profile.temporal_columns) > 0:
            recommendations.append("Leverage for predictive analytics")
        
        return recommendations
    
    def _get_ml_readiness_recommendations(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Get ML readiness recommendations"""
        recommendations = []
        
        if not profile.ml_readiness_score or profile.ml_readiness_score < 50:
            recommendations.append("Focus on data quality improvements")
            recommendations.append("Increase data volume if possible")
        elif profile.ml_readiness_score < 70:
            recommendations.append("Address remaining data quality issues")
            recommendations.append("Consider feature engineering")
        else:
            recommendations.append("Ready for ML model development")
            recommendations.append("Consider advanced ML techniques")
        
        return recommendations
    
    def _recommend_ml_use_cases(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Recommend ML use cases based on profile"""
        use_cases = []
        
        if profile.temporal_columns:
            use_cases.append("Time series forecasting")
            use_cases.append("Trend analysis")
        
        if len(profile.measure_columns) > 2:
            use_cases.append("Predictive modeling")
            use_cases.append("Anomaly detection")
        
        if len(profile.dimension_columns) > 3:
            use_cases.append("Classification models")
            use_cases.append("Customer segmentation")
        
        return use_cases
    
    def _identify_preprocessing_needs(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Identify preprocessing requirements"""
        needs = []
        
        if profile.data_quality_score < 0.9:
            needs.append("Data cleaning and validation")
        
        if profile.outlier_analysis and len(profile.outlier_analysis.get('high_impact_outliers', [])) > 0:
            needs.append("Outlier treatment")
        
        if len(profile.dimension_columns) > 0:
            needs.append("Categorical encoding")
        
        return needs
    
    def _identify_feature_opportunities(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Identify feature engineering opportunities"""
        opportunities = []
        
        if profile.temporal_columns:
            opportunities.append("Time-based features (day, month, season)")
        
        if profile.correlation_analysis:
            strong_corr = len(profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', []))
            if strong_corr > 0:
                opportunities.append("Interaction features from correlated variables")
        
        if len(profile.measure_columns) > 1:
            opportunities.append("Ratio and derived metrics")
        
        return opportunities
    
    def _assess_quality_business_impact(self, quality_profile: Any) -> str:
        """Assess business impact of data quality"""
        if quality_profile.overall_quality_score > 90:
            return "Minimal impact - high quality data supports reliable business decisions"
        elif quality_profile.overall_quality_score > 70:
            return "Moderate impact - some quality issues may affect analysis accuracy"
        else:
            return "High impact - significant quality issues require immediate attention"