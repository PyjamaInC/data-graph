"""
Rich Metadata Prompt Templates

This module provides sophisticated prompt templates for generating comprehensive
metadata using LLM capabilities with detailed profiling context.
"""

from typing import Dict, Any


class RichMetadataPromptTemplates:
    """Advanced prompt templates for comprehensive metadata generation"""
    
    def __init__(self):
        self.templates = {
            'business_summary': {
                'system': '''You are a senior data analyst and business intelligence expert. 
                Generate comprehensive, business-focused summaries that executives and stakeholders can understand.
                Focus on business impact, opportunities, and actionable insights.''',
                
                'user_template': '''Analyze this database table and create a comprehensive business summary.

TABLE ANALYSIS:
Name: {table_name}
Domain: {business_domain}
Type: {table_type}
Volume: {row_count} records, {column_count} columns

DATA QUALITY INSIGHTS:
- Overall Quality Score: {quality_score}/100
- Critical Issues: {critical_issues}
- Warning Issues: {warning_issues}
- Key Recommendations: {quality_recommendations}

COLUMN COMPOSITION:
- Business Measures: {measures}
- Business Dimensions: {dimensions}
- Identifiers: {identifiers}
- Time-based Columns: {temporal}

RELATIONSHIP PATTERNS:
- Strong Correlations Found: {strong_relationships}
- Redundant Features: {redundant_features}

ANOMALY DETECTION:
- High-Impact Outliers: {high_impact_outliers}
- Outlier Patterns: {outlier_recommendations}

ML READINESS:
- Score: {ml_score}/100
- Key Factors: {ml_factors}

Generate a comprehensive business summary that includes:
1. **Business Purpose**: What this table represents in business terms
2. **Data Quality Assessment**: Current state and improvement opportunities
3. **Key Insights**: Important patterns, relationships, and anomalies discovered
4. **Business Value**: How this data can drive business decisions
5. **Risk Assessment**: Data quality risks and mitigation strategies
6. **Recommended Actions**: Specific steps to improve data utility

Write in executive-friendly language, focusing on business impact rather than technical details.'''
            },
            
            'ml_strategy_recommendation': {
                'system': '''You are a machine learning architect and data scientist. 
                Provide strategic ML recommendations based on comprehensive data profiling.
                Focus on practical, actionable recommendations for ML implementation.''',
                
                'user_template': '''Based on comprehensive data profiling, provide ML strategy recommendations.

DATA PROFILE:
Table: {table_name}
Volume: {row_count} records
Features: {feature_count} total ({measure_count} measures, {dimension_count} dimensions)
Quality Score: {quality_score}/100
ML Readiness: {ml_readiness_score}/100

DETAILED ANALYSIS:
- Missing Data: {missing_percentage}%
- Outlier Impact: {outlier_impact}
- Correlation Density: {correlation_density}
- Temporal Features: {has_temporal}
- Data Types: {data_type_distribution}

QUALITY FACTORS:
{ml_readiness_factors}

Generate strategic ML recommendations covering:
1. **ML Readiness Assessment**: Current state for ML applications
2. **Recommended ML Use Cases**: Specific ML applications suited for this data
3. **Data Preparation Strategy**: Required preprocessing steps
4. **Feature Engineering Opportunities**: How to enhance predictive power
5. **Model Architecture Suggestions**: Suitable ML model types
6. **Risk Mitigation**: Data quality risks for ML models
7. **Performance Expectations**: Realistic accuracy expectations
8. **Implementation Roadmap**: Step-by-step ML deployment strategy

Focus on actionable, practical recommendations with specific next steps.'''
            },
            
            'data_story_narrative': {
                'system': '''You are a data storyteller who transforms complex data insights into compelling narratives.
                Create engaging stories that make data insights accessible to non-technical stakeholders.''',
                
                'user_template': '''Transform this data profiling analysis into a compelling data story.

DATA STORY ELEMENTS:
Table: {table_name}
Business Context: {business_domain}
Data Scale: {row_count} records across {column_count} dimensions

QUALITY NARRATIVE:
- Overall Health: {quality_score}/100
- Critical Issues: {critical_issues}
- Success Stories: {quality_strengths}

PATTERN DISCOVERIES:
- Correlations: {relationship_patterns}
- Anomalies: {anomaly_patterns}
- Trends: {temporal_patterns}

BUSINESS IMPLICATIONS:
- Opportunities: {business_opportunities}
- Risks: {business_risks}
- Value Potential: {value_assessment}

Create a narrative that includes:
1. **The Data Journey**: How this data tells a business story
2. **Hidden Patterns**: Surprising insights from correlation and outlier analysis
3. **Quality Narrative**: The data quality story and its implications
4. **Predictive Potential**: What this data can predict about the future
5. **Operational Impact**: How data quality affects business operations
6. **Strategic Opportunities**: Untapped potential in the data

Write as an engaging story that connects data insights to business outcomes.'''
            },
            
            'quality_deep_dive': {
                'system': '''You are a data quality specialist with deep expertise in data governance and quality management.
                Provide detailed, actionable analysis of data quality issues and improvement strategies.''',
                
                'user_template': '''Perform a deep dive analysis of data quality for this dataset.

QUALITY METRICS:
Overall Score: {quality_score}/100
Alert Breakdown:
- Critical: {critical_alerts}
- Warning: {warning_alerts} 
- Info: {info_alerts}

SPECIFIC ISSUES:
Critical Problems: {critical_issues}
Data Patterns: {data_patterns}
Distribution Issues: {distribution_problems}
Missing Data Patterns: {missing_patterns}

IMPACT ASSESSMENT:
Business Impact: {business_impact}
Analysis Reliability: {analysis_impact}
ML Suitability: {ml_impact}

Provide comprehensive analysis including:
1. **Root Cause Analysis**: Likely causes of quality issues
2. **Impact Assessment**: How quality affects business operations
3. **Remediation Strategy**: Step-by-step improvement plan
4. **Monitoring Framework**: Ongoing quality assurance approach
5. **Cost-Benefit Analysis**: Investment vs. quality improvement
6. **Success Metrics**: How to measure quality improvements

Focus on practical, implementable solutions with clear priorities.'''
            },
            
            'executive_dashboard': {
                'system': '''You are a business intelligence expert creating executive dashboards.
                Generate clear, high-level insights for C-level executives and senior management.''',
                
                'user_template': '''Create executive dashboard insights for this data asset.

ASSET OVERVIEW:
Name: {table_name}
Business Function: {business_domain}
Strategic Value: {strategic_value}

KEY METRICS:
- Data Volume: {row_count} records
- Quality Score: {quality_score}%
- ML Readiness: {ml_readiness}%
- Business Criticality: {criticality_level}

OPPORTUNITY ANALYSIS:
Growth Potential: {growth_potential}
Risk Factors: {risk_factors}
Investment Priority: {investment_priority}

COMPETITIVE ADVANTAGE:
Unique Insights: {unique_insights}
Market Differentiation: {differentiation_potential}

Generate executive summary including:
1. **Strategic Value Proposition**: Why this data matters to business strategy
2. **Key Performance Indicators**: Most important metrics for tracking
3. **Investment Recommendations**: Where to allocate resources
4. **Risk Management**: Critical risks and mitigation strategies
5. **Competitive Advantage**: How this data creates business value
6. **Success Timeline**: Expected outcomes and milestones

Present in bullet points suitable for executive presentation.'''
            },
            
            'technical_architecture': {
                'system': '''You are a data architecture expert and technical lead.
                Provide detailed technical recommendations for data infrastructure and processing.''',
                
                'user_template': '''Analyze technical architecture requirements for this dataset.

TECHNICAL PROFILE:
Dataset: {table_name}
Scale: {row_count} records, {column_count} columns
Growth Rate: {growth_pattern}
Access Patterns: {access_patterns}

PROCESSING REQUIREMENTS:
Query Complexity: {query_complexity}
Real-time Needs: {realtime_requirements}
Analytics Workload: {analytics_load}

QUALITY CONSTRAINTS:
Data Quality: {quality_score}%
Processing Reliability: {reliability_needs}
Compliance Requirements: {compliance_needs}

Provide technical recommendations for:
1. **Storage Architecture**: Optimal storage solution and partitioning
2. **Processing Framework**: ETL/ELT recommendations and tools
3. **Performance Optimization**: Indexing, caching, and query optimization
4. **Quality Assurance**: Automated quality monitoring and validation
5. **Scalability Planning**: Handling growth and increased demand
6. **Integration Strategy**: Connecting with existing systems
7. **Monitoring and Alerting**: Operational monitoring requirements

Focus on practical, cost-effective technical solutions.'''
            },
            
            'compliance_assessment': {
                'system': '''You are a data governance and compliance expert.
                Assess compliance requirements and data governance implications.''',
                
                'user_template': '''Assess compliance and governance requirements for this dataset.

DATASET PROFILE:
Name: {table_name}
Business Domain: {business_domain}
Data Sensitivity: {sensitivity_level}
Geographic Scope: {geographic_scope}

DATA CHARACTERISTICS:
Personal Data Elements: {personal_data_indicators}
Financial Information: {financial_data_indicators}
Confidential Business Data: {confidential_indicators}

QUALITY AND LINEAGE:
Data Quality: {quality_score}%
Source Systems: {source_systems}
Data Lineage: {lineage_clarity}

Provide compliance assessment covering:
1. **Regulatory Requirements**: Applicable regulations (GDPR, CCPA, SOX, etc.)
2. **Data Classification**: Sensitivity and confidentiality levels
3. **Access Controls**: Required permissions and restrictions
4. **Retention Policies**: Data lifecycle and deletion requirements
5. **Audit Requirements**: Logging and monitoring needs
6. **Privacy Considerations**: Personal data protection measures
7. **Risk Assessment**: Compliance risks and mitigation strategies

Provide clear, actionable governance recommendations.'''
            },
            
            'predictive_analytics_roadmap': {
                'system': '''You are a predictive analytics consultant and data science strategist.
                Create strategic roadmaps for implementing predictive analytics capabilities.''',
                
                'user_template': '''Create a predictive analytics roadmap for this dataset.

ANALYTICS FOUNDATION:
Dataset: {table_name}
Predictive Potential: {predictive_score}%
Historical Depth: {temporal_coverage}
Feature Richness: {feature_diversity}

CURRENT CAPABILITIES:
ML Readiness: {ml_readiness}%
Data Quality: {quality_score}%
Processing Infrastructure: {infrastructure_maturity}

BUSINESS CONTEXT:
Use Case Potential: {use_case_opportunities}
Business Value: {business_value_potential}
Implementation Complexity: {complexity_assessment}

Develop comprehensive roadmap including:
1. **Use Case Prioritization**: Most valuable predictive applications
2. **Technical Roadmap**: Infrastructure and capability development
3. **Data Preparation Strategy**: Quality improvement and feature engineering
4. **Model Development Plan**: Algorithm selection and validation approach
5. **Deployment Strategy**: Production implementation and monitoring
6. **Success Metrics**: KPIs and ROI measurement
7. **Risk Mitigation**: Addressing model and data risks
8. **Resource Requirements**: Team, technology, and budget needs

Provide 6-12 month implementation timeline with clear milestones.'''
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, str]:
        """Get a specific template by name"""
        return self.templates.get(template_name, {})
    
    def list_templates(self) -> list:
        """List all available template names"""
        return list(self.templates.keys())
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """Format a template with provided context variables"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        user_template = template.get('user_template', '')
        try:
            return user_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
    
    def get_system_prompt(self, template_name: str) -> str:
        """Get the system prompt for a specific template"""
        template = self.get_template(template_name)
        return template.get('system', 'You are a helpful data analysis assistant.')


class PromptContextBuilder:
    """Helper class to build context dictionaries for prompt templates"""
    
    @staticmethod
    def build_business_summary_context(enhanced_profile) -> Dict[str, Any]:
        """Build context for business summary template"""
        return {
            'table_name': enhanced_profile.table_name,
            'business_domain': enhanced_profile.business_domain or 'General',
            'table_type': enhanced_profile.table_type,
            'row_count': f"{enhanced_profile.row_count:,}",
            'column_count': enhanced_profile.column_count,
            'quality_score': enhanced_profile.quality_profile.overall_quality_score if enhanced_profile.quality_profile else 'N/A',
            'critical_issues': len(enhanced_profile.quality_profile.critical_alerts) if enhanced_profile.quality_profile else 0,
            'warning_issues': len(enhanced_profile.quality_profile.warning_alerts) if enhanced_profile.quality_profile else 0,
            'quality_recommendations': ', '.join(enhanced_profile.quality_profile.quality_recommendations[:3]) if enhanced_profile.quality_profile else 'None',
            'measures': ', '.join(enhanced_profile.measure_columns[:5]),
            'dimensions': ', '.join(enhanced_profile.dimension_columns[:5]),
            'identifiers': ', '.join(enhanced_profile.identifier_columns[:3]),
            'temporal': ', '.join(enhanced_profile.temporal_columns),
            'strong_relationships': len(enhanced_profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', [])) if enhanced_profile.correlation_analysis else 0,
            'redundant_features': len(enhanced_profile.correlation_analysis.get('feature_redundancy', [])) if enhanced_profile.correlation_analysis else 0,
            'high_impact_outliers': len(enhanced_profile.outlier_analysis.get('high_impact_outliers', [])) if enhanced_profile.outlier_analysis else 0,
            'outlier_recommendations': ', '.join(enhanced_profile.outlier_analysis.get('outlier_recommendations', [])[:2]) if enhanced_profile.outlier_analysis else 'None',
            'ml_score': enhanced_profile.ml_readiness_score or 0,
            'ml_factors': ', '.join(enhanced_profile.ml_readiness_factors[:3]) if enhanced_profile.ml_readiness_factors else 'Not assessed'
        }
    
    @staticmethod
    def build_ml_strategy_context(enhanced_profile) -> Dict[str, Any]:
        """Build context for ML strategy template"""
        return {
            'table_name': enhanced_profile.table_name,
            'row_count': f"{enhanced_profile.row_count:,}",
            'feature_count': enhanced_profile.column_count,
            'measure_count': len(enhanced_profile.measure_columns),
            'dimension_count': len(enhanced_profile.dimension_columns),
            'quality_score': enhanced_profile.quality_profile.overall_quality_score if enhanced_profile.quality_profile else 0,
            'ml_readiness_score': enhanced_profile.ml_readiness_score or 0,
            'missing_percentage': 100 - (enhanced_profile.data_quality_score * 100),
            'outlier_impact': 'High' if enhanced_profile.outlier_analysis and len(enhanced_profile.outlier_analysis.get('high_impact_outliers', [])) > 0 else 'Low',
            'correlation_density': 'High' if enhanced_profile.correlation_analysis and len(enhanced_profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', [])) > 3 else 'Low',
            'has_temporal': 'Yes' if enhanced_profile.temporal_columns else 'No',
            'data_type_distribution': f"{len(enhanced_profile.measure_columns)} numeric, {len(enhanced_profile.dimension_columns)} categorical",
            'ml_readiness_factors': '; '.join(enhanced_profile.ml_readiness_factors) if enhanced_profile.ml_readiness_factors else 'Not assessed'
        }
    
    @staticmethod
    def build_data_story_context(enhanced_profile) -> Dict[str, Any]:
        """Build context for data story template"""
        return {
            'table_name': enhanced_profile.table_name,
            'business_domain': enhanced_profile.business_domain or 'General Business',
            'row_count': f"{enhanced_profile.row_count:,}",
            'column_count': enhanced_profile.column_count,
            'quality_score': enhanced_profile.quality_profile.overall_quality_score if enhanced_profile.quality_profile else 0,
            'critical_issues': len(enhanced_profile.quality_profile.critical_alerts) if enhanced_profile.quality_profile else 0,
            'quality_strengths': 'High completeness' if enhanced_profile.data_quality_score > 0.9 else 'Adequate data coverage',
            'relationship_patterns': f"{len(enhanced_profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', []))} strong correlations" if enhanced_profile.correlation_analysis else 'Correlation analysis completed',
            'anomaly_patterns': f"{len(enhanced_profile.outlier_analysis.get('high_impact_outliers', []))} significant outliers" if enhanced_profile.outlier_analysis else 'Outlier analysis completed',
            'temporal_patterns': 'Time-based trends available' if enhanced_profile.temporal_columns else 'Static dataset',
            'business_opportunities': 'ML/Analytics applications' if (enhanced_profile.ml_readiness_score or 0) > 60 else 'Data quality improvements',
            'business_risks': 'Data quality concerns' if enhanced_profile.data_quality_score < 0.8 else 'Minimal data risks',
            'value_assessment': 'High' if enhanced_profile.row_count > 10000 and enhanced_profile.data_quality_score > 0.8 else 'Medium'
        }
    
    @staticmethod
    def build_quality_deep_dive_context(enhanced_profile) -> Dict[str, Any]:
        """Build context for quality deep dive template"""
        if not enhanced_profile.quality_profile:
            return {
                'quality_score': enhanced_profile.data_quality_score * 100,
                'critical_alerts': 0,
                'warning_alerts': 0,
                'info_alerts': 0,
                'critical_issues': 'No detailed quality analysis available',
                'data_patterns': 'Basic profiling completed',
                'distribution_problems': 'Not assessed',
                'missing_patterns': 'Not assessed',
                'business_impact': 'Medium',
                'analysis_impact': 'Medium',
                'ml_impact': 'Medium'
            }
        
        return {
            'quality_score': enhanced_profile.quality_profile.overall_quality_score,
            'critical_alerts': len(enhanced_profile.quality_profile.critical_alerts),
            'warning_alerts': len(enhanced_profile.quality_profile.warning_alerts),
            'info_alerts': len(enhanced_profile.quality_profile.info_alerts),
            'critical_issues': '; '.join([alert.get('alert_type', 'Unknown') for alert in enhanced_profile.quality_profile.critical_alerts[:3]]),
            'data_patterns': 'Distribution and correlation analysis completed',
            'distribution_problems': f"{len(enhanced_profile.quality_profile.distribution_alerts)} distribution issues" if hasattr(enhanced_profile.quality_profile, 'distribution_alerts') else 'Not assessed',
            'missing_patterns': 'Missing data patterns analyzed',
            'business_impact': 'High' if enhanced_profile.quality_profile.overall_quality_score < 70 else 'Low',
            'analysis_impact': 'High' if enhanced_profile.quality_profile.overall_quality_score < 80 else 'Medium',
            'ml_impact': 'High' if enhanced_profile.quality_profile.overall_quality_score < 85 else 'Low'
        }
    
    @staticmethod
    def build_executive_dashboard_context(enhanced_profile) -> Dict[str, Any]:
        """Build context for executive dashboard template"""
        return {
            'table_name': enhanced_profile.table_name,
            'business_domain': enhanced_profile.business_domain or 'General Business',
            'strategic_value': 'High' if enhanced_profile.row_count > 50000 and enhanced_profile.data_quality_score > 0.8 else 'Medium',
            'row_count': f"{enhanced_profile.row_count:,}",
            'quality_score': int(enhanced_profile.data_quality_score * 100),
            'ml_readiness': int(enhanced_profile.ml_readiness_score) if enhanced_profile.ml_readiness_score else 0,
            'criticality_level': 'High' if enhanced_profile.row_count > 100000 else 'Medium',
            'growth_potential': 'High' if (enhanced_profile.ml_readiness_score or 0) > 70 else 'Medium',
            'risk_factors': 'Data quality issues' if enhanced_profile.data_quality_score < 0.8 else 'Minimal risks',
            'investment_priority': 'High' if enhanced_profile.data_quality_score > 0.8 and enhanced_profile.row_count > 10000 else 'Medium',
            'unique_insights': 'Correlation patterns and anomalies detected',
            'differentiation_potential': 'High' if enhanced_profile.temporal_columns else 'Medium'
        }