"""
Advanced Metadata Generator

This module provides sophisticated metadata generation capabilities using LLM
and comprehensive profiling insights to create business-focused metadata.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from .enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer

from .rich_metadata_templates import RichMetadataPromptTemplates, PromptContextBuilder

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveMetadata:
    """Complete metadata structure with all generated content"""
    
    # Core metadata
    executive_summary: Dict[str, Any]
    technical_specifications: Dict[str, Any]
    ml_strategy_guide: Dict[str, Any]
    data_quality_report: Dict[str, Any]
    
    # Business intelligence
    business_intelligence_insights: Dict[str, Any]
    predictive_analytics_recommendations: Dict[str, Any]
    data_governance_guidelines: Dict[str, Any]
    usage_patterns_analysis: Dict[str, Any]
    
    # Advanced insights
    compliance_assessment: Optional[Dict[str, Any]] = None
    competitive_advantage_analysis: Optional[Dict[str, Any]] = None
    cost_benefit_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata about metadata
    generation_timestamp: str = ""
    generation_method: str = "enhanced_llm"
    confidence_score: float = 0.0


class AdvancedMetadataGenerator:
    """Generate sophisticated metadata using LLM and profiling insights"""
    
    def __init__(self, llm_summarizer: 'EnhancedLLMSemanticSummarizer'):
        """Initialize with enhanced LLM summarizer"""
        self.llm = llm_summarizer
        self.templates = RichMetadataPromptTemplates()
        self.context_builder = PromptContextBuilder()
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_metadata(self, enhanced_profile: 'EnhancedTableProfile') -> ComprehensiveMetadata:
        """Generate all types of rich metadata"""
        
        self.logger.info(f"Generating comprehensive metadata for {enhanced_profile.table_name}")
        
        try:
            # Generate core metadata components
            executive_summary = self._generate_executive_summary(enhanced_profile)
            technical_specs = self._generate_technical_specs(enhanced_profile)
            ml_strategy = self._generate_ml_strategy(enhanced_profile)
            quality_report = self._generate_quality_report(enhanced_profile)
            
            # Generate business intelligence components
            bi_insights = self._generate_bi_insights(enhanced_profile)
            predictive_recs = self._generate_predictive_recs(enhanced_profile)
            governance_guide = self._generate_governance_guide(enhanced_profile)
            usage_patterns = self._generate_usage_patterns(enhanced_profile)
            
            # Generate advanced components if applicable
            compliance_assessment = self._generate_compliance_assessment(enhanced_profile)
            competitive_analysis = self._generate_competitive_analysis(enhanced_profile)
            cost_benefit = self._generate_cost_benefit_analysis(enhanced_profile)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(enhanced_profile)
            
            return ComprehensiveMetadata(
                executive_summary=executive_summary,
                technical_specifications=technical_specs,
                ml_strategy_guide=ml_strategy,
                data_quality_report=quality_report,
                business_intelligence_insights=bi_insights,
                predictive_analytics_recommendations=predictive_recs,
                data_governance_guidelines=governance_guide,
                usage_patterns_analysis=usage_patterns,
                compliance_assessment=compliance_assessment,
                competitive_advantage_analysis=competitive_analysis,
                cost_benefit_analysis=cost_benefit,
                generation_timestamp=self._get_timestamp(),
                generation_method="enhanced_llm_with_profiling",
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive metadata: {e}")
            return self._generate_fallback_metadata(enhanced_profile)
    
    def _generate_executive_summary(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate executive-level summary"""
        
        try:
            # Build context for executive dashboard
            context = self.context_builder.build_executive_dashboard_context(profile)
            
            # Generate executive narrative
            executive_narrative = self._generate_with_template(
                'executive_dashboard',
                context,
                max_tokens=400
            )
            
            return {
                'narrative': executive_narrative,
                'key_metrics': self._extract_key_metrics(profile),
                'risk_indicators': self._identify_risk_indicators(profile),
                'opportunity_score': self._calculate_opportunity_score(profile),
                'strategic_recommendations': self._generate_strategic_recommendations(profile),
                'business_impact_assessment': self._assess_business_impact(profile),
                'investment_priority': self._assess_investment_priority(profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate executive summary: {e}")
            return self._fallback_executive_summary(profile)
    
    def _generate_technical_specs(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate technical specifications"""
        
        try:
            # Build technical context
            technical_context = {
                'table_name': profile.table_name,
                'row_count': f"{profile.row_count:,}",
                'column_count': profile.column_count,
                'growth_pattern': self._assess_growth_pattern(profile),
                'access_patterns': self._assess_access_patterns(profile),
                'query_complexity': self._assess_query_complexity(profile),
                'realtime_requirements': self._assess_realtime_needs(profile),
                'analytics_load': self._assess_analytics_load(profile),
                'quality_score': profile.data_quality_score * 100,
                'reliability_needs': self._assess_reliability_needs(profile),
                'compliance_needs': self._assess_compliance_needs(profile)
            }
            
            # Generate technical narrative
            technical_narrative = self._generate_with_template(
                'technical_architecture',
                technical_context,
                max_tokens=500
            )
            
            return {
                'narrative': technical_narrative,
                'storage_recommendations': self._recommend_storage_architecture(profile),
                'processing_recommendations': self._recommend_processing_framework(profile),
                'performance_optimizations': self._recommend_performance_optimizations(profile),
                'scalability_considerations': self._assess_scalability_needs(profile),
                'integration_requirements': self._assess_integration_needs(profile),
                'monitoring_strategy': self._recommend_monitoring_strategy(profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate technical specs: {e}")
            return self._fallback_technical_specs(profile)
    
    def _generate_ml_strategy(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate comprehensive ML strategy"""
        
        try:
            # Build ML strategy context
            ml_context = self.context_builder.build_ml_strategy_context(profile)
            
            # Generate ML strategy narrative
            ml_narrative = self._generate_with_template(
                'ml_strategy_recommendation',
                ml_context,
                max_tokens=600
            )
            
            return {
                'strategy_document': ml_narrative,
                'readiness_assessment': {
                    'score': profile.ml_readiness_score or 0,
                    'factors': profile.ml_readiness_factors or [],
                    'recommendations': self._get_ml_readiness_recommendations(profile)
                },
                'recommended_models': self._recommend_ml_models(profile),
                'preprocessing_pipeline': self._design_preprocessing_pipeline(profile),
                'feature_engineering_suggestions': self._suggest_feature_engineering(profile),
                'validation_strategy': self._design_validation_strategy(profile),
                'deployment_considerations': self._assess_deployment_considerations(profile),
                'performance_expectations': self._set_performance_expectations(profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate ML strategy: {e}")
            return self._fallback_ml_strategy(profile)
    
    def _generate_quality_report(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        try:
            # Build quality context
            quality_context = self.context_builder.build_quality_deep_dive_context(profile)
            
            # Generate quality narrative
            quality_narrative = self._generate_with_template(
                'quality_deep_dive',
                quality_context,
                max_tokens=500
            )
            
            if not profile.quality_profile:
                return {
                    'narrative': quality_narrative,
                    'overall_score': profile.data_quality_score * 100,
                    'issue_breakdown': {'basic_assessment': 'Limited quality analysis available'},
                    'improvement_roadmap': ['Implement comprehensive data profiling'],
                    'business_impact': self._assess_quality_business_impact_basic(profile)
                }
            
            return {
                'narrative': quality_narrative,
                'overall_score': profile.quality_profile.overall_quality_score,
                'issue_breakdown': {
                    'critical': len(profile.quality_profile.critical_alerts),
                    'warning': len(profile.quality_profile.warning_alerts),
                    'info': len(profile.quality_profile.info_alerts)
                },
                'improvement_roadmap': self._create_improvement_roadmap(profile.quality_profile),
                'quality_monitoring_suggestions': self._suggest_quality_monitoring(profile.quality_profile),
                'business_impact': self._assess_quality_business_impact(profile.quality_profile),
                'cost_of_poor_quality': self._estimate_quality_costs(profile),
                'quality_metrics_framework': self._design_quality_metrics(profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate quality report: {e}")
            return self._fallback_quality_report(profile)
    
    def _generate_bi_insights(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate business intelligence insights"""
        
        return {
            'analytical_opportunities': self._identify_analytical_opportunities(profile),
            'dashboard_recommendations': self._recommend_dashboards(profile),
            'kpi_suggestions': self._suggest_kpis(profile),
            'reporting_framework': self._design_reporting_framework(profile),
            'data_storytelling_opportunities': self._identify_storytelling_opportunities(profile),
            'stakeholder_value_propositions': self._create_stakeholder_value_props(profile)
        }
    
    def _generate_predictive_recs(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate predictive analytics recommendations"""
        
        try:
            # Build predictive context
            predictive_context = {
                'table_name': profile.table_name,
                'predictive_score': (profile.ml_readiness_score or 0),
                'temporal_coverage': 'Yes' if profile.temporal_columns else 'No',
                'feature_diversity': len(profile.measure_columns) + len(profile.dimension_columns),
                'ml_readiness': profile.ml_readiness_score or 0,
                'quality_score': profile.data_quality_score * 100,
                'infrastructure_maturity': self._assess_infrastructure_maturity(profile),
                'use_case_opportunities': len(self._identify_use_case_opportunities(profile)),
                'business_value_potential': self._assess_business_value_potential(profile),
                'complexity_assessment': self._assess_implementation_complexity(profile)
            }
            
            # Generate predictive roadmap
            predictive_narrative = self._generate_with_template(
                'predictive_analytics_roadmap',
                predictive_context,
                max_tokens=600
            )
            
            return {
                'roadmap_document': predictive_narrative,
                'use_case_prioritization': self._prioritize_use_cases(profile),
                'technical_roadmap': self._create_technical_roadmap(profile),
                'success_metrics': self._define_success_metrics(profile),
                'resource_requirements': self._estimate_resource_requirements(profile),
                'implementation_timeline': self._create_implementation_timeline(profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate predictive recommendations: {e}")
            return self._fallback_predictive_recs(profile)
    
    def _generate_governance_guide(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate data governance guidelines"""
        
        return {
            'data_stewardship_framework': self._design_stewardship_framework(profile),
            'access_control_recommendations': self._recommend_access_controls(profile),
            'lifecycle_management': self._design_lifecycle_management(profile),
            'audit_requirements': self._define_audit_requirements(profile),
            'privacy_considerations': self._assess_privacy_considerations(profile),
            'retention_policies': self._recommend_retention_policies(profile)
        }
    
    def _generate_usage_patterns(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Generate usage patterns analysis"""
        
        return {
            'access_patterns': self._analyze_access_patterns(profile),
            'query_patterns': self._analyze_query_patterns(profile),
            'user_personas': self._identify_user_personas(profile),
            'optimization_opportunities': self._identify_optimization_opportunities(profile),
            'capacity_planning': self._recommend_capacity_planning(profile),
            'performance_benchmarks': self._establish_performance_benchmarks(profile)
        }
    
    def _generate_compliance_assessment(self, profile: 'EnhancedTableProfile') -> Optional[Dict[str, Any]]:
        """Generate compliance assessment if applicable"""
        
        # Only generate if there are compliance indicators
        if not self._has_compliance_implications(profile):
            return None
        
        try:
            compliance_context = {
                'table_name': profile.table_name,
                'business_domain': profile.business_domain or 'General',
                'sensitivity_level': self._assess_sensitivity_level(profile),
                'geographic_scope': 'Global',  # Could be enhanced with actual geo data
                'personal_data_indicators': self._identify_personal_data_indicators(profile),
                'financial_data_indicators': self._identify_financial_data_indicators(profile),
                'confidential_indicators': self._identify_confidential_indicators(profile),
                'quality_score': profile.data_quality_score * 100,
                'source_systems': 'Various',  # Could be enhanced with lineage data
                'lineage_clarity': 'Medium'  # Could be enhanced with actual lineage analysis
            }
            
            compliance_narrative = self._generate_with_template(
                'compliance_assessment',
                compliance_context,
                max_tokens=500
            )
            
            return {
                'assessment_document': compliance_narrative,
                'regulatory_requirements': self._identify_regulatory_requirements(profile),
                'compliance_score': self._calculate_compliance_score(profile),
                'risk_assessment': self._assess_compliance_risks(profile),
                'remediation_plan': self._create_compliance_remediation_plan(profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate compliance assessment: {e}")
            return None
    
    def _generate_competitive_analysis(self, profile: 'EnhancedTableProfile') -> Optional[Dict[str, Any]]:
        """Generate competitive advantage analysis"""
        
        # Only generate if dataset has strategic value
        if not self._has_strategic_value(profile):
            return None
        
        return {
            'data_asset_value': self._assess_data_asset_value(profile),
            'uniqueness_factors': self._identify_uniqueness_factors(profile),
            'market_differentiation': self._assess_market_differentiation(profile),
            'competitive_moat': self._assess_competitive_moat(profile),
            'monetization_opportunities': self._identify_monetization_opportunities(profile)
        }
    
    def _generate_cost_benefit_analysis(self, profile: 'EnhancedTableProfile') -> Optional[Dict[str, Any]]:
        """Generate cost-benefit analysis"""
        
        return {
            'investment_categories': self._categorize_investments(profile),
            'expected_benefits': self._estimate_benefits(profile),
            'roi_projections': self._project_roi(profile),
            'payback_period': self._estimate_payback_period(profile),
            'risk_adjusted_value': self._calculate_risk_adjusted_value(profile)
        }
    
    def _generate_with_template(self, template_name: str, context: Dict[str, Any], max_tokens: int = 400) -> str:
        """Generate content using a template"""
        
        try:
            # Get system prompt
            system_prompt = self.templates.get_system_prompt(template_name)
            
            # Format user prompt
            user_prompt = self.templates.format_template(template_name, **context)
            
            # Generate using base LLM
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            return self.llm.base_summarizer.generate_summary(full_prompt, max_tokens=max_tokens)
            
        except Exception as e:
            self.logger.warning(f"Template generation failed for {template_name}: {e}")
            return f"Generated analysis for {template_name} - {context.get('table_name', 'dataset')}"
    
    def _calculate_confidence_score(self, profile: 'EnhancedTableProfile') -> float:
        """Calculate overall confidence score for metadata generation"""
        
        confidence = 0.0
        
        # Base confidence from data volume
        if profile.row_count > 10000:
            confidence += 30
        elif profile.row_count > 1000:
            confidence += 20
        else:
            confidence += 10
        
        # Quality confidence
        confidence += profile.data_quality_score * 30
        
        # Profiling completeness
        if profile.quality_profile:
            confidence += 20
        
        if profile.correlation_analysis:
            confidence += 10
        
        if profile.outlier_analysis:
            confidence += 10
        
        return min(100.0, confidence)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Fallback methods
    def _generate_fallback_metadata(self, profile: 'EnhancedTableProfile') -> ComprehensiveMetadata:
        """Generate fallback metadata when full generation fails"""
        
        return ComprehensiveMetadata(
            executive_summary=self._fallback_executive_summary(profile),
            technical_specifications=self._fallback_technical_specs(profile),
            ml_strategy_guide=self._fallback_ml_strategy(profile),
            data_quality_report=self._fallback_quality_report(profile),
            business_intelligence_insights={'status': 'Basic BI analysis available'},
            predictive_analytics_recommendations={'status': 'Basic predictive assessment available'},
            data_governance_guidelines={'status': 'Basic governance recommendations available'},
            usage_patterns_analysis={'status': 'Basic usage analysis available'},
            generation_timestamp=self._get_timestamp(),
            generation_method="fallback",
            confidence_score=50.0
        )
    
    def _fallback_executive_summary(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Fallback executive summary"""
        return {
            'narrative': f"Dataset {profile.table_name} contains {profile.row_count:,} records with data quality score of {profile.data_quality_score:.2f}",
            'key_metrics': self._extract_key_metrics(profile),
            'risk_indicators': ['Basic assessment available'],
            'opportunity_score': 50.0,
            'strategic_recommendations': ['Conduct detailed analysis']
        }
    
    def _fallback_technical_specs(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Fallback technical specifications"""
        return {
            'narrative': f"Technical analysis for {profile.table_name} with {profile.row_count:,} records",
            'storage_recommendations': ['Standard relational storage'],
            'processing_recommendations': ['ETL pipeline recommended'],
            'performance_optimizations': ['Index on key columns']
        }
    
    def _fallback_ml_strategy(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Fallback ML strategy"""
        return {
            'strategy_document': f"ML assessment for {profile.table_name}",
            'readiness_assessment': {
                'score': profile.ml_readiness_score or 0,
                'factors': profile.ml_readiness_factors or ['Assessment needed'],
                'recommendations': ['Improve data quality', 'Increase data volume']
            },
            'recommended_models': ['Supervised learning models']
        }
    
    def _fallback_quality_report(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Fallback quality report"""
        return {
            'narrative': f"Quality assessment for {profile.table_name}",
            'overall_score': profile.data_quality_score * 100,
            'issue_breakdown': {'general': 'Basic quality assessment completed'},
            'improvement_roadmap': ['Implement data profiling', 'Add quality monitoring']
        }
    
    # Helper methods for generating specific components
    def _extract_key_metrics(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Extract key metrics for executive summary"""
        return {
            'data_volume': f"{profile.row_count:,} records",
            'data_quality': f"{profile.data_quality_score * 100:.1f}%",
            'ml_readiness': f"{profile.ml_readiness_score or 0:.1f}%",
            'business_value': self._assess_business_value(profile)
        }
    
    def _identify_risk_indicators(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Identify key risk indicators"""
        risks = []
        
        if profile.data_quality_score < 0.7:
            risks.append("🔴 Low data quality detected")
        
        if profile.quality_profile and len(profile.quality_profile.critical_alerts) > 0:
            risks.append("🔴 Critical data issues require immediate attention")
        
        if profile.row_count < 1000:
            risks.append("🟡 Limited data volume may impact analysis")
        
        if not risks:
            risks.append("✅ No major risks identified")
        
        return risks
    
    def _calculate_opportunity_score(self, profile: 'EnhancedTableProfile') -> float:
        """Calculate opportunity score"""
        score = 0
        
        # Volume factor (40%)
        if profile.row_count > 100000:
            score += 40
        elif profile.row_count > 10000:
            score += 30
        elif profile.row_count > 1000:
            score += 20
        else:
            score += 10
        
        # Quality factor (30%)
        score += profile.data_quality_score * 30
        
        # ML readiness factor (20%)
        if profile.ml_readiness_score:
            score += profile.ml_readiness_score * 0.2
        
        # Feature diversity factor (10%)
        feature_diversity = len(profile.measure_columns) + len(profile.dimension_columns)
        if feature_diversity > 10:
            score += 10
        elif feature_diversity > 5:
            score += 7
        else:
            score += 3
        
        return min(100.0, score)
    
    def _generate_strategic_recommendations(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        if profile.ml_readiness_score and profile.ml_readiness_score > 70:
            recommendations.append("🎯 Prioritize for AI/ML initiatives")
        
        if profile.data_quality_score < 0.8:
            recommendations.append("🔧 Invest in data quality improvements")
        
        if len(profile.temporal_columns) > 0:
            recommendations.append("📈 Leverage for predictive analytics")
        
        if profile.row_count > 50000:
            recommendations.append("💼 Consider for advanced analytics programs")
        
        if not recommendations:
            recommendations.append("📊 Conduct detailed feasibility analysis")
        
        return recommendations
    
    # Placeholder methods for various assessments (to be implemented based on specific needs)
    def _assess_business_value(self, profile: 'EnhancedTableProfile') -> str:
        """Assess business value of the dataset"""
        if profile.row_count > 100000 and profile.data_quality_score > 0.8:
            return "High"
        elif profile.row_count > 10000 and profile.data_quality_score > 0.6:
            return "Medium"
        else:
            return "Moderate"
    
    # Additional helper methods would be implemented here for all the various assessments
    # This is a comprehensive framework that can be extended with specific business logic
    
    def _assess_growth_pattern(self, profile) -> str:
        return "Steady" if profile.row_count > 10000 else "Emerging"
    
    def _assess_access_patterns(self, profile) -> str:
        return "Mixed read/write" if profile.temporal_columns else "Primarily read"
    
    def _assess_query_complexity(self, profile) -> str:
        complexity_score = len(profile.measure_columns) + len(profile.dimension_columns)
        return "High" if complexity_score > 10 else "Medium" if complexity_score > 5 else "Low"
    
    def _assess_realtime_needs(self, profile) -> str:
        return "Yes" if profile.temporal_columns else "No"
    
    def _assess_analytics_load(self, profile) -> str:
        return "High" if profile.row_count > 100000 else "Medium"
    
    def _assess_reliability_needs(self, profile) -> str:
        return "High" if profile.data_quality_score > 0.9 else "Medium"
    
    def _assess_compliance_needs(self, profile) -> str:
        return "High" if 'customer' in (profile.business_domain or '').lower() else "Medium"
    
    def _has_compliance_implications(self, profile) -> bool:
        """Check if dataset has compliance implications"""
        business_domain = (profile.business_domain or '').lower()
        return any(indicator in business_domain for indicator in ['customer', 'financial', 'personal', 'payment'])
    
    def _has_strategic_value(self, profile) -> bool:
        """Check if dataset has strategic value"""
        return profile.row_count > 50000 and profile.data_quality_score > 0.7
    
    # Additional methods would continue here...