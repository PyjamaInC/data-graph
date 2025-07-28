"""
Contextual Metadata Enricher

This module enriches metadata with contextual intelligence, business context,
and strategic insights based on profiling data and business domain knowledge.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BusinessCriticalityLevel(Enum):
    """Business criticality levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ComplianceRiskLevel(Enum):
    """Compliance risk levels"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


@dataclass
class BusinessContext:
    """Business context information"""
    criticality_level: BusinessCriticalityLevel
    compliance_implications: Dict[str, Any]
    lineage_insights: Dict[str, Any]
    usage_recommendations: List[str]
    stakeholder_impact: Dict[str, Any]
    integration_opportunities: List[Dict[str, Any]]
    competitive_advantage: Optional[Dict[str, Any]] = None
    regulatory_considerations: Optional[List[str]] = None


@dataclass
class EnrichedMetadata:
    """Metadata enriched with business context"""
    original_metadata: Dict[str, Any]
    business_context: BusinessContext
    strategic_assessment: Dict[str, Any]
    operational_impact: Dict[str, Any]
    investment_guidance: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    enrichment_timestamp: str
    confidence_score: float


class ContextualMetadataEnricher:
    """Enrich metadata with contextual intelligence"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Business domain indicators
        self.financial_indicators = [
            'price', 'amount', 'cost', 'revenue', 'payment', 'value', 'profit',
            'expense', 'budget', 'invoice', 'transaction', 'balance', 'credit',
            'debit', 'fee', 'commission', 'salary', 'wage'
        ]
        
        self.customer_indicators = [
            'customer', 'client', 'user', 'buyer', 'subscriber', 'member',
            'account', 'contact', 'prospect', 'lead', 'visitor'
        ]
        
        self.product_indicators = [
            'product', 'item', 'sku', 'catalog', 'inventory', 'stock',
            'category', 'brand', 'model', 'variant'
        ]
        
        self.operational_indicators = [
            'order', 'delivery', 'shipment', 'logistics', 'warehouse',
            'supplier', 'vendor', 'process', 'workflow', 'operation'
        ]
        
        self.personal_data_indicators = [
            'name', 'email', 'phone', 'address', 'ssn', 'id', 'birth',
            'gender', 'age', 'location', 'zip', 'postal'
        ]
    
    def enrich_with_business_context(self, 
                                   metadata: Dict[str, Any], 
                                   profile: 'EnhancedTableProfile') -> EnrichedMetadata:
        """Add business context to metadata"""
        
        self.logger.info(f"Enriching metadata with business context for {profile.table_name}")
        
        try:
            # Determine business criticality
            business_criticality = self._assess_business_criticality(profile)
            
            # Identify data lineage patterns
            lineage_insights = self._infer_data_lineage(profile)
            
            # Assess regulatory compliance implications
            compliance_assessment = self._assess_compliance_implications(profile)
            
            # Generate usage recommendations
            usage_recommendations = self._generate_contextual_usage_recommendations(profile)
            
            # Assess stakeholder impact
            stakeholder_impact = self._assess_stakeholder_impact(profile)
            
            # Identify integration opportunities
            integration_opportunities = self._identify_integration_opportunities(profile)
            
            # Assess competitive advantage
            competitive_advantage = self._assess_competitive_advantage(profile)
            
            # Generate regulatory considerations
            regulatory_considerations = self._generate_regulatory_considerations(profile)
            
            # Create business context
            business_context = BusinessContext(
                criticality_level=business_criticality,
                compliance_implications=compliance_assessment,
                lineage_insights=lineage_insights,
                usage_recommendations=usage_recommendations,
                stakeholder_impact=stakeholder_impact,
                integration_opportunities=integration_opportunities,
                competitive_advantage=competitive_advantage,
                regulatory_considerations=regulatory_considerations
            )
            
            # Generate strategic assessment
            strategic_assessment = self._generate_strategic_assessment(profile, business_context)
            
            # Assess operational impact
            operational_impact = self._assess_operational_impact(profile, business_context)
            
            # Generate investment guidance
            investment_guidance = self._generate_investment_guidance(profile, business_context)
            
            # Assess overall risk
            risk_assessment = self._generate_risk_assessment(profile, business_context)
            
            # Calculate enrichment confidence
            confidence_score = self._calculate_enrichment_confidence(profile, business_context)
            
            return EnrichedMetadata(
                original_metadata=metadata,
                business_context=business_context,
                strategic_assessment=strategic_assessment,
                operational_impact=operational_impact,
                investment_guidance=investment_guidance,
                risk_assessment=risk_assessment,
                enrichment_timestamp=self._get_timestamp(),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error enriching metadata: {e}")
            return self._generate_fallback_enriched_metadata(metadata, profile)
    
    def _assess_business_criticality(self, profile: 'EnhancedTableProfile') -> BusinessCriticalityLevel:
        """Assess business criticality based on profiling data"""
        
        criticality_score = 0
        
        # High volume data is typically more critical
        if profile.row_count > 1000000:
            criticality_score += 4
        elif profile.row_count > 100000:
            criticality_score += 3
        elif profile.row_count > 10000:
            criticality_score += 2
        else:
            criticality_score += 1
        
        # Financial data is business critical
        if self._contains_financial_data(profile):
            criticality_score += 4
        
        # Customer data is typically critical
        if self._contains_customer_data(profile):
            criticality_score += 3
        
        # High quality data is more business-critical
        if profile.data_quality_score > 0.9:
            criticality_score += 2
        elif profile.data_quality_score > 0.8:
            criticality_score += 1
        
        # Temporal data for trend analysis
        if profile.temporal_columns:
            criticality_score += 2
        
        # ML readiness indicates strategic value
        if profile.ml_readiness_score and profile.ml_readiness_score > 80:
            criticality_score += 2
        
        # Map score to criticality level
        if criticality_score >= 12:
            return BusinessCriticalityLevel.CRITICAL
        elif criticality_score >= 8:
            return BusinessCriticalityLevel.HIGH
        elif criticality_score >= 5:
            return BusinessCriticalityLevel.MEDIUM
        else:
            return BusinessCriticalityLevel.LOW
    
    def _assess_compliance_implications(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Assess regulatory compliance implications"""
        
        compliance_risks = []
        applicable_regulations = []
        
        # Check for personal data
        if self._contains_personal_data(profile):
            compliance_risks.append("Personal data protection required")
            applicable_regulations.extend(["GDPR", "CCPA", "PIPEDA"])
        
        # Check for financial data
        if self._contains_financial_data(profile):
            compliance_risks.append("Financial data security requirements")
            applicable_regulations.extend(["SOX", "PCI-DSS", "Basel III"])
        
        # Check for healthcare data indicators
        if self._contains_healthcare_data(profile):
            compliance_risks.append("Healthcare data protection required")
            applicable_regulations.append("HIPAA")
        
        # Assess data quality compliance risk
        if profile.data_quality_score < 0.8:
            compliance_risks.append("Poor data quality may impact compliance reporting")
        
        # Determine overall compliance risk level
        risk_level = self._calculate_compliance_risk_level(compliance_risks, profile)
        
        return {
            'risk_level': risk_level.value,
            'applicable_regulations': applicable_regulations,
            'compliance_risks': compliance_risks,
            'data_protection_requirements': self._identify_data_protection_requirements(profile),
            'audit_recommendations': self._generate_audit_recommendations(profile),
            'retention_requirements': self._identify_retention_requirements(profile)
        }
    
    def _infer_data_lineage(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Infer data lineage patterns from profiling"""
        
        lineage_indicators = {
            'source_system_type': self._infer_source_system_type(profile),
            'data_freshness_pattern': self._infer_data_freshness(profile),
            'update_frequency': self._infer_update_frequency(profile),
            'data_flow_pattern': self._infer_data_flow_pattern(profile),
            'transformation_complexity': self._assess_transformation_complexity(profile),
            'upstream_dependencies': self._identify_upstream_dependencies(profile),
            'downstream_impact': self._assess_downstream_impact(profile)
        }
        
        return lineage_indicators
    
    def _generate_contextual_usage_recommendations(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Generate usage recommendations based on business context"""
        
        recommendations = []
        
        # Data quality recommendations
        if profile.data_quality_score < 0.8:
            recommendations.append("🔧 Implement comprehensive data quality monitoring before business-critical use")
        
        # Volume-based recommendations
        if profile.row_count > 1000000:
            recommendations.append("⚡ Consider data partitioning and performance optimization for large-scale analytics")
        
        # ML recommendations based on business context
        if profile.ml_readiness_score and profile.ml_readiness_score > 70:
            if self._contains_customer_data(profile):
                recommendations.append("🎯 Leverage for customer analytics and personalization initiatives")
            if self._contains_financial_data(profile):
                recommendations.append("💰 Implement fraud detection and financial forecasting models")
            if self._contains_operational_data(profile):
                recommendations.append("📊 Deploy operational optimization and predictive maintenance")
        
        # Temporal data recommendations
        if profile.temporal_columns:
            recommendations.append("📈 Implement trend analysis and forecasting capabilities")
            recommendations.append("⏰ Consider real-time analytics and alerting systems")
        
        # Correlation-based recommendations
        if profile.correlation_analysis:
            redundant_count = len(profile.correlation_analysis.get('feature_redundancy', []))
            if redundant_count > 0:
                recommendations.append("🔄 Review and consolidate redundant features to improve analysis efficiency")
        
        # Business domain specific recommendations
        business_domain = (profile.business_domain or '').lower()
        if 'customer' in business_domain:
            recommendations.append("👥 Implement customer segmentation and lifetime value analysis")
        elif 'product' in business_domain:
            recommendations.append("📦 Enable product performance analytics and recommendation systems")
        elif 'order' in business_domain:
            recommendations.append("🛒 Deploy order optimization and demand forecasting")
        
        # Compliance recommendations
        if self._contains_personal_data(profile):
            recommendations.append("🔒 Implement privacy-preserving analytics and data anonymization")
        
        return recommendations
    
    def _assess_stakeholder_impact(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Assess impact on different stakeholders"""
        
        stakeholder_impact = {
            'executives': self._assess_executive_impact(profile),
            'data_analysts': self._assess_analyst_impact(profile),
            'data_scientists': self._assess_data_scientist_impact(profile),
            'business_users': self._assess_business_user_impact(profile),
            'it_operations': self._assess_it_operations_impact(profile),
            'compliance_team': self._assess_compliance_team_impact(profile),
            'customers': self._assess_customer_impact(profile)
        }
        
        return stakeholder_impact
    
    def _identify_integration_opportunities(self, profile: 'EnhancedTableProfile') -> List[Dict[str, Any]]:
        """Identify integration opportunities with other systems"""
        
        opportunities = []
        
        # CRM integration opportunities
        if self._contains_customer_data(profile):
            opportunities.append({
                'type': 'CRM Integration',
                'description': 'Integrate with CRM systems for 360-degree customer view',
                'business_value': 'Enhanced customer insights and personalization',
                'complexity': 'Medium',
                'priority': 'High'
            })
        
        # ERP integration opportunities
        if self._contains_financial_data(profile) or self._contains_operational_data(profile):
            opportunities.append({
                'type': 'ERP Integration',
                'description': 'Connect with ERP systems for operational analytics',
                'business_value': 'Improved operational efficiency and cost optimization',
                'complexity': 'High',
                'priority': 'Medium'
            })
        
        # BI/Analytics platform integration
        if profile.row_count > 10000 and profile.data_quality_score > 0.7:
            opportunities.append({
                'type': 'BI Platform Integration',
                'description': 'Integrate with business intelligence and analytics platforms',
                'business_value': 'Self-service analytics and automated reporting',
                'complexity': 'Low',
                'priority': 'High'
            })
        
        # ML platform integration
        if profile.ml_readiness_score and profile.ml_readiness_score > 60:
            opportunities.append({
                'type': 'ML Platform Integration',
                'description': 'Connect with machine learning and AI platforms',
                'business_value': 'Automated insights and predictive capabilities',
                'complexity': 'Medium',
                'priority': 'High'
            })
        
        # Real-time streaming integration
        if profile.temporal_columns and profile.row_count > 100000:
            opportunities.append({
                'type': 'Streaming Analytics',
                'description': 'Implement real-time data streaming and processing',
                'business_value': 'Real-time insights and immediate response capabilities',
                'complexity': 'High',
                'priority': 'Medium'
            })
        
        return opportunities
    
    def _assess_competitive_advantage(self, profile: 'EnhancedTableProfile') -> Optional[Dict[str, Any]]:
        """Assess competitive advantage potential"""
        
        # Only assess if data has strategic value
        if not self._has_strategic_value(profile):
            return None
        
        advantage_factors = []
        uniqueness_score = 0
        
        # Large, high-quality datasets provide competitive advantage
        if profile.row_count > 100000 and profile.data_quality_score > 0.8:
            advantage_factors.append("Large, high-quality dataset enables superior analytics")
            uniqueness_score += 3
        
        # Temporal data provides forecasting advantage
        if profile.temporal_columns and profile.row_count > 50000:
            advantage_factors.append("Historical data depth enables predictive capabilities")
            uniqueness_score += 2
        
        # Customer behavior data is highly valuable
        if self._contains_customer_data(profile) and len(profile.measure_columns) > 3:
            advantage_factors.append("Rich customer behavior data enables personalization")
            uniqueness_score += 3
        
        # Real-time operational data provides operational advantage
        if self._contains_operational_data(profile) and profile.temporal_columns:
            advantage_factors.append("Operational data enables process optimization")
            uniqueness_score += 2
        
        # High ML readiness enables AI-driven competitive advantage
        if profile.ml_readiness_score and profile.ml_readiness_score > 80:
            advantage_factors.append("ML-ready data enables AI-driven competitive advantage")
            uniqueness_score += 2
        
        if not advantage_factors:
            return None
        
        return {
            'advantage_factors': advantage_factors,
            'uniqueness_score': uniqueness_score,
            'competitive_moat_potential': 'High' if uniqueness_score >= 6 else 'Medium' if uniqueness_score >= 3 else 'Low',
            'monetization_opportunities': self._identify_monetization_opportunities(profile),
            'strategic_value_proposition': self._generate_strategic_value_proposition(profile, advantage_factors)
        }
    
    def _generate_regulatory_considerations(self, profile: 'EnhancedTableProfile') -> List[str]:
        """Generate regulatory considerations"""
        
        considerations = []
        
        if self._contains_personal_data(profile):
            considerations.append("Ensure GDPR/CCPA compliance for personal data processing")
            considerations.append("Implement data subject rights (access, rectification, erasure)")
            considerations.append("Conduct privacy impact assessments for new use cases")
        
        if self._contains_financial_data(profile):
            considerations.append("Maintain SOX compliance for financial reporting")
            considerations.append("Implement proper access controls for financial data")
            considerations.append("Ensure audit trail for all financial data modifications")
        
        if profile.data_quality_score < 0.8:
            considerations.append("Address data quality issues for regulatory reporting accuracy")
        
        if profile.row_count > 1000000:
            considerations.append("Consider data governance framework for large datasets")
        
        return considerations
    
    def _generate_strategic_assessment(self, 
                                     profile: 'EnhancedTableProfile', 
                                     business_context: BusinessContext) -> Dict[str, Any]:
        """Generate strategic assessment"""
        
        return {
            'strategic_value': self._calculate_strategic_value(profile, business_context),
            'investment_priority': self._calculate_investment_priority(profile, business_context),
            'business_alignment': self._assess_business_alignment(profile, business_context),
            'market_opportunity': self._assess_market_opportunity(profile, business_context),
            'innovation_potential': self._assess_innovation_potential(profile, business_context),
            'strategic_recommendations': self._generate_strategic_recommendations(profile, business_context)
        }
    
    def _assess_operational_impact(self, 
                                 profile: 'EnhancedTableProfile', 
                                 business_context: BusinessContext) -> Dict[str, Any]:
        """Assess operational impact"""
        
        return {
            'process_optimization_potential': self._assess_process_optimization_potential(profile),
            'automation_opportunities': self._identify_automation_opportunities(profile),
            'efficiency_gains': self._estimate_efficiency_gains(profile),
            'operational_risks': self._identify_operational_risks(profile, business_context),
            'resource_requirements': self._estimate_operational_resources(profile),
            'change_management_needs': self._assess_change_management_needs(profile, business_context)
        }
    
    def _generate_investment_guidance(self, 
                                    profile: 'EnhancedTableProfile', 
                                    business_context: BusinessContext) -> Dict[str, Any]:
        """Generate investment guidance"""
        
        return {
            'investment_categories': self._categorize_investments(profile, business_context),
            'budget_recommendations': self._recommend_budget_allocation(profile, business_context),
            'roi_expectations': self._set_roi_expectations(profile, business_context),
            'payback_timeline': self._estimate_payback_timeline(profile, business_context),
            'risk_adjusted_returns': self._calculate_risk_adjusted_returns(profile, business_context),
            'funding_strategy': self._recommend_funding_strategy(profile, business_context)
        }
    
    def _generate_risk_assessment(self, 
                                profile: 'EnhancedTableProfile', 
                                business_context: BusinessContext) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        
        return {
            'data_quality_risks': self._assess_data_quality_risks(profile),
            'compliance_risks': self._assess_compliance_risks(profile, business_context),
            'operational_risks': self._assess_operational_risks(profile, business_context),
            'strategic_risks': self._assess_strategic_risks(profile, business_context),
            'technical_risks': self._assess_technical_risks(profile),
            'mitigation_strategies': self._recommend_risk_mitigation(profile, business_context),
            'risk_monitoring_framework': self._design_risk_monitoring(profile, business_context)
        }
    
    def _calculate_enrichment_confidence(self, 
                                       profile: 'EnhancedTableProfile', 
                                       business_context: BusinessContext) -> float:
        """Calculate confidence in enrichment analysis"""
        
        confidence = 0.0
        
        # Base confidence from data characteristics
        if profile.row_count > 10000:
            confidence += 25
        elif profile.row_count > 1000:
            confidence += 15
        else:
            confidence += 5
        
        # Quality confidence
        confidence += profile.data_quality_score * 30
        
        # Business context clarity
        if business_context.criticality_level in [BusinessCriticalityLevel.CRITICAL, BusinessCriticalityLevel.HIGH]:
            confidence += 20
        
        # Domain knowledge availability
        if profile.business_domain:
            confidence += 15
        
        # Profiling completeness
        if profile.quality_profile:
            confidence += 10
        
        return min(100.0, confidence)
    
    # Helper methods for business domain detection
    def _contains_financial_data(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if profile contains financial data indicators"""
        all_columns = (profile.measure_columns + profile.dimension_columns + 
                      profile.identifier_columns + profile.temporal_columns)
        all_text = ' '.join(all_columns + [profile.business_domain or '', profile.table_name]).lower()
        
        return any(indicator in all_text for indicator in self.financial_indicators)
    
    def _contains_customer_data(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if profile contains customer data indicators"""
        all_columns = (profile.measure_columns + profile.dimension_columns + 
                      profile.identifier_columns + profile.temporal_columns)
        all_text = ' '.join(all_columns + [profile.business_domain or '', profile.table_name]).lower()
        
        return any(indicator in all_text for indicator in self.customer_indicators)
    
    def _contains_product_data(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if profile contains product data indicators"""
        all_columns = (profile.measure_columns + profile.dimension_columns + 
                      profile.identifier_columns + profile.temporal_columns)
        all_text = ' '.join(all_columns + [profile.business_domain or '', profile.table_name]).lower()
        
        return any(indicator in all_text for indicator in self.product_indicators)
    
    def _contains_operational_data(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if profile contains operational data indicators"""
        all_columns = (profile.measure_columns + profile.dimension_columns + 
                      profile.identifier_columns + profile.temporal_columns)
        all_text = ' '.join(all_columns + [profile.business_domain or '', profile.table_name]).lower()
        
        return any(indicator in all_text for indicator in self.operational_indicators)
    
    def _contains_personal_data(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if profile contains personal data indicators"""
        all_columns = (profile.measure_columns + profile.dimension_columns + 
                      profile.identifier_columns + profile.temporal_columns)
        all_text = ' '.join(all_columns).lower()
        
        return any(indicator in all_text for indicator in self.personal_data_indicators)
    
    def _contains_healthcare_data(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if profile contains healthcare data indicators"""
        healthcare_indicators = ['health', 'medical', 'patient', 'diagnosis', 'treatment', 'hospital', 'clinic']
        all_text = ' '.join([profile.business_domain or '', profile.table_name]).lower()
        
        return any(indicator in all_text for indicator in healthcare_indicators)
    
    def _has_strategic_value(self, profile: 'EnhancedTableProfile') -> bool:
        """Check if dataset has strategic value"""
        return (profile.row_count > 50000 and 
                profile.data_quality_score > 0.7 and 
                (self._contains_customer_data(profile) or 
                 self._contains_financial_data(profile) or 
                 (profile.ml_readiness_score and profile.ml_readiness_score > 70)))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _generate_fallback_enriched_metadata(self, 
                                           metadata: Dict[str, Any], 
                                           profile: 'EnhancedTableProfile') -> EnrichedMetadata:
        """Generate fallback enriched metadata when full enrichment fails"""
        
        # Create minimal business context
        business_context = BusinessContext(
            criticality_level=BusinessCriticalityLevel.MEDIUM,
            compliance_implications={'status': 'Assessment needed'},
            lineage_insights={'status': 'Analysis needed'},
            usage_recommendations=['Conduct detailed analysis'],
            stakeholder_impact={'status': 'Assessment needed'},
            integration_opportunities=[]
        )
        
        return EnrichedMetadata(
            original_metadata=metadata,
            business_context=business_context,
            strategic_assessment={'status': 'Basic assessment available'},
            operational_impact={'status': 'Basic assessment available'},
            investment_guidance={'status': 'Basic guidance available'},
            risk_assessment={'status': 'Basic risk assessment available'},
            enrichment_timestamp=self._get_timestamp(),
            confidence_score=30.0
        )
    
    # Placeholder methods for various assessments
    # These would be implemented with specific business logic
    
    def _calculate_compliance_risk_level(self, risks: List[str], profile) -> ComplianceRiskLevel:
        """Calculate compliance risk level"""
        if len(risks) >= 3:
            return ComplianceRiskLevel.VERY_HIGH
        elif len(risks) >= 2:
            return ComplianceRiskLevel.HIGH
        elif len(risks) >= 1:
            return ComplianceRiskLevel.MEDIUM
        else:
            return ComplianceRiskLevel.LOW
    
    def _infer_source_system_type(self, profile) -> str:
        """Infer source system type"""
        if self._contains_customer_data(profile):
            return "CRM/Customer System"
        elif self._contains_financial_data(profile):
            return "Financial/ERP System"
        elif self._contains_operational_data(profile):
            return "Operational System"
        else:
            return "Generic Data System"
    
    def _infer_data_freshness(self, profile) -> str:
        """Infer data freshness pattern"""
        if profile.temporal_columns:
            return "Regular updates detected"
        else:
            return "Static or infrequent updates"
    
    def _infer_update_frequency(self, profile) -> str:
        """Infer update frequency"""
        if profile.row_count > 1000000:
            return "High frequency (likely daily or real-time)"
        elif profile.row_count > 100000:
            return "Medium frequency (likely weekly)"
        else:
            return "Low frequency (monthly or less)"
    
    # Additional helper methods would continue here with specific business logic
    # for each assessment function...