"""
Intelligent Data Catalog

This module creates natural language data catalogs using Ollama and rich profiling data.
Transforms technical profiling results into human-friendly, business-oriented descriptions.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from .enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer

logger = logging.getLogger(__name__)


@dataclass
class CatalogEntry:
    """Intelligent catalog entry with business-friendly content"""
    title: str
    description: str
    usage_guide: str
    data_lineage_story: str
    quality_badge: Dict[str, Any]
    recommended_queries: List[Dict[str, str]]
    business_glossary: List[Dict[str, str]]
    technical_summary: Dict[str, Any]
    last_updated: str
    confidence_score: float


class IntelligentDataCatalog:
    """Generate natural language data catalog using Ollama"""
    
    def __init__(self, enhanced_summarizer: 'EnhancedLLMSemanticSummarizer'):
        self.summarizer = enhanced_summarizer
        self.logger = logging.getLogger(__name__)
    
    def generate_catalog_entry(self, profile: 'EnhancedTableProfile') -> CatalogEntry:
        """Generate comprehensive catalog entry"""
        
        self.logger.info(f"Generating catalog entry for {profile.table_name}")
        
        try:
            # Generate user-friendly title
            title = self._generate_friendly_title(profile)
            
            # Use existing rich metadata generation
            rich_metadata = self.summarizer.generate_rich_metadata(
                profile.table_name, profile
            )
            
            # Generate additional catalog-specific content
            usage_guide = self._generate_usage_guide(profile)
            lineage_story = self._generate_lineage_story(profile)
            quality_badge = self._create_quality_badge(profile)
            recommended_queries = self._suggest_starter_queries(profile)
            business_glossary = self._extract_business_terms(profile)
            technical_summary = self._create_technical_summary(profile)
            
            return CatalogEntry(
                title=title,
                description=rich_metadata.business_summary,
                usage_guide=usage_guide,
                data_lineage_story=lineage_story,
                quality_badge=quality_badge,
                recommended_queries=recommended_queries,
                business_glossary=business_glossary,
                technical_summary=technical_summary,
                last_updated=datetime.now().isoformat(),
                confidence_score=self._calculate_catalog_confidence(profile)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating catalog entry: {e}")
            return self._generate_fallback_catalog_entry(profile)
    
    def _generate_friendly_title(self, profile: 'EnhancedTableProfile') -> str:
        """Generate human-friendly table title"""
        
        prompt = f"""
        Create a business-friendly title for this data table:
        
        Table: {profile.table_name}
        Business Domain: {profile.business_domain or 'General'}
        Contains: {profile.row_count:,} records
        Key Measures: {', '.join(profile.measure_columns[:3]) if profile.measure_columns else 'None'}
        Key Dimensions: {', '.join(profile.dimension_columns[:3]) if profile.dimension_columns else 'None'}
        
        Generate a clear, descriptive title that business users would understand.
        Examples: "Customer Purchase History", "Product Performance Metrics", "Order Processing Data"
        
        Return only the title, nothing else.
        """
        
        try:
            title = self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=50).strip()
            # Clean up the title
            title = title.replace('"', '').replace("'", '').strip()
            if not title or len(title) < 5:
                return self._generate_fallback_title(profile)
            return title
        except Exception as e:
            self.logger.warning(f"Failed to generate friendly title: {e}")
            return self._generate_fallback_title(profile)
    
    def _generate_usage_guide(self, profile: 'EnhancedTableProfile') -> str:
        """Generate usage guide for the table"""
        
        prompt = f"""
        Create a practical usage guide for this data table:
        
        Table: {profile.table_name}
        Records: {profile.row_count:,}
        Data Quality: {profile.data_quality_score * 100:.1f}%
        ML Readiness: {profile.ml_readiness_score or 0:.1f}%
        Business Domain: {profile.business_domain or 'General'}
        
        Key Features:
        - Measures: {', '.join(profile.measure_columns[:5]) if profile.measure_columns else 'None'}
        - Dimensions: {', '.join(profile.dimension_columns[:5]) if profile.dimension_columns else 'None'}
        - Identifiers: {', '.join(profile.identifier_columns[:3]) if profile.identifier_columns else 'None'}
        - Temporal: {', '.join(profile.temporal_columns) if profile.temporal_columns else 'None'}
        
        Create a comprehensive usage guide including:
        
        **Who Should Use This Data:**
        - Specific roles and departments that would benefit
        - Required expertise levels and training needs
        - Decision-making authorities who should have access
        
        **Common Use Cases:**
        - Daily operational tasks and reporting needs
        - Strategic analysis and planning applications  
        - Compliance and audit requirements
        - Performance monitoring and KPI tracking
        
        **Best Practices:**
        - Data refresh and update frequency recommendations
        - Quality monitoring and validation procedures
        - Integration patterns with other systems
        - Security and privacy considerations
        
        **Implementation Guidance:**
        - Technical requirements and infrastructure needs
        - Training and onboarding recommendations
        - Success metrics and evaluation criteria
        - Troubleshooting common issues
        
        **Business Value Realization:**
        - Expected ROI and value creation opportunities
        - Key performance indicators to track
        - Success stories and use case examples
        
        Write in clear, actionable language for business users.
        """
        
        try:
            return self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=1200)  # 🚀 INCREASED from 300 to 1200!
        except Exception as e:
            self.logger.warning(f"Failed to generate usage guide: {e}")
            return self._generate_fallback_usage_guide(profile)
    
    def _generate_lineage_story(self, profile: 'EnhancedTableProfile') -> str:
        """Generate data lineage story"""
        
        prompt = f"""
        Tell the story of where this data comes from and how it's used:
        
        Table: {profile.table_name}
        Business Context: {profile.business_domain or 'General business operations'}
        Data Volume: {profile.row_count:,} records
        Update Pattern: {'Regular updates' if profile.temporal_columns else 'Static or infrequent updates'}
        
        Create a brief story (2-3 sentences) explaining:
        1. Where this data likely originates (source systems)
        2. How it flows through the organization
        3. Who depends on it for decisions
        
        Write in narrative style that non-technical users can understand.
        """
        
        try:
            return self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=200)
        except Exception as e:
            self.logger.warning(f"Failed to generate lineage story: {e}")
            return self._generate_fallback_lineage_story(profile)
    
    def _create_quality_badge(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Create quality badge with visual indicators"""
        
        quality_score = profile.data_quality_score * 100
        
        if quality_score >= 90:
            badge_level = "Excellent"
            badge_color = "green"
            badge_icon = "✅"
        elif quality_score >= 80:
            badge_level = "Good"
            badge_color = "blue"
            badge_icon = "👍"
        elif quality_score >= 70:
            badge_level = "Fair"
            badge_color = "yellow"
            badge_icon = "⚠️"
        else:
            badge_level = "Needs Attention"
            badge_color = "red"
            badge_icon = "🔴"
        
        # Add specific quality insights
        quality_insights = []
        if profile.quality_profile:
            if len(profile.quality_profile.critical_alerts) > 0:
                quality_insights.append("Critical issues detected")
            if len(profile.quality_profile.warning_alerts) > 0:
                quality_insights.append("Some data quality concerns")
            if not quality_insights:
                quality_insights.append("No major quality issues")
        else:
            quality_insights.append("Basic quality assessment")
        
        return {
            'score': quality_score,
            'level': badge_level,
            'color': badge_color,
            'icon': badge_icon,
            'insights': quality_insights,
            'recommendation': self._get_quality_recommendation(quality_score)
        }
    
    def _suggest_starter_queries(self, profile: 'EnhancedTableProfile') -> List[Dict[str, str]]:
        """Suggest starter queries for exploration"""
        
        queries = []
        
        # Basic overview query
        if profile.measure_columns and profile.dimension_columns:
            queries.append({
                'title': 'Data Overview',
                'description': 'Get a high-level summary of the data',
                'sql': f'''SELECT 
    COUNT(*) as total_records,
    {', '.join([f'COUNT(DISTINCT {col}) as unique_{col}' for col in profile.dimension_columns[:2]])}
FROM {profile.table_name}''',
                'business_value': 'Understand data volume and diversity'
            })
        
        # Trend analysis if temporal data exists
        if profile.temporal_columns and profile.measure_columns:
            time_col = profile.temporal_columns[0]
            measure_col = profile.measure_columns[0]
            queries.append({
                'title': 'Trend Analysis',
                'description': 'Analyze trends over time',
                'sql': f'''SELECT 
    DATE_TRUNC('month', {time_col}) as period,
    COUNT(*) as record_count,
    AVG({measure_col}) as avg_{measure_col}
FROM {profile.table_name}
WHERE {time_col} >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', {time_col})
ORDER BY period''',
                'business_value': 'Track performance trends and seasonality'
            })
        
        # Top performers analysis
        if profile.dimension_columns and profile.measure_columns:
            dim_col = profile.dimension_columns[0]
            measure_col = profile.measure_columns[0]
            queries.append({
                'title': 'Top Performers',
                'description': f'Find top {dim_col} by {measure_col}',
                'sql': f'''SELECT 
    {dim_col},
    COUNT(*) as frequency,
    SUM({measure_col}) as total_{measure_col},
    AVG({measure_col}) as avg_{measure_col}
FROM {profile.table_name}
GROUP BY {dim_col}
ORDER BY total_{measure_col} DESC
LIMIT 10''',
                'business_value': 'Identify top performers and opportunities'
            })
        
        # Data quality check
        null_checks = [f'SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as null_{col}'
                       for col in (profile.measure_columns + profile.dimension_columns)[:5]]
        if null_checks:
            queries.append({
                'title': 'Data Quality Check',
                'description': 'Check for missing values and data completeness',
                'sql': f'''SELECT 
    COUNT(*) as total_records,
    {', '.join(null_checks)}
FROM {profile.table_name}''',
                'business_value': 'Assess data completeness and reliability'
            })
        
        return queries[:4]  # Return top 4 queries
    
    def _extract_business_terms(self, profile: 'EnhancedTableProfile') -> List[Dict[str, str]]:
        """Extract and explain business terms from the data"""
        
        terms = []
        
        # Extract from table name
        table_term = self._create_term_definition(profile.table_name, "table", profile)
        if table_term:
            terms.append(table_term)
        
        # Extract from key columns
        key_columns = (profile.measure_columns[:3] + 
                      profile.dimension_columns[:3] + 
                      profile.identifier_columns[:2])
        
        for column in key_columns:
            term_def = self._create_term_definition(column, "column", profile)
            if term_def:
                terms.append(term_def)
        
        return terms[:8]  # Limit to 8 terms
    
    def _create_term_definition(self, term: str, term_type: str, profile: 'EnhancedTableProfile') -> Optional[Dict[str, str]]:
        """Create definition for a business term"""
        
        prompt = f"""
        Define this business term in simple language:
        
        Term: {term}
        Context: {term_type} in {profile.table_name} table
        Business Domain: {profile.business_domain or 'General business'}
        
        Provide a brief, clear definition (1-2 sentences) that a business user would understand.
        Focus on what it means for business operations.
        
        Definition:
        """
        
        try:
            definition = self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=100)
            definition = definition.strip().replace('Definition:', '').strip()
            
            if definition and len(definition) > 10:
                return {
                    'term': term,
                    'definition': definition,
                    'type': term_type
                }
        except Exception as e:
            self.logger.warning(f"Failed to generate term definition for {term}: {e}")
        
        return None
    
    def _create_technical_summary(self, profile: 'EnhancedTableProfile') -> Dict[str, Any]:
        """Create technical summary for advanced users"""
        
        return {
            'table_name': profile.table_name,
            'row_count': profile.row_count,
            'column_count': profile.column_count,
            'data_types': {
                'measures': len(profile.measure_columns),
                'dimensions': len(profile.dimension_columns),
                'identifiers': len(profile.identifier_columns),
                'temporal': len(profile.temporal_columns)
            },
            'quality_metrics': {
                'overall_score': profile.data_quality_score * 100,
                'ml_readiness': profile.ml_readiness_score or 0,
                'has_quality_profile': profile.quality_profile is not None,
                'has_correlation_analysis': profile.correlation_analysis is not None,
                'has_outlier_analysis': profile.outlier_analysis is not None
            },
            'business_classification': {
                'domain': profile.business_domain,
                'table_type': profile.table_type,
                'key_concepts': profile.key_concepts[:5] if profile.key_concepts else []
            }
        }
    
    def _calculate_catalog_confidence(self, profile: 'EnhancedTableProfile') -> float:
        """Calculate confidence score for catalog entry"""
        
        confidence = 0.0
        
        # Data volume confidence
        if profile.row_count > 10000:
            confidence += 25
        elif profile.row_count > 1000:
            confidence += 15
        else:
            confidence += 5
        
        # Quality confidence
        confidence += profile.data_quality_score * 30
        
        # Business context confidence
        if profile.business_domain:
            confidence += 15
        
        # Column classification confidence
        if profile.measure_columns or profile.dimension_columns:
            confidence += 20
        
        # Advanced analysis confidence
        if profile.quality_profile:
            confidence += 10
        
        return min(100.0, confidence)
    
    # Fallback methods
    def _generate_fallback_title(self, profile: 'EnhancedTableProfile') -> str:
        """Generate fallback title when LLM fails"""
        name_parts = profile.table_name.replace('_', ' ').title()
        if profile.business_domain:
            return f"{profile.business_domain.title()} {name_parts}"
        return name_parts
    
    def _generate_fallback_usage_guide(self, profile: 'EnhancedTableProfile') -> str:
        """Generate fallback usage guide"""
        guide_parts = [
            f"This table contains {profile.row_count:,} records",
            f"with {profile.column_count} columns."
        ]
        
        if profile.measure_columns:
            guide_parts.append(f"Key metrics include {', '.join(profile.measure_columns[:3])}.")
        
        if profile.dimension_columns:
            guide_parts.append(f"Can be analyzed by {', '.join(profile.dimension_columns[:3])}.")
        
        if profile.data_quality_score > 0.8:
            guide_parts.append("Data quality is good for reliable analysis.")
        else:
            guide_parts.append("Review data quality before using for critical analysis.")
        
        return ' '.join(guide_parts)
    
    def _generate_fallback_lineage_story(self, profile: 'EnhancedTableProfile') -> str:
        """Generate fallback lineage story"""
        if profile.temporal_columns:
            return f"This data is regularly updated and contains historical records. It supports {profile.business_domain or 'business'} operations and decision-making."
        else:
            return f"This data provides reference information for {profile.business_domain or 'business'} operations and may be updated periodically."
    
    def _generate_fallback_catalog_entry(self, profile: 'EnhancedTableProfile') -> CatalogEntry:
        """Generate fallback catalog entry when full generation fails"""
        
        return CatalogEntry(
            title=self._generate_fallback_title(profile),
            description=f"Data table containing {profile.row_count:,} records for {profile.business_domain or 'business'} analysis.",
            usage_guide=self._generate_fallback_usage_guide(profile),
            data_lineage_story=self._generate_fallback_lineage_story(profile),
            quality_badge=self._create_quality_badge(profile),
            recommended_queries=[],
            business_glossary=[],
            technical_summary=self._create_technical_summary(profile),
            last_updated=datetime.now().isoformat(),
            confidence_score=40.0
        )
    
    def _get_quality_recommendation(self, quality_score: float) -> str:
        """Get quality recommendation based on score"""
        if quality_score >= 90:
            return "Excellent quality - suitable for all analytics use cases"
        elif quality_score >= 80:
            return "Good quality - suitable for most business analysis"
        elif quality_score >= 70:
            return "Fair quality - review data before critical analysis"
        else:
            return "Quality needs improvement - implement data quality measures"
    
    def generate_catalog_summary(self, catalog_entries: List[CatalogEntry]) -> Dict[str, Any]:
        """Generate summary of the entire catalog"""
        
        total_tables = len(catalog_entries)
        total_records = sum(entry.technical_summary['row_count'] for entry in catalog_entries)
        avg_quality = sum(entry.quality_badge['score'] for entry in catalog_entries) / total_tables if total_tables > 0 else 0
        
        # Count tables by quality level
        quality_distribution = {}
        for entry in catalog_entries:
            level = entry.quality_badge['level']
            quality_distribution[level] = quality_distribution.get(level, 0) + 1
        
        # Identify business domains
        domains = set()
        for entry in catalog_entries:
            domain = entry.technical_summary.get('business_classification', {}).get('domain')
            if domain:
                domains.add(domain)
        
        return {
            'catalog_overview': {
                'total_tables': total_tables,
                'total_records': total_records,
                'average_quality_score': avg_quality,
                'business_domains': list(domains),
                'quality_distribution': quality_distribution
            },
            'high_value_tables': [
                entry for entry in catalog_entries 
                if entry.quality_badge['score'] >= 80 and entry.technical_summary['row_count'] >= 10000
            ][:5],
            'tables_needing_attention': [
                entry for entry in catalog_entries 
                if entry.quality_badge['score'] < 70
            ][:5]
        }