"""
Auto Documentation Generator

This module generates comprehensive, living documentation that updates automatically
using profiling insights and Ollama's natural language capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
import json

if TYPE_CHECKING:
    from .enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentationSection:
    """Documentation section structure"""
    title: str
    content: str
    section_type: str
    metadata: Dict[str, Any]
    last_updated: str


@dataclass
class DocumentationPackage:
    """Complete documentation package"""
    overview: str
    sections: List[DocumentationSection]
    table_of_contents: List[Dict[str, str]]
    executive_summary: str
    technical_appendix: str
    generated_timestamp: str
    confidence_score: float


class AutoDocumentationGenerator:
    """Generate living documentation using profiling insights"""
    
    def __init__(self, enhanced_summarizer: 'EnhancedLLMSemanticSummarizer'):
        self.summarizer = enhanced_summarizer
        self.logger = logging.getLogger(__name__)
    
    def generate_data_dictionary(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate comprehensive data dictionary"""
        
        self.logger.info(f"Generating data dictionary for {len(profiles)} tables")
        
        try:
            # Generate overview
            overview = self._generate_system_overview(profiles)
            
            # Generate table sections
            sections = []
            for profile in profiles:
                section = self._generate_table_section(profile)
                sections.append(section)
            
            # Generate cross-table analysis
            cross_analysis = self._generate_cross_table_analysis(profiles)
            
            # Compile full documentation
            full_doc = self._compile_data_dictionary(overview, sections, cross_analysis)
            
            return full_doc
            
        except Exception as e:
            self.logger.error(f"Error generating data dictionary: {e}")
            return self._generate_fallback_dictionary(profiles)
    
    def generate_comprehensive_documentation(self, 
                                           profiles: List['EnhancedTableProfile'],
                                           include_technical: bool = True,
                                           include_business: bool = True) -> DocumentationPackage:
        """Generate comprehensive documentation package"""
        
        self.logger.info(f"Generating comprehensive documentation for {len(profiles)} tables")
        
        try:
            # Generate executive summary
            exec_summary = self._generate_executive_summary(profiles)
            
            # Generate system overview
            overview = self._generate_detailed_overview(profiles)
            
            # Generate documentation sections
            sections = []
            
            if include_business:
                sections.extend(self._generate_business_sections(profiles))
            
            if include_technical:
                sections.extend(self._generate_technical_sections(profiles))
            
            # Generate table of contents
            toc = self._generate_table_of_contents(sections)
            
            # Generate technical appendix
            tech_appendix = self._generate_technical_appendix(profiles) if include_technical else ""
            
            # Calculate confidence
            confidence = self._calculate_documentation_confidence(profiles, sections)
            
            return DocumentationPackage(
                overview=overview,
                sections=sections,
                table_of_contents=toc,
                executive_summary=exec_summary,
                technical_appendix=tech_appendix,
                generated_timestamp=datetime.now().isoformat(),
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive documentation: {e}")
            return self._generate_fallback_documentation(profiles)
    
    def _generate_system_overview(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate overview of the data system"""
        
        # Calculate system metrics
        total_records = sum(p.row_count for p in profiles)
        total_columns = sum(p.column_count for p in profiles)
        avg_quality = sum(p.data_quality_score for p in profiles) / len(profiles) if profiles else 0
        business_domains = list(set(p.business_domain for p in profiles if p.business_domain))
        
        prompt = f"""
        Create an executive overview for this data system documentation:
        
        SYSTEM METRICS:
        - Tables analyzed: {len(profiles)}
        - Total records: {total_records:,}
        - Total columns: {total_columns}
        - Average data quality: {avg_quality * 100:.1f}%
        - Business domains: {', '.join(business_domains) if business_domains else 'General'}
        
        TABLE BREAKDOWN:
        {self._format_table_summary(profiles)}
        
        Write a 2-3 paragraph overview explaining:
        1. What this data system contains and its business purpose
        2. Key insights about data quality and analytics potential
        3. Strategic value and recommended use cases
        
        Use executive-friendly language that highlights business value.
        """
        
        try:
            return self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=400)
        except Exception as e:
            self.logger.warning(f"Failed to generate system overview: {e}")
            return self._generate_fallback_overview(profiles)
    
    def _generate_table_section(self, profile: 'EnhancedTableProfile') -> str:
        """Generate documentation section for one table"""
        
        try:
            # Get rich metadata
            rich_metadata = self.summarizer.generate_rich_metadata(
                profile.table_name, profile
            )
            
            # Generate detailed table documentation
            table_prompt = f"""
            Create comprehensive documentation for this data table:
            
            TABLE: {profile.table_name}
            BUSINESS CONTEXT: {profile.business_domain or 'General'}
            RECORDS: {profile.row_count:,}
            QUALITY SCORE: {profile.data_quality_score * 100:.1f}%
            
            COLUMNS:
            - Measures ({len(profile.measure_columns)}): {', '.join(profile.measure_columns[:5])}
            - Dimensions ({len(profile.dimension_columns)}): {', '.join(profile.dimension_columns[:5])}
            - Identifiers ({len(profile.identifier_columns)}): {', '.join(profile.identifier_columns[:3])}
            - Temporal ({len(profile.temporal_columns)}): {', '.join(profile.temporal_columns)}
            
            Write detailed documentation including:
            1. Business purpose and use cases
            2. Data structure explanation
            3. Key relationships and dependencies
            4. Quality assessment and recommendations
            5. Usage guidelines and best practices
            
            Make it comprehensive but accessible to both technical and business users.
            """
            
            detailed_content = self.summarizer.base_summarizer.generate_summary(table_prompt, max_tokens=600)
            
            return f"""
### {profile.table_name}

{detailed_content}

**Technical Summary:**
- **Records:** {profile.row_count:,}
- **Data Quality:** {profile.data_quality_score * 100:.1f}%
- **ML Readiness:** {profile.ml_readiness_score or 0:.1f}%
- **Business Domain:** {profile.business_domain or 'General'}

**Column Classification:**
- 📊 **Measures ({len(profile.measure_columns)}):** {', '.join(profile.measure_columns[:5])}
- 🏷️ **Dimensions ({len(profile.dimension_columns)}):** {', '.join(profile.dimension_columns[:5])}
- 🔑 **Identifiers ({len(profile.identifier_columns)}):** {', '.join(profile.identifier_columns[:3])}
- 📅 **Temporal ({len(profile.temporal_columns)}):** {', '.join(profile.temporal_columns)}

**Usage Recommendations:**
{chr(10).join('- ' + rec for rec in (rich_metadata.usage_recommendations or ['General analysis use']))}

**Key Insights:**
- {rich_metadata.business_summary if rich_metadata.business_summary else 'Business analysis available'}

---
            """.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate table section for {profile.table_name}: {e}")
            return self._generate_fallback_table_section(profile)
    
    def _generate_cross_table_analysis(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate cross-table analysis section"""
        
        if len(profiles) < 2:
            return ""
        
        # Identify potential relationships
        relationships = self._identify_table_relationships(profiles)
        
        prompt = f"""
        Analyze relationships and integration opportunities across these data tables:
        
        TABLES: {', '.join(p.table_name for p in profiles)}
        
        POTENTIAL RELATIONSHIPS:
        {chr(10).join(relationships)}
        
        BUSINESS DOMAINS: {', '.join(set(p.business_domain for p in profiles if p.business_domain))}
        
        Create analysis covering:
        1. Data integration opportunities
        2. Cross-table business insights
        3. Recommended join strategies
        4. Analytics use cases spanning multiple tables
        5. Data governance considerations
        
        Focus on business value and practical implementation.
        """
        
        try:
            analysis = self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=500)
            return f"\n## Cross-Table Analysis\n\n{analysis}\n"
        except Exception as e:
            self.logger.warning(f"Failed to generate cross-table analysis: {e}")
            return ""
    
    def _generate_executive_summary(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate executive summary"""
        
        # Calculate key metrics
        total_value = sum(p.row_count for p in profiles)
        high_quality_tables = [p for p in profiles if p.data_quality_score > 0.8]
        ml_ready_tables = [p for p in profiles if (p.ml_readiness_score or 0) > 70]
        
        prompt = f"""
        Create an executive summary for this data portfolio:
        
        PORTFOLIO OVERVIEW:
        - Total Tables: {len(profiles)}
        - Total Records: {total_value:,}
        - High Quality Tables: {len(high_quality_tables)}/{len(profiles)}
        - ML-Ready Tables: {len(ml_ready_tables)}/{len(profiles)}
        
        KEY ASSETS:
        {chr(10).join(f'- {p.table_name}: {p.row_count:,} records, {p.data_quality_score*100:.1f}% quality' for p in profiles[:5])}
        
        Write an executive summary covering:
        1. Strategic value of the data portfolio
        2. Key opportunities for business impact
        3. Investment priorities and recommendations
        4. Risk assessment and mitigation needs
        
        Target audience: C-level executives and senior management.
        """
        
        try:
            return self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=400)
        except Exception as e:
            self.logger.warning(f"Failed to generate executive summary: {e}")
            return self._generate_fallback_executive_summary(profiles)
    
    def _generate_detailed_overview(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate detailed system overview"""
        
        system_stats = self._calculate_system_statistics(profiles)
        
        prompt = f"""
        Create a detailed overview of this data system:
        
        SYSTEM STATISTICS:
        {json.dumps(system_stats, indent=2)}
        
        QUALITY DISTRIBUTION:
        - Excellent (90%+): {sum(1 for p in profiles if p.data_quality_score >= 0.9)} tables
        - Good (80-89%): {sum(1 for p in profiles if 0.8 <= p.data_quality_score < 0.9)} tables
        - Fair (70-79%): {sum(1 for p in profiles if 0.7 <= p.data_quality_score < 0.8)} tables
        - Needs Work (<70%): {sum(1 for p in profiles if p.data_quality_score < 0.7)} tables
        
        Create comprehensive overview including:
        1. System architecture and data flow
        2. Business domain coverage
        3. Data quality assessment
        4. Analytics and ML capabilities
        5. Integration and scalability considerations
        
        Balance technical detail with business context.
        """
        
        try:
            return self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=600)
        except Exception as e:
            self.logger.warning(f"Failed to generate detailed overview: {e}")
            return self._generate_fallback_detailed_overview(profiles)
    
    def _generate_business_sections(self, profiles: List['EnhancedTableProfile']) -> List[DocumentationSection]:
        """Generate business-focused documentation sections"""
        
        sections = []
        
        # Business value section
        business_value = self._generate_business_value_section(profiles)
        sections.append(DocumentationSection(
            title="Business Value Assessment",
            content=business_value,
            section_type="business",
            metadata={"focus": "value_proposition"},
            last_updated=datetime.now().isoformat()
        ))
        
        # Use cases section
        use_cases = self._generate_use_cases_section(profiles)
        sections.append(DocumentationSection(
            title="Analytics Use Cases",
            content=use_cases,
            section_type="business",
            metadata={"focus": "use_cases"},
            last_updated=datetime.now().isoformat()
        ))
        
        # Data governance section
        governance = self._generate_governance_section(profiles)
        sections.append(DocumentationSection(
            title="Data Governance Guidelines",
            content=governance,
            section_type="business",
            metadata={"focus": "governance"},
            last_updated=datetime.now().isoformat()
        ))
        
        return sections
    
    def _generate_technical_sections(self, profiles: List['EnhancedTableProfile']) -> List[DocumentationSection]:
        """Generate technical documentation sections"""
        
        sections = []
        
        # Architecture section
        architecture = self._generate_architecture_section(profiles)
        sections.append(DocumentationSection(
            title="Data Architecture",
            content=architecture,
            section_type="technical",
            metadata={"focus": "architecture"},
            last_updated=datetime.now().isoformat()
        ))
        
        # Integration section
        integration = self._generate_integration_section(profiles)
        sections.append(DocumentationSection(
            title="Integration Guidelines",
            content=integration,
            section_type="technical",
            metadata={"focus": "integration"},
            last_updated=datetime.now().isoformat()
        ))
        
        # Quality monitoring section
        monitoring = self._generate_monitoring_section(profiles)
        sections.append(DocumentationSection(
            title="Quality Monitoring",
            content=monitoring,
            section_type="technical",
            metadata={"focus": "monitoring"},
            last_updated=datetime.now().isoformat()
        ))
        
        return sections
    
    def _compile_data_dictionary(self, overview: str, sections: List[str], cross_analysis: str) -> str:
        """Compile complete data dictionary"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""
# Data System Documentation

## Executive Overview
{overview}

## Table Details
{chr(10).join(sections)}

{cross_analysis}

## Appendix

### Generation Details
- **Generated on:** {timestamp}
- **Method:** Enhanced profiling with LLM analysis
- **Tables Analyzed:** {len(sections)}

### Data Quality Summary
This documentation is based on comprehensive data profiling and quality analysis. 
Quality scores reflect completeness, consistency, and analytical readiness.

### Usage Guidelines
- Use quality scores to assess analytical reliability
- Review data lineage before making business-critical decisions  
- Consider ML readiness scores for advanced analytics projects
- Follow governance guidelines for data access and usage

---
*This documentation is automatically generated and should be updated regularly as data evolves.*
        """.strip()
    
    # Helper methods for generating specific content
    def _format_table_summary(self, profiles: List['EnhancedTableProfile']) -> str:
        """Format table summary for overview"""
        summaries = []
        for profile in profiles[:5]:  # Top 5 tables
            summaries.append(f"- {profile.table_name}: {profile.row_count:,} records, "
                           f"{profile.data_quality_score*100:.1f}% quality, "
                           f"{profile.business_domain or 'General'}")
        if len(profiles) > 5:
            summaries.append(f"- ... and {len(profiles) - 5} more tables")
        return '\n'.join(summaries)
    
    def _identify_table_relationships(self, profiles: List['EnhancedTableProfile']) -> List[str]:
        """Identify potential relationships between tables"""
        relationships = []
        
        for i, profile1 in enumerate(profiles):
            for profile2 in profiles[i+1:]:
                # Check for common identifier patterns
                common_ids = set(profile1.identifier_columns) & set(profile2.identifier_columns)
                if common_ids:
                    relationships.append(f"{profile1.table_name} ↔ {profile2.table_name}: "
                                       f"Potential join on {', '.join(list(common_ids)[:2])}")
                
                # Check for similar business domains
                if (profile1.business_domain and profile2.business_domain and 
                    profile1.business_domain == profile2.business_domain):
                    relationships.append(f"{profile1.table_name} ↔ {profile2.table_name}: "
                                       f"Related business domain ({profile1.business_domain})")
        
        return relationships[:5]  # Limit to top 5
    
    def _calculate_system_statistics(self, profiles: List['EnhancedTableProfile']) -> Dict[str, Any]:
        """Calculate comprehensive system statistics"""
        
        return {
            "total_tables": len(profiles),
            "total_records": sum(p.row_count for p in profiles),
            "total_columns": sum(p.column_count for p in profiles),
            "average_quality": sum(p.data_quality_score for p in profiles) / len(profiles) if profiles else 0,
            "business_domains": list(set(p.business_domain for p in profiles if p.business_domain)),
            "largest_table": max(profiles, key=lambda p: p.row_count).table_name if profiles else None,
            "highest_quality": max(profiles, key=lambda p: p.data_quality_score).table_name if profiles else None,
            "ml_ready_count": sum(1 for p in profiles if (p.ml_readiness_score or 0) > 70),
            "temporal_tables": sum(1 for p in profiles if p.temporal_columns),
            "total_measures": sum(len(p.measure_columns) for p in profiles),
            "total_dimensions": sum(len(p.dimension_columns) for p in profiles)
        }
    
    def _calculate_documentation_confidence(self, 
                                          profiles: List['EnhancedTableProfile'], 
                                          sections: List[DocumentationSection]) -> float:
        """Calculate confidence in documentation quality"""
        
        if not profiles:
            return 0.0
        
        confidence = 0.0
        
        # Data completeness factor (40%)
        avg_quality = sum(p.data_quality_score for p in profiles) / len(profiles)
        confidence += avg_quality * 40
        
        # Volume factor (20%)
        total_records = sum(p.row_count for p in profiles)
        if total_records > 1000000:
            confidence += 20
        elif total_records > 100000:
            confidence += 15
        elif total_records > 10000:
            confidence += 10
        else:
            confidence += 5
        
        # Analysis depth factor (20%)
        advanced_profiles = sum(1 for p in profiles if p.quality_profile and p.correlation_analysis)
        if advanced_profiles == len(profiles):
            confidence += 20
        elif advanced_profiles > len(profiles) / 2:
            confidence += 15
        else:
            confidence += 10
        
        # Section completeness factor (20%)
        confidence += min(20, len(sections) * 3)
        
        return min(100.0, confidence)
    
    # Fallback methods
    def _generate_fallback_overview(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate fallback overview when LLM fails"""
        total_records = sum(p.row_count for p in profiles)
        avg_quality = sum(p.data_quality_score for p in profiles) / len(profiles) if profiles else 0
        
        return f"""This data system contains {len(profiles)} tables with {total_records:,} total records. 
        The average data quality score is {avg_quality*100:.1f}%. The system supports various 
        business analytics and reporting needs."""
    
    def _generate_fallback_table_section(self, profile: 'EnhancedTableProfile') -> str:
        """Generate fallback table section"""
        return f"""
### {profile.table_name}

**Overview:** Contains {profile.row_count:,} records with {profile.column_count} columns.

**Data Quality:** {profile.data_quality_score * 100:.1f}%

**Key Features:**
- Measures: {', '.join(profile.measure_columns[:3]) if profile.measure_columns else 'None'}
- Dimensions: {', '.join(profile.dimension_columns[:3]) if profile.dimension_columns else 'None'}
- Business Domain: {profile.business_domain or 'General'}

---
        """
    
    def _generate_fallback_dictionary(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate fallback dictionary when full generation fails"""
        sections = []
        for profile in profiles:
            sections.append(self._generate_fallback_table_section(profile))
        
        return f"""
# Data Dictionary

## Overview
Documentation for {len(profiles)} data tables.

{chr(10).join(sections)}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
    
    def _generate_fallback_documentation(self, profiles: List['EnhancedTableProfile']) -> DocumentationPackage:
        """Generate fallback documentation package"""
        
        overview = self._generate_fallback_overview(profiles)
        exec_summary = self._generate_fallback_executive_summary(profiles)
        
        sections = [
            DocumentationSection(
                title="Table Summary",
                content=f"Summary of {len(profiles)} tables",
                section_type="basic",
                metadata={},
                last_updated=datetime.now().isoformat()
            )
        ]
        
        return DocumentationPackage(
            overview=overview,
            sections=sections,
            table_of_contents=[{"title": "Table Summary", "section": "basic"}],
            executive_summary=exec_summary,
            technical_appendix="",
            generated_timestamp=datetime.now().isoformat(),
            confidence_score=30.0
        )
    
    def _generate_fallback_executive_summary(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate fallback executive summary"""
        return f"Data portfolio contains {len(profiles)} tables with varying quality levels. Review recommended for strategic planning."
    
    def _generate_fallback_detailed_overview(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate fallback detailed overview"""
        return f"Detailed analysis of {len(profiles)} data tables. Further analysis recommended for comprehensive insights."
    
    # Placeholder methods for specialized sections
    def _generate_business_value_section(self, profiles: List['EnhancedTableProfile']) -> str:
        return "Business value assessment based on data volume, quality, and strategic importance."
    
    def _generate_use_cases_section(self, profiles: List['EnhancedTableProfile']) -> str:
        return "Analytics use cases including reporting, trend analysis, and predictive modeling."
    
    def _generate_governance_section(self, profiles: List['EnhancedTableProfile']) -> str:
        return "Data governance guidelines including access controls, quality standards, and usage policies."
    
    def _generate_architecture_section(self, profiles: List['EnhancedTableProfile']) -> str:
        return "Technical architecture including storage, processing, and integration patterns."
    
    def _generate_integration_section(self, profiles: List['EnhancedTableProfile']) -> str:
        return "Integration guidelines for connecting with external systems and data pipelines."
    
    def _generate_monitoring_section(self, profiles: List['EnhancedTableProfile']) -> str:
        return "Quality monitoring framework including metrics, alerts, and remediation procedures."
    
    def _generate_technical_appendix(self, profiles: List['EnhancedTableProfile']) -> str:
        """Generate technical appendix"""
        return f"Technical details for {len(profiles)} tables including schemas, relationships, and performance characteristics."
    
    def _generate_table_of_contents(self, sections: List[DocumentationSection]) -> List[Dict[str, str]]:
        """Generate table of contents"""
        return [{"title": section.title, "type": section.section_type} for section in sections]