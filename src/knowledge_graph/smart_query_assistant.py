"""
Smart Query Assistant

This module provides intelligent query assistance using profiling context and Ollama.
It can answer business questions about data and suggest appropriate SQL queries.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import re

if TYPE_CHECKING:
    from .enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Response from the query assistant"""
    answer: str
    suggested_sql: Optional[str]
    relevant_tables: List[str]
    data_quality_warnings: List[str]
    suggested_followup_questions: List[str]
    confidence_score: float
    interpretation_notes: List[str]


class SmartQueryAssistant:
    """Intelligent query assistant using profiling context"""
    
    def __init__(self, enhanced_summarizer: 'EnhancedLLMSemanticSummarizer'):
        self.summarizer = enhanced_summarizer
        self.table_profiles = {}  # Store profiles for context
        self.logger = logging.getLogger(__name__)
        
        # Business domain keywords for table detection
        self.domain_keywords = {
            'customer': ['customer', 'client', 'user', 'buyer', 'subscriber', 'member'],
            'order': ['order', 'purchase', 'transaction', 'sale', 'booking'],
            'product': ['product', 'item', 'sku', 'catalog', 'inventory'],
            'payment': ['payment', 'billing', 'invoice', 'fee', 'charge'],
            'shipping': ['shipping', 'delivery', 'logistics', 'shipment'],
            'seller': ['seller', 'vendor', 'supplier', 'merchant'],
            'review': ['review', 'rating', 'feedback', 'comment']
        }
    
    def register_table(self, profile: 'EnhancedTableProfile'):
        """Register a table profile for query assistance"""
        self.table_profiles[profile.table_name] = profile
        self.logger.info(f"Registered table {profile.table_name} for query assistance")
    
    def answer_data_question(self, 
                           question: str, 
                           context_tables: Optional[List[str]] = None) -> QueryResponse:
        """Answer business questions about the data"""
        
        self.logger.info(f"Answering question: {question}")
        
        try:
            # Get relevant table context
            if context_tables:
                relevant_profiles = [
                    self.table_profiles[t] for t in context_tables 
                    if t in self.table_profiles
                ]
            else:
                # Auto-detect relevant tables based on question
                relevant_profiles = self._detect_relevant_tables(question)
            
            if not relevant_profiles:
                return self._generate_no_data_response(question)
            
            # Build context for the question
            context = self._build_query_context(relevant_profiles, question)
            
            # Generate comprehensive answer
            answer_response = self._generate_comprehensive_answer(question, context, relevant_profiles)
            
            # Extract components from response
            answer_parts = self._parse_answer_response(answer_response)
            
            # Generate additional insights
            quality_warnings = self._get_quality_warnings(relevant_profiles)
            followup_questions = self._suggest_followup_questions(question, relevant_profiles)
            interpretation_notes = self._generate_interpretation_notes(relevant_profiles)
            
            # Calculate confidence
            confidence_score = self._calculate_response_confidence(question, relevant_profiles)
            
            return QueryResponse(
                answer=answer_parts.get('answer', answer_response),
                suggested_sql=answer_parts.get('sql'),
                relevant_tables=[p.table_name for p in relevant_profiles],
                data_quality_warnings=quality_warnings,
                suggested_followup_questions=followup_questions,
                confidence_score=confidence_score,
                interpretation_notes=interpretation_notes
            )
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return self._generate_fallback_response(question)
    
    def _detect_relevant_tables(self, question: str) -> List['EnhancedTableProfile']:
        """Auto-detect relevant tables based on question content"""
        
        question_lower = question.lower()
        relevant_profiles = []
        table_scores = {}
        
        # Score tables based on keyword matching
        for table_name, profile in self.table_profiles.items():
            score = 0
            
            # Direct table name matching
            if table_name.lower() in question_lower:
                score += 10
            
            # Business domain matching
            domain = (profile.business_domain or '').lower()
            for domain_key, keywords in self.domain_keywords.items():
                if domain_key in domain:
                    for keyword in keywords:
                        if keyword in question_lower:
                            score += 5
                            break
            
            # Column name matching
            all_columns = (profile.measure_columns + profile.dimension_columns + 
                          profile.identifier_columns + profile.temporal_columns)
            for column in all_columns:
                if column.lower() in question_lower:
                    score += 3
            
            # Key concept matching
            for concept in (profile.key_concepts or []):
                if concept.lower() in question_lower:
                    score += 2
            
            if score > 0:
                table_scores[table_name] = score
        
        # Select top scoring tables (max 3)
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        for table_name, score in sorted_tables[:3]:
            relevant_profiles.append(self.table_profiles[table_name])
        
        # If no tables found, try broader matching
        if not relevant_profiles and self.table_profiles:
            # Include the largest table as a fallback
            largest_table = max(self.table_profiles.values(), key=lambda p: p.row_count)
            relevant_profiles.append(largest_table)
        
        return relevant_profiles
    
    def _build_query_context(self, 
                           profiles: List['EnhancedTableProfile'], 
                           question: str) -> str:
        """Build context for query answering"""
        
        context_parts = []
        
        for profile in profiles:
            context_parts.append(f"""
Table: {profile.table_name}
- Records: {profile.row_count:,}
- Business Purpose: {profile.business_domain or 'General data'}
- Data Quality: {profile.data_quality_score * 100:.1f}%
- Key Measures: {', '.join(profile.measure_columns[:5]) if profile.measure_columns else 'None'}
- Key Dimensions: {', '.join(profile.dimension_columns[:5]) if profile.dimension_columns else 'None'}
- Identifiers: {', '.join(profile.identifier_columns[:3]) if profile.identifier_columns else 'None'}
- Time Columns: {', '.join(profile.temporal_columns) if profile.temporal_columns else 'None'}
            """.strip())
        
        return '\n\n'.join(context_parts)
    
    def _generate_comprehensive_answer(self, 
                                     question: str, 
                                     context: str, 
                                     profiles: List['EnhancedTableProfile']) -> str:
        """Generate comprehensive answer using enhanced context"""
        
        question_type = self._classify_question_type(question)
        
        prompt = f"""
        You are a senior data analyst helping business users understand their data.
        
        QUESTION: {question}
        QUESTION TYPE: {question_type}
        
        DATA CONTEXT:
        {context}
        
        Provide a comprehensive business-focused answer that includes:
        
        **DIRECT ANSWER**
        - Clear, specific response to the exact question asked
        - Key insights and findings based on available data
        - Quantitative details where possible
        
        **SQL QUERY**
        - Production-ready SQL query to answer the question
        - Include proper table joins, filters, and aggregations
        - Add comments explaining complex logic
        - Optimize for performance and readability
        
        **BUSINESS INSIGHTS**
        - What the data reveals about business operations
        - Trends, patterns, and anomalies to investigate
        - Strategic implications and recommendations
        
        **IMPLEMENTATION GUIDANCE**
        - How to set up regular monitoring for this metric
        - Dashboard and visualization recommendations
        - Automation opportunities and data pipeline considerations
        
        **NEXT STEPS**
        - Follow-up questions to explore deeper insights
        - Additional data sources that could enhance analysis
        - Action items for business stakeholders
        
        Write in executive-friendly language while providing technical depth where appropriate.
        Focus on actionable insights that drive business value.
        """
        
        try:
            return self.summarizer.base_summarizer.generate_summary(prompt, max_tokens=1500)  # 🚀 INCREASED from 800 to 1500!
        except Exception as e:
            self.logger.warning(f"Failed to generate comprehensive answer: {e}")
            return self._generate_basic_answer(question, profiles)
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question being asked"""
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['trend', 'over time', 'month', 'year', 'season']):
            return "Trend Analysis"
        elif any(word in question_lower for word in ['top', 'best', 'highest', 'most', 'rank']):
            return "Top Performers"
        elif any(word in question_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return "Comparison"
        elif any(word in question_lower for word in ['why', 'reason', 'cause', 'factor']):
            return "Root Cause Analysis"
        elif any(word in question_lower for word in ['predict', 'forecast', 'future', 'expect']):
            return "Predictive"
        elif any(word in question_lower for word in ['count', 'how many', 'number of']):
            return "Count/Volume"
        elif any(word in question_lower for word in ['average', 'mean', 'median', 'typical']):
            return "Statistical Summary"
        elif any(word in question_lower for word in ['distribution', 'spread', 'range']):
            return "Distribution Analysis"
        else:
            return "General Inquiry"
    
    def _parse_answer_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract structured components"""
        
        parts = {}
        
        # Try to extract SQL query
        sql_patterns = [
            r'SQL QUERY[:\s]*\n(.*?)(?=\n\d+\.|$)',
            r'```sql\n(.*?)\n```',
            r'SELECT\s+.*?(?=\n\n|\n\d+\.|$)'
        ]
        
        for pattern in sql_patterns:
            sql_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if sql_match:
                parts['sql'] = sql_match.group(1).strip()
                break
        
        # Extract direct answer
        answer_patterns = [
            r'DIRECT ANSWER[:\s]*\n(.*?)(?=\n\d+\.|$)',
            r'1\.\s*DIRECT ANSWER[:\s]*\n(.*?)(?=\n\d+\.|$)'
        ]
        
        for pattern in answer_patterns:
            answer_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                parts['answer'] = answer_match.group(1).strip()
                break
        
        return parts
    
    def _get_quality_warnings(self, profiles: List['EnhancedTableProfile']) -> List[str]:
        """Get data quality warnings for the relevant tables"""
        
        warnings = []
        
        for profile in profiles:
            table_warnings = []
            
            # Quality score warnings
            if profile.data_quality_score < 0.7:
                table_warnings.append(f"Low data quality score ({profile.data_quality_score * 100:.1f}%)")
            
            # Specific quality issues
            if profile.quality_profile:
                if len(profile.quality_profile.critical_alerts) > 0:
                    table_warnings.append("Critical data quality issues detected")
                if len(profile.quality_profile.warning_alerts) > 0:
                    table_warnings.append("Data quality warnings present")
            
            # Volume warnings
            if profile.row_count < 1000:
                table_warnings.append("Limited data volume may affect analysis reliability")
            
            # Missing temporal data for trend questions
            if not profile.temporal_columns:
                table_warnings.append("No time-based columns for trend analysis")
            
            if table_warnings:
                warnings.append(f"{profile.table_name}: {'; '.join(table_warnings)}")
        
        return warnings
    
    def _suggest_followup_questions(self, 
                                  original_question: str, 
                                  profiles: List['EnhancedTableProfile']) -> List[str]:
        """Suggest relevant followup questions"""
        
        followups = []
        
        # Generate followups based on available data
        for profile in profiles:
            # Temporal followups
            if profile.temporal_columns:
                followups.append(f"How has this trend changed over the past year in {profile.table_name}?")
            
            # Dimensional breakdown followups
            if profile.dimension_columns:
                top_dim = profile.dimension_columns[0]
                followups.append(f"How does this vary by {top_dim}?")
            
            # Quality investigation followups
            if profile.data_quality_score < 0.8:
                followups.append(f"What data quality issues might be affecting {profile.table_name} analysis?")
        
        # General analytical followups
        if len(profiles) > 1:
            followups.append("How do these metrics correlate across different tables?")
        
        followups.append("What external factors might influence these patterns?")
        followups.append("What actions should we take based on these insights?")
        
        return followups[:5]  # Limit to 5 suggestions
    
    def _generate_interpretation_notes(self, profiles: List['EnhancedTableProfile']) -> List[str]:
        """Generate interpretation notes for the analysis"""
        
        notes = []
        
        for profile in profiles:
            # Data freshness notes
            if profile.temporal_columns:
                notes.append(f"Data in {profile.table_name} includes temporal information for trend analysis")
            else:
                notes.append(f"Data in {profile.table_name} is static - may not reflect recent changes")
            
            # Sample size notes
            if profile.row_count > 100000:
                notes.append(f"Large sample size in {profile.table_name} provides statistical reliability")
            elif profile.row_count < 1000:
                notes.append(f"Small sample size in {profile.table_name} may limit statistical significance")
            
            # ML readiness notes
            if profile.ml_readiness_score and profile.ml_readiness_score > 70:
                notes.append(f"Data in {profile.table_name} is suitable for advanced analytics and ML")
        
        return notes
    
    def _calculate_response_confidence(self, 
                                     question: str, 
                                     profiles: List['EnhancedTableProfile']) -> float:
        """Calculate confidence in the response"""
        
        confidence = 0.0
        
        # Base confidence from table relevance
        if profiles:
            confidence += 30
        
        # Quality confidence
        avg_quality = sum(p.data_quality_score for p in profiles) / len(profiles) if profiles else 0
        confidence += avg_quality * 30
        
        # Data volume confidence
        total_records = sum(p.row_count for p in profiles)
        if total_records > 100000:
            confidence += 25
        elif total_records > 10000:
            confidence += 15
        else:
            confidence += 5
        
        # Feature richness confidence
        total_features = sum(len(p.measure_columns) + len(p.dimension_columns) for p in profiles)
        if total_features > 10:
            confidence += 15
        elif total_features > 5:
            confidence += 10
        else:
            confidence += 5
        
        return min(100.0, confidence)
    
    def _generate_basic_answer(self, question: str, profiles: List['EnhancedTableProfile']) -> str:
        """Generate basic answer when LLM fails"""
        
        if not profiles:
            return f"I don't have access to data that can answer: {question}"
        
        table_names = [p.table_name for p in profiles]
        total_records = sum(p.row_count for p in profiles)
        
        return f"""Based on available data in {', '.join(table_names)} ({total_records:,} total records), 
I can help analyze this question. However, I need more specific information about what 
metrics or comparisons you're looking for to provide a detailed SQL query and analysis."""
    
    def _generate_no_data_response(self, question: str) -> QueryResponse:
        """Generate response when no relevant data is found"""
        
        return QueryResponse(
            answer=f"I don't have access to data that can directly answer: {question}. "
                  f"Available tables: {', '.join(self.table_profiles.keys())}",
            suggested_sql=None,
            relevant_tables=[],
            data_quality_warnings=[],
            suggested_followup_questions=[
                "What data sources do we have available?",
                "Can you help me understand what's in our data catalog?"
            ],
            confidence_score=0.0,
            interpretation_notes=["No relevant data sources identified for this question"]
        )
    
    def _generate_fallback_response(self, question: str) -> QueryResponse:
        """Generate fallback response when processing fails"""
        
        return QueryResponse(
            answer=f"I encountered an issue processing your question: {question}. "
                  "Please try rephrasing or asking a more specific question.",
            suggested_sql=None,
            relevant_tables=list(self.table_profiles.keys()),
            data_quality_warnings=[],
            suggested_followup_questions=[
                "What tables do we have available for analysis?",
                "Can you help me explore our data catalog?"
            ],
            confidence_score=25.0,
            interpretation_notes=["Response generated using fallback method"]
        )
    
    def get_available_capabilities(self) -> Dict[str, Any]:
        """Get summary of available query capabilities"""
        
        if not self.table_profiles:
            return {"status": "No data sources registered"}
        
        capabilities = {
            "registered_tables": len(self.table_profiles),
            "total_records": sum(p.row_count for p in self.table_profiles.values()),
            "business_domains": list(set(
                p.business_domain for p in self.table_profiles.values() 
                if p.business_domain
            )),
            "available_analyses": [],
            "sample_questions": []
        }
        
        # Determine available analyses
        has_temporal = any(p.temporal_columns for p in self.table_profiles.values())
        has_measures = any(p.measure_columns for p in self.table_profiles.values())
        has_dimensions = any(p.dimension_columns for p in self.table_profiles.values())
        
        if has_temporal and has_measures:
            capabilities["available_analyses"].append("Trend Analysis")
            capabilities["sample_questions"].append("How have sales trends changed over time?")
        
        if has_measures and has_dimensions:
            capabilities["available_analyses"].append("Performance Analysis")
            capabilities["sample_questions"].append("What are our top performing products?")
        
        if len(self.table_profiles) > 1:
            capabilities["available_analyses"].append("Cross-table Analysis")
            capabilities["sample_questions"].append("How do customer behaviors correlate with order patterns?")
        
        capabilities["available_analyses"].extend([
            "Data Quality Assessment",
            "Summary Statistics",
            "Distribution Analysis"
        ])
        
        capabilities["sample_questions"].extend([
            "What is the data quality status of our tables?",
            "What are the key statistics for our main datasets?"
        ])
        
        return capabilities