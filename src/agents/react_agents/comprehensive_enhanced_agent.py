"""
Comprehensive Enhanced Data Exploration ReAct Agent

This agent fully leverages the knowledge graph infrastructure including:
- Question-specific analysis strategies  
- Context preservation between iterations
- Intelligent result interpretation
- Business-relevant insight generation
- Real data schema understanding
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage

from .enhanced_data_exploration_agent import (
    EnhancedDataExplorationReActAgent, 
    IntelligentExplorationState
)
from .intelligence_driven_toolkit import IntelligenceDrivenToolkit
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from knowledge_graph.table_intelligence import EnhancedTableIntelligenceLayer


@dataclass
class ComprehensiveExplorationState(IntelligentExplorationState):
    """Enhanced exploration state with full intelligence integration"""
    
    # Intelligence toolkit integration
    intelligence_summary: Dict[str, Any] = field(default_factory=dict)
    analysis_plan: Dict[str, Any] = field(default_factory=dict)
    operation_suggestions: List[str] = field(default_factory=list)
    
    # Real data understanding
    schema_understanding: Dict[str, Any] = field(default_factory=dict)
    business_context: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced tracking
    operations_executed: List[Dict[str, Any]] = field(default_factory=list)
    insights_confidence: Dict[str, float] = field(default_factory=dict)


class ComprehensiveEnhancedAgent(EnhancedDataExplorationReActAgent):
    """Comprehensive agent with full knowledge graph intelligence integration"""
    
    def __init__(self, 
                 enhanced_intelligence: EnhancedTableIntelligenceLayer,
                 semantic_graph_builder: Any,
                 enhanced_summarizer: Optional[Any] = None,
                 llm_model: str = "gpt-4"):
        
        super().__init__(enhanced_intelligence, semantic_graph_builder, enhanced_summarizer, llm_model)
        
        # Initialize intelligence-driven toolkit
        self.intelligence_toolkit = IntelligenceDrivenToolkit(enhanced_intelligence)
        
        # Enhanced components
        self.schema_analyzer = enhanced_intelligence
        self.current_analysis_plan = None
        self._comprehensive_state: Optional[ComprehensiveExplorationState] = None
        
        # Initialize logger
        import logging
        self.logger = logging.getLogger(__name__)
    
    def explore_for_insights(self, user_question: str, 
                           tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive exploration using full intelligence stack"""
        
        print("🚀 Starting Comprehensive Intelligence-Driven Exploration")
        print("=" * 80)
        
        # Step 1: Enhanced Intelligence Profiling
        print("🧠 Phase 1: Comprehensive Intelligence Analysis...")
        enhanced_profiles = self._generate_comprehensive_profiles(tables)
        
        # Step 2: Register profiles with intelligence toolkit
        print("🔧 Phase 2: Configuring Intelligence Toolkit...")
        for name, profile in enhanced_profiles.items():
            self.intelligence_toolkit.register_table_profile(name, profile)
        
        # Step 3: Generate intelligence summaries and analysis plans
        print("📋 Phase 3: Creating Analysis Strategy...")
        intelligence_summaries = {}
        analysis_plans = {}
        
        for table_name in tables.keys():
            intelligence_summaries[table_name] = self.intelligence_toolkit.get_intelligence_summary(table_name)
            analysis_plans[table_name] = self.intelligence_toolkit.get_comprehensive_analysis_plan(table_name, user_question)
        
        # Step 4: Schema and Business Context Understanding
        print("🏗️ Phase 4: Schema and Business Context Analysis...")
        schema_understanding = self._analyze_schema_context(tables, enhanced_profiles)
        business_context = self._extract_business_context(enhanced_profiles, user_question)
        
        # Step 5: Initialize comprehensive exploration state
        self._comprehensive_state = ComprehensiveExplorationState(
            user_question=user_question,
            tables=tables,
            enhanced_profiles=enhanced_profiles,
            table_relationships=self._build_relationships_context(),
            intelligence_summary=intelligence_summaries,
            analysis_plan=analysis_plans,
            schema_understanding=schema_understanding,
            business_context=business_context
        )
        
        # Extract intelligence-driven information
        self._extract_comprehensive_intelligence_info()
        
        # Step 6: Execute intelligence-driven exploration
        print("🔍 Phase 5: Executing Intelligence-Driven Exploration...")
        result = self._execute_comprehensive_exploration()
        
        # Step 7: Generate comprehensive insights with intelligence context
        final_insights = self._generate_comprehensive_insights()
        
        return {
            'user_question': user_question,
            'exploration_summary': {
                'iterations_used': self._comprehensive_state.iteration_count,
                'confidence_level': self._comprehensive_state.confidence_level,
                'total_findings': len(self._comprehensive_state.current_findings),
                'intelligence_driven': True,
                'operations_executed': len(self._comprehensive_state.operations_executed)
            },
            'intelligence_context': {
                'profiles_generated': len(enhanced_profiles),
                'intelligence_summaries': intelligence_summaries,
                'analysis_plans': {k: len(v.get('analysis_phases', [])) for k, v in analysis_plans.items()},
                'schema_understanding': schema_understanding,
                'business_context': business_context
            },
            'insights': final_insights,
            'exploration_history': self._comprehensive_state.exploration_history,
            'recommendations': self._generate_intelligence_recommendations(),
            'operations_executed': self._comprehensive_state.operations_executed,
            'data_quality_summary': self._summarize_data_quality(enhanced_profiles)
        }
    
    def _generate_comprehensive_profiles(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive profiles using enhanced intelligence"""
        enhanced_profiles = {}
        
        for name, df in tables.items():
            print(f"  📊 Comprehensive profiling: {name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            try:
                # Use the comprehensive analysis method
                profile = self.enhanced_intelligence.analyze_table_comprehensive(name, df)
                enhanced_profiles[name] = profile
                
                # Log intelligence insights
                if hasattr(profile, 'key_insights') and profile.key_insights:
                    print(f"    💡 Key insights: {len(profile.key_insights)} discovered")
                
                if hasattr(profile, 'data_quality_score'):
                    print(f"    📈 Data quality: {profile.data_quality_score:.1%}")
                
                if hasattr(profile, 'ml_readiness_score') and profile.ml_readiness_score:
                    print(f"    🤖 ML readiness: {profile.ml_readiness_score:.0f}%")
                
            except Exception as e:
                print(f"    ⚠️ Fallback profiling for {name}: {e}")
                # Use the basic profile as fallback
                enhanced_profiles[name] = self._basic_profile(name, df)
        
        return enhanced_profiles
    
    def _analyze_schema_context(self, tables: Dict[str, pd.DataFrame], 
                              profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema context for better understanding"""
        
        schema_context = {
            'table_types': {},
            'relationships': [],
            'business_domains': {},
            'data_patterns': {},
            'integration_opportunities': []
        }
        
        for table_name, profile in profiles.items():
            # Table type classification
            schema_context['table_types'][table_name] = {
                'type': getattr(profile, 'table_type', 'unknown'),
                'measures': len(getattr(profile, 'measure_columns', [])),
                'dimensions': len(getattr(profile, 'dimension_columns', [])),
                'temporal': len(getattr(profile, 'temporal_columns', [])),
                'identifiers': len(getattr(profile, 'identifier_columns', []))
            }
            
            # Business domain
            domain = getattr(profile, 'business_domain', None)
            if domain:
                schema_context['business_domains'][table_name] = domain
            
            # Data patterns
            schema_context['data_patterns'][table_name] = {
                'row_count': len(tables[table_name]),
                'column_count': len(tables[table_name].columns),
                'data_quality': getattr(profile, 'data_quality_score', 0),
                'completeness': 1 - (tables[table_name].isnull().sum().sum() / 
                                   (len(tables[table_name]) * len(tables[table_name].columns)))
            }
        
        # Identify integration opportunities
        for table1_name in tables.keys():
            for table2_name in tables.keys():
                if table1_name != table2_name:
                    common_patterns = self._find_common_patterns(
                        profiles[table1_name], profiles[table2_name]
                    )
                    if common_patterns:
                        schema_context['integration_opportunities'].append({
                            'table1': table1_name,
                            'table2': table2_name,
                            'common_patterns': common_patterns
                        })
        
        return schema_context
    
    def _extract_business_context(self, profiles: Dict[str, Any], user_question: str) -> Dict[str, Any]:
        """Extract business context from profiles and question"""
        
        business_context = {
            'domain_focus': None,
            'analysis_type': self._classify_analysis_type(user_question),
            'business_metrics': [],
            'key_entities': [],
            'temporal_scope': None
        }
        
        # Extract business metrics and entities
        for table_name, profile in profiles.items():
            # Business metrics (measure columns)
            measures = getattr(profile, 'measure_columns', [])
            business_context['business_metrics'].extend([
                {'table': table_name, 'metric': col} for col in measures
            ])
            
            # Key entities (dimension columns)
            dimensions = getattr(profile, 'dimension_columns', [])
            business_context['key_entities'].extend([
                {'table': table_name, 'entity': col} for col in dimensions
            ])
            
            # Domain focus
            domain = getattr(profile, 'business_domain', None)
            if domain and not business_context['domain_focus']:
                business_context['domain_focus'] = domain
        
        # Temporal scope analysis
        temporal_info = self._analyze_temporal_scope(profiles)
        business_context['temporal_scope'] = temporal_info
        
        return business_context
    
    def _classify_analysis_type(self, user_question: str) -> str:
        """Classify the type of analysis requested"""
        question_lower = user_question.lower()
        
        if any(keyword in question_lower for keyword in ['trend', 'seasonal', 'time', 'temporal']):
            return 'temporal_analysis'
        elif any(keyword in question_lower for keyword in ['segment', 'group', 'category']):
            return 'segmentation_analysis'
        elif any(keyword in question_lower for keyword in ['correlation', 'relationship', 'factor']):
            return 'correlation_analysis'
        elif any(keyword in question_lower for keyword in ['outlier', 'anomaly', 'unusual']):
            return 'anomaly_detection'
        elif any(keyword in question_lower for keyword in ['quality', 'missing', 'complete']):
            return 'quality_assessment'
        else:
            return 'exploratory_analysis'
    
    def _analyze_temporal_scope(self, profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal scope across tables"""
        temporal_info = {
            'has_temporal_data': False,
            'temporal_columns': {},
            'date_ranges': {},
            'temporal_granularity': 'unknown'
        }
        
        for table_name, profile in profiles.items():
            temporal_cols = getattr(profile, 'temporal_columns', [])
            if temporal_cols:
                temporal_info['has_temporal_data'] = True
                temporal_info['temporal_columns'][table_name] = temporal_cols
                
                # Check temporal analysis if available
                temporal_analysis = getattr(profile, 'temporal_analysis', None)
                if temporal_analysis:
                    temporal_info['date_ranges'][table_name] = temporal_analysis
        
        return temporal_info
    
    def _extract_comprehensive_intelligence_info(self):
        """Extract comprehensive intelligence information"""
        state = self._comprehensive_state
        
        # Set the state for parent method - parent uses _current_exploration_state
        self._current_exploration_state = state
        
        # Add comprehensive intelligence insights
        for table_name, profile in state.enhanced_profiles.items():
            # Extract key insights
            if hasattr(profile, 'key_insights') and profile.key_insights:
                state.current_findings.extend([
                    f"Intelligence Insight for {table_name}: {insight}" 
                    for insight in profile.key_insights[:3]
                ])
            
            # Extract quality insights
            if hasattr(profile, 'quality_profile') and profile.quality_profile:
                quality_profile = profile.quality_profile
                if quality_profile.critical_alerts:
                    state.quality_concerns.append(
                        f"{table_name}: {len(quality_profile.critical_alerts)} critical data quality issues"
                    )
                
                state.current_findings.extend(quality_profile.quality_recommendations[:2])
            
            # Extract correlation insights
            if hasattr(profile, 'correlation_analysis') and profile.correlation_analysis:
                corr_analysis = profile.correlation_analysis
                linear_rels = corr_analysis.get('linear_relationships', {})
                strong_corr = linear_rels.get('strong_linear', [])
                if strong_corr:
                    state.current_findings.append(
                        f"{table_name}: {len(strong_corr)} strong correlations detected"
                    )
            
            # Extract outlier insights
            if hasattr(profile, 'outlier_analysis') and profile.outlier_analysis:
                outlier_analysis = profile.outlier_analysis
                high_impact = outlier_analysis.get('high_impact_outliers', [])
                if high_impact:
                    state.current_findings.append(
                        f"{table_name}: {len(high_impact)} high-impact outliers detected"
                    )
    
    def _execute_comprehensive_exploration(self) -> Dict[str, Any]:
        """Execute comprehensive exploration with intelligence guidance"""
        
        state = self._comprehensive_state
        print(f"🔍 Starting comprehensive exploration: '{state.user_question}'")
        
        # Create analysis strategy based on intelligence
        strategy = self._create_intelligence_strategy()
        
        # Execute exploration cycles with intelligence guidance
        while (state.iteration_count < state.max_iterations and 
               state.confidence_level < 0.85):  # Higher threshold for comprehensive agent
            
            state.iteration_count += 1
            print(f"\n🔄 Intelligence Cycle {state.iteration_count}")
            
            # Get intelligent operation suggestion
            operation_result = self._execute_intelligence_cycle(strategy)
            
            # Update confidence based on intelligence insights
            self._update_confidence_with_intelligence(operation_result)
            
            if operation_result.get('sufficient_insights', False):
                print("✅ Sufficient insights achieved through intelligence analysis")
                break
        
        return {"exploration_completed": True, "cycles": state.iteration_count}
    
    def _create_intelligence_strategy(self) -> Dict[str, Any]:
        """Create exploration strategy based on intelligence insights"""
        state = self._comprehensive_state
        
        strategy = {
            'approach': 'intelligence_driven',
            'primary_table': self._select_primary_table(),
            'focus_areas': [],
            'operation_sequence': [],
            'intelligence_guidance': {}
        }
        
        # Determine focus areas based on question and intelligence
        analysis_type = state.business_context['analysis_type']
        
        if analysis_type == 'temporal_analysis':
            if state.business_context['temporal_scope']['has_temporal_data']:
                strategy['focus_areas'].append('temporal_patterns')
            else:
                strategy['focus_areas'].append('temporal_limitation_explanation')
        
        elif analysis_type == 'correlation_analysis':
            strategy['focus_areas'].append('correlation_investigation')
        
        elif analysis_type == 'segmentation_analysis':
            strategy['focus_areas'].append('segment_analysis')
        
        elif analysis_type == 'anomaly_detection':
            strategy['focus_areas'].append('outlier_investigation')
        
        else:
            strategy['focus_areas'].extend(['data_overview', 'relationship_analysis'])
        
        # Add quality assessment if needed
        quality_issues = any(
            len(state.quality_concerns) > 0 for concerns in [state.quality_concerns]
        )
        if quality_issues:
            strategy['focus_areas'].insert(0, 'quality_assessment')
        
        return strategy
    
    def _select_primary_table(self) -> str:
        """Select primary table for analysis based on intelligence and question context"""
        state = self._comprehensive_state
        question_lower = state.user_question.lower()
        
        # For temporal questions, prioritize tables with temporal columns
        if any(keyword in question_lower for keyword in ['time', 'temporal', 'seasonal', 'trend', 'delivery', 'order_purchase', 'monthly', 'daily']):
            for table_name, profile in state.enhanced_profiles.items():
                temporal_cols = getattr(profile, 'temporal_columns', [])
                if temporal_cols:
                    print(f"  🎯 Selected {table_name} for temporal analysis (temporal columns: {temporal_cols})")
                    return table_name
        
        # For correlation questions, prioritize tables with multiple numeric columns  
        if any(keyword in question_lower for keyword in ['correlation', 'relationship', 'factor', 'shipping', 'price', 'freight']):
            best_correlation_table = None
            max_numeric_cols = 0
            for table_name, profile in state.enhanced_profiles.items():
                measure_cols = getattr(profile, 'measure_columns', [])
                if len(measure_cols) > max_numeric_cols:
                    max_numeric_cols = len(measure_cols)
                    best_correlation_table = table_name
            if best_correlation_table and max_numeric_cols >= 2:
                print(f"  🎯 Selected {best_correlation_table} for correlation analysis ({max_numeric_cols} numeric columns)")
                return best_correlation_table
        
        # For outlier/pricing questions, prioritize tables with price-related columns
        if any(keyword in question_lower for keyword in ['outlier', 'unusual', 'pricing', 'price', 'abnormal', 'pattern']):
            for table_name, profile in state.enhanced_profiles.items():
                measure_cols = getattr(profile, 'measure_columns', [])
                if any('price' in col.lower() or 'cost' in col.lower() or 'freight' in col.lower() for col in measure_cols):
                    print(f"  🎯 Selected {table_name} for pricing/outlier analysis")
                    return table_name
        
        # For segmentation questions, prioritize tables with categorical columns
        if any(keyword in question_lower for keyword in ['segment', 'customer', 'group', 'category', 'behavior']):
            for table_name, profile in state.enhanced_profiles.items():
                dimension_cols = getattr(profile, 'dimension_columns', [])
                if dimension_cols and 'customer' in table_name.lower():
                    print(f"  🎯 Selected {table_name} for segmentation analysis")
                    return table_name
        
        # Default scoring logic
        table_scores = {}
        
        for table_name, profile in state.enhanced_profiles.items():
            score = 0
            
            # Data quality score
            if hasattr(profile, 'data_quality_score'):
                score += profile.data_quality_score * 30
            
            # ML readiness score
            if hasattr(profile, 'ml_readiness_score') and profile.ml_readiness_score:
                score += profile.ml_readiness_score * 20
            
            # Number of insights
            if hasattr(profile, 'key_insights') and profile.key_insights:
                score += len(profile.key_insights) * 10
            
            # Column richness
            measure_count = len(getattr(profile, 'measure_columns', []))
            dimension_count = len(getattr(profile, 'dimension_columns', []))
            score += (measure_count + dimension_count) * 5
            
            # Table size (moderate preference for medium-sized tables)
            row_count = len(state.tables[table_name])
            if 100 <= row_count <= 10000:
                score += 15
            elif row_count > 10000:
                score += 10
            
            table_scores[table_name] = score
        
        # Return table with highest score
        selected = max(table_scores.items(), key=lambda x: x[1])[0]
        print(f"  🎯 Selected {selected} using default scoring")
        return selected
    
    def _execute_intelligence_cycle(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one intelligence-driven exploration cycle"""
        
        state = self._comprehensive_state
        
        # Get primary table for this iteration
        primary_table = strategy['primary_table']
        
        # Generate intelligent operation
        operation = self.intelligence_toolkit.generate_intelligent_operation(
            primary_table, 
            state.user_question, 
            state.iteration_count
        )
        
        print(f"🎯 Table: {primary_table}")
        print(f"🔧 Operation: {operation}")
        
        # Validate and sanitize operation before execution
        operation = self._validate_and_sanitize_operation(operation)
        
        # Execute operation
        execution_result = self.toolkit.execute_pandas_operation(operation, state.tables)
        
        # Record operation
        operation_record = {
            'iteration': state.iteration_count,
            'table': primary_table,
            'operation': operation,
            'success': execution_result['success'],
            'intelligence_driven': True
        }
        
        if execution_result['success']:
            operation_record['result_summary'] = self._summarize_result(execution_result)
            print(f"✅ {operation_record['result_summary']}")
        else:
            operation_record['error'] = execution_result['error']['message']
            print(f"❌ Error: {execution_result['error']['message']}")
        
        state.operations_executed.append(operation_record)
        
        # Generate intelligence-enhanced insights
        insights = self._synthesize_intelligence_insights(
            operation, execution_result, primary_table
        )
        
        # Update state with new findings
        state.current_findings.extend(insights.get('findings', []))
        
        return {
            'operation': operation,
            'result': execution_result,
            'insights': insights,
            'sufficient_insights': self._assess_insight_sufficiency(insights)
        }
    
    def _synthesize_intelligence_insights(self, operation: str, 
                                        execution_result: Dict[str, Any], 
                                        table_name: str) -> Dict[str, Any]:
        """Synthesize insights using intelligence context"""
        
        state = self._comprehensive_state
        profile = state.enhanced_profiles.get(table_name)
        
        # Get intelligence context
        intelligence_summary = state.intelligence_summary.get(table_name, {})
        
        if not execution_result['success']:
            return {
                'findings': [f"Operation failed: {execution_result['error']['message']}"],
                'confidence': 0.2,
                'intelligence_enhanced': False
            }
        
        # Create intelligence-enhanced prompt
        prompt = f"""
Analyze this data exploration result using comprehensive intelligence context:

OPERATION: {operation}
TABLE: {table_name}
RESULT: {self._format_result_for_analysis(execution_result)}

INTELLIGENCE CONTEXT:
- Data Quality Score: {profile.data_quality_score:.1%} if hasattr(profile, 'data_quality_score') else 'Unknown'
- ML Readiness: {getattr(profile, 'ml_readiness_score', 'Unknown')}%
- Key Insights: {getattr(profile, 'key_insights', ['None'])[:3]}
- Quality Issues: {intelligence_summary.get('quality_insights', {}).get('recommendations', [])}
- Correlation Insights: {intelligence_summary.get('correlation_insights', {})}
- Outlier Insights: {intelligence_summary.get('outlier_insights', {})}

USER QUESTION: "{state.user_question}"

Based on this comprehensive intelligence context, provide insights that:
1. Directly address the user's question using the intelligence
2. Explain what the operation reveals in business terms
3. Reference specific intelligence insights (quality, correlations, outliers)
4. Assess confidence based on data quality and intelligence
5. Suggest next steps based on intelligence recommendations

Respond in JSON format:
{{
    "findings": ["intelligence-enhanced finding 1", "finding 2"],
    "confidence": 0.8,
    "intelligence_insights": ["specific intelligence insight"],
    "business_relevance": "how this relates to business question",
    "next_steps": ["recommended next step"],
    "data_quality_notes": ["quality consideration"]
}}
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Clean up JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            result = json.loads(content)
            result['intelligence_enhanced'] = True
            
            return result
            
        except Exception as e:
            # Fallback with intelligence context
            return {
                'findings': [
                    f"Executed {operation} on {table_name}",
                    f"Intelligence context: Quality {profile.data_quality_score:.1%}" if hasattr(profile, 'data_quality_score') else "Intelligence analysis applied"
                ],
                'confidence': 0.6,
                'intelligence_enhanced': True,
                'error': f"Intelligence synthesis error: {str(e)}"
            }
    
    def _format_result_for_analysis(self, execution_result: Dict[str, Any]) -> str:
        """Format execution result for LLM analysis"""
        output = execution_result.get('output', {})
        
        if output.get('type') == 'dataframe':
            shape = output.get('shape', (0, 0))
            columns = output.get('columns', [])
            return f"DataFrame with {shape[0]} rows, {shape[1]} columns. Columns: {columns[:5]}"
        
        elif output.get('type') == 'series':
            data = output.get('data', {})
            return f"Series with {len(data)} values. Sample: {dict(list(data.items())[:3])}"
        
        elif output.get('type') == 'scalar':
            return f"Scalar value: {output.get('value')}"
        
        else:
            return str(output)[:200]
    
    def _update_confidence_with_intelligence(self, operation_result: Dict[str, Any]):
        """Update confidence using intelligence insights"""
        state = self._comprehensive_state
        
        # Base confidence from operation
        base_confidence = operation_result.get('insights', {}).get('confidence', 0.5)
        
        # Adjust based on intelligence quality
        intelligence_bonus = 0
        
        # Data quality bonus
        avg_quality = np.mean([
            getattr(profile, 'data_quality_score', 0.5) 
            for profile in state.enhanced_profiles.values()
        ])
        intelligence_bonus += (avg_quality - 0.5) * 0.3
        
        # Intelligence insights bonus
        total_insights = sum([
            len(getattr(profile, 'key_insights', [])) 
            for profile in state.enhanced_profiles.values()
        ])
        if total_insights > 5:
            intelligence_bonus += 0.1
        
        # Operation success bonus
        if operation_result['result']['success']:
            intelligence_bonus += 0.1
        
        # Update confidence
        new_confidence = min(0.95, base_confidence + intelligence_bonus)
        state.confidence_level = new_confidence
    
    def _assess_insight_sufficiency(self, insights: Dict[str, Any]) -> bool:
        """Assess if we have sufficient insights based on intelligence"""
        state = self._comprehensive_state
        
        # Check confidence threshold
        if insights.get('confidence', 0) < 0.6:
            return False
        
        # Check if we have addressed the main question type
        analysis_type = state.business_context['analysis_type']
        
        findings_text = ' '.join(insights.get('findings', [])).lower()
        
        if analysis_type == 'temporal_analysis':
            return any(keyword in findings_text for keyword in ['temporal', 'time', 'date', 'seasonal'])
        
        elif analysis_type == 'correlation_analysis':
            return any(keyword in findings_text for keyword in ['correlation', 'relationship', 'association'])
        
        elif analysis_type == 'segmentation_analysis':
            return any(keyword in findings_text for keyword in ['segment', 'group', 'category'])
        
        # General sufficiency check
        return (len(state.current_findings) >= 3 and 
                state.confidence_level >= 0.7 and 
                state.iteration_count >= 2)
    
    def _generate_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights with full intelligence context"""
        
        state = self._comprehensive_state
        
        # Collect all intelligence context
        intelligence_context = []
        for table_name, summary in state.intelligence_summary.items():
            if summary.get('key_insights'):
                intelligence_context.extend([
                    f"{table_name}: {insight}" for insight in summary['key_insights'][:2]
                ])
        
        # Create comprehensive analysis prompt
        prompt = f"""
Generate comprehensive insights from this intelligence-driven data exploration:

ORIGINAL QUESTION: "{state.user_question}"
ANALYSIS TYPE: {state.business_context['analysis_type']}

INTELLIGENCE CONTEXT:
{chr(10).join(f"- {ctx}" for ctx in intelligence_context[:10])}

EXPLORATION FINDINGS:
{chr(10).join(f"- {finding}" for finding in state.current_findings[-8:])}

OPERATIONS EXECUTED: {len(state.operations_executed)}
FINAL CONFIDENCE: {state.confidence_level:.2%}

DATA QUALITY CONTEXT:
{chr(10).join(f"- {table}: {summary.get('data_quality_score', 0):.1%} quality" for table, summary in state.intelligence_summary.items())}

Generate comprehensive insights that:
1. Directly answer the user's question using intelligence insights
2. Explain findings in business context
3. Reference specific data quality and intelligence considerations
4. Provide actionable recommendations based on the analysis

Respond in JSON format:
{{
    "direct_answer": "Clear answer leveraging intelligence insights",
    "key_insights": ["insight 1 with intelligence context", "insight 2"],
    "supporting_evidence": ["specific evidence from operations"],
    "intelligence_applied": ["how intelligence enhanced the analysis"],
    "data_quality_assessment": ["quality considerations"],
    "business_implications": ["business impact"],
    "confidence_score": 0.85
}}
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            
            insights = json.loads(content)
            
            # Enhance with intelligence metadata
            insights['intelligence_metadata'] = {
                'profiles_used': len(state.enhanced_profiles),
                'intelligence_insights_count': len(intelligence_context),
                'operations_executed': len(state.operations_executed),
                'analysis_approach': 'comprehensive_intelligence_driven'
            }
            
            return insights
            
        except Exception as e:
            # Fallback comprehensive insights
            return {
                "direct_answer": f"Comprehensive analysis completed using {len(state.enhanced_profiles)} intelligence profiles",
                "key_insights": state.current_findings[-5:],
                "supporting_evidence": [f"Executed {len(state.operations_executed)} intelligence-driven operations"],
                "intelligence_applied": [f"Used comprehensive profiling on {len(state.enhanced_profiles)} tables"],
                "data_quality_assessment": state.quality_concerns,
                "business_implications": ["Analysis leveraged full knowledge graph intelligence"],
                "confidence_score": state.confidence_level,
                "intelligence_metadata": {
                    "error": str(e),
                    "fallback_used": True
                }
            }
    
    def _generate_intelligence_recommendations(self) -> List[str]:
        """Generate recommendations based on intelligence insights"""
        state = self._comprehensive_state
        recommendations = []
        
        # Quality-based recommendations
        for table_name, summary in state.intelligence_summary.items():
            quality_recs = summary.get('quality_insights', {}).get('recommendations', [])
            recommendations.extend(quality_recs[:2])
        
        # Intelligence-specific recommendations
        if state.confidence_level < 0.7:
            recommendations.append("Consider additional data sources to increase analysis confidence")
        
        if len(state.operations_executed) < 3:
            recommendations.append("Extend exploration with more targeted intelligence-driven operations")
        
        # Analysis type specific recommendations
        analysis_type = state.business_context['analysis_type']
        if analysis_type == 'temporal_analysis' and not state.business_context['temporal_scope']['has_temporal_data']:
            recommendations.append("Collect temporal data for more comprehensive time-series analysis")
        
        recommendations.append("Leverage the comprehensive intelligence profiles for future analysis")
        
        return recommendations[:7]
    
    def _summarize_data_quality(self, profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize data quality across all tables"""
        quality_summary = {
            'overall_score': 0,
            'table_scores': {},
            'critical_issues': 0,
            'recommendations': []
        }
        
        scores = []
        for table_name, profile in profiles.items():
            if hasattr(profile, 'data_quality_score'):
                score = profile.data_quality_score
                scores.append(score)
                quality_summary['table_scores'][table_name] = score
                
                if hasattr(profile, 'quality_profile') and profile.quality_profile:
                    quality_summary['critical_issues'] += len(profile.quality_profile.critical_alerts)
                    quality_summary['recommendations'].extend(profile.quality_profile.quality_recommendations[:2])
        
        if scores:
            quality_summary['overall_score'] = np.mean(scores)
        
        return quality_summary
    
    def _build_relationships_context(self) -> Dict[str, Any]:
        """Build relationships context using semantic graph builder"""
        try:
            if self.semantic_graph_builder:
                # Try different methods that might exist
                if hasattr(self.semantic_graph_builder, 'get_relationship_summary'):
                    return self.semantic_graph_builder.get_relationship_summary()
                elif hasattr(self.semantic_graph_builder, 'export_graph_summary'):
                    return self.semantic_graph_builder.export_graph_summary()
                else:
                    # Return basic graph info
                    return {"nodes": 0, "edges": 0, "note": "Graph builder available but no summary method"}
        except Exception as e:
            self.logger.warning(f"Failed to build relationships context: {e}")
        
        return {}
    
    def _find_common_patterns(self, profile1: Any, profile2: Any) -> List[str]:
        """Find common patterns between two table profiles"""
        patterns = []
        
        # Check common business domain
        domain1 = getattr(profile1, 'business_domain', None)
        domain2 = getattr(profile2, 'business_domain', None)
        if domain1 and domain2 and domain1 == domain2:
            patterns.append(f"Same business domain: {domain1}")
        
        # Check common column types
        measures1 = set(getattr(profile1, 'measure_columns', []))
        measures2 = set(getattr(profile2, 'measure_columns', []))
        common_measures = measures1.intersection(measures2)
        if common_measures:
            patterns.append(f"Common measures: {list(common_measures)[:3]}")
        
        return patterns
    
    def _validate_and_sanitize_operation(self, operation: str) -> str:
        """Validate and sanitize operation before execution"""
        if not operation or not operation.strip():
            return "print('No operation generated')"
        
        operation = operation.strip()
        
        # Convert comments to executable Python print statements
        if operation.startswith('#'):
            comment_text = operation[1:].strip()
            return f"print('{comment_text}')"
        
        # Handle multiline comments
        if operation.startswith('# ') and '\n' in operation:
            lines = operation.split('\n')
            first_line = lines[0][1:].strip()
            return f"print('{first_line}')"
        
        # Ensure operation doesn't contain dangerous code
        dangerous_keywords = ['import os', 'import sys', '__import__', 'exec(', 'eval(']
        if any(keyword in operation for keyword in dangerous_keywords):
            return "print('Operation blocked for security')"
        
        return operation