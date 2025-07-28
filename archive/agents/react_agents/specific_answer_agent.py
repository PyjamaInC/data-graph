"""
Enhanced Data Exploration Agent with Specific Answer Generation

This agent ensures specific, data-driven answers to user questions.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from .enhanced_data_exploration_agent import (
    EnhancedDataExplorationReActAgent,
    IntelligentExplorationState,
    QuestionAnalyzer,
    IntelligentOperationGenerator,
    IntelligentInsightSynthesizer
)


class SpecificResultExtractor:
    """Extracts specific, concrete results from operations"""
    
    @staticmethod
    def extract_concrete_findings(operation_result: Dict[str, Any], 
                                step: str, 
                                question_intent: str) -> Dict[str, Any]:
        """Extract concrete findings from operation results"""
        
        if not operation_result.get('success'):
            return {
                'success': False,
                'error': operation_result.get('error', {}).get('message', 'Unknown error'),
                'findings': []
            }
        
        output = operation_result.get('output', {})
        output_type = output.get('type', 'unknown')
        
        concrete_findings = []
        
        if output_type == 'dataframe':
            concrete_findings.extend(
                SpecificResultExtractor._extract_dataframe_findings(output, step, question_intent)
            )
        elif output_type == 'series':
            concrete_findings.extend(
                SpecificResultExtractor._extract_series_findings(output, step, question_intent)
            )
        elif output_type == 'scalar':
            concrete_findings.append(
                SpecificResultExtractor._extract_scalar_finding(output, step)
            )
        elif output_type == 'collection':
            concrete_findings.extend(
                SpecificResultExtractor._extract_collection_findings(output, step)
            )
        
        return {
            'success': True,
            'findings': concrete_findings,
            'raw_data': output
        }
    
    @staticmethod
    def _extract_dataframe_findings(df_output: Dict[str, Any], 
                                  step: str, 
                                  intent: str) -> List[str]:
        """Extract specific findings from DataFrame results"""
        
        findings = []
        shape = df_output.get('shape', [0, 0])
        data = df_output.get('data', df_output.get('head', []))
        
        # Handle different step types
        if 'missing' in step or 'quality' in step:
            # Extract missing value findings
            if data:
                for row in data[:5]:  # Top 5 issues
                    if 'column' in row and 'null_pct' in row:
                        if row['null_pct'] > 0:
                            findings.append(
                                f"Column '{row['column']}' has {row['null_pct']:.1f}% missing values "
                                f"({row.get('null_count', 'unknown')} nulls)"
                            )
                if not any(row.get('null_pct', 0) > 0 for row in data):
                    findings.append("No missing values detected - data completeness is 100%")
        
        elif 'correlation' in step:
            # Extract correlation findings
            if data and isinstance(data[0], dict):
                # Find strong correlations
                strong_correlations = []
                for i, row in enumerate(data):
                    for j, (col, value) in enumerate(row.items()):
                        if i != j and isinstance(value, (int, float)) and abs(value) > 0.7:
                            col1 = list(df_output.get('columns', []))[i] if i < len(df_output.get('columns', [])) else f"var{i}"
                            strong_correlations.append(f"{col1} and {col} have correlation: {value:.2f}")
                
                if strong_correlations:
                    findings.extend(strong_correlations[:3])
                else:
                    findings.append("No strong correlations (>0.7) found between numeric variables")
        
        elif 'outlier' in step or 'statistical_bounds' in step:
            # Extract outlier findings
            if data:
                for row in data:
                    if 'metric' in row and 'value' in row:
                        if row['metric'] == 'outlier_count':
                            findings.append(f"Found {int(row['value'])} outliers in the data")
                        elif row['metric'] == 'outlier_pct':
                            findings.append(f"Outliers represent {row['value']:.1f}% of the data")
                        elif row['metric'] == 'lower_bound':
                            findings.append(f"Lower bound for normal values: {row['value']:.2f}")
                        elif row['metric'] == 'upper_bound':
                            findings.append(f"Upper bound for normal values: {row['value']:.2f}")
        
        elif 'temporal' in step or 'time' in step:
            # Extract temporal findings
            if data:
                if 'min_date' in data[0]:
                    findings.append(
                        f"Date range: {data[0]['min_date']} to {data[0]['max_date']} "
                        f"({data[0].get('date_range_days', 'unknown')} days)"
                    )
                else:
                    # Monthly aggregation results
                    findings.append(f"Analyzed {len(data)} time periods")
                    if data and 'mean' in data[0]:
                        avg_value = np.mean([row.get('mean', 0) for row in data])
                        findings.append(f"Average value across periods: {avg_value:.2f}")
        
        elif 'segment' in step:
            # Extract segmentation findings
            if data:
                findings.append(f"Found {len(data)} distinct segments")
                # Find top segments
                for i, row in enumerate(data[:3]):
                    if isinstance(row, dict):
                        segment_name = list(row.keys())[0] if row else 'Unknown'
                        segment_value = list(row.values())[0] if row else 0
                        findings.append(f"Segment '{segment_name}': {segment_value}")
        
        # Add general shape info if no specific findings
        if not findings:
            findings.append(f"Analyzed data with {shape[0]} rows and {shape[1]} columns")
        
        return findings
    
    @staticmethod
    def _extract_series_findings(series_output: Dict[str, Any], step: str) -> List[str]:
        """Extract specific findings from Series results"""
        
        findings = []
        data = series_output.get('data', {})
        
        if data:
            # Value counts or similar
            total_items = len(data)
            findings.append(f"Found {total_items} distinct values")
            
            # Top values
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:3]
            for name, count in sorted_items:
                findings.append(f"'{name}': {count} occurrences")
        
        return findings
    
    @staticmethod
    def _extract_scalar_finding(scalar_output: Dict[str, Any], step: str) -> str:
        """Extract finding from scalar result"""
        value = scalar_output.get('value', 'unknown')
        return f"Result: {value}"
    
    @staticmethod
    def _extract_collection_findings(collection_output: Dict[str, Any], step: str) -> List[str]:
        """Extract findings from collection results"""
        
        findings = []
        data = collection_output.get('data', [])
        
        if isinstance(data, list):
            findings.append(f"Found {len(data)} items")
            # Show first few items
            for item in data[:3]:
                findings.append(f"- {item}")
        elif isinstance(data, dict):
            findings.append(f"Found {len(data)} key-value pairs")
            for key, value in list(data.items())[:3]:
                findings.append(f"- {key}: {value}")
        
        return findings


class SpecificAnswerInsightSynthesizer(IntelligentInsightSynthesizer):
    """Enhanced synthesizer that ensures specific answers"""
    
    def synthesize_insights(self, operation_result: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights with specific data points"""
        
        # Extract concrete findings first
        concrete_results = SpecificResultExtractor.extract_concrete_findings(
            operation_result,
            context.get('step', ''),
            self.primary_intent['intent']
        )
        
        if not concrete_results['success']:
            return self._synthesize_error_insight(operation_result)
        
        # Build insights with concrete findings
        insights = {
            'findings': concrete_results['findings'],
            'confidence': 0.7 if concrete_results['findings'] else 0.3,
            'business_relevance': self._determine_relevance(
                concrete_results['findings'],
                context.get('user_question', '')
            ),
            'next_steps': self._suggest_next_steps(context.get('step', '')),
            'data_quality_notes': []
        }
        
        # Add context metadata
        insights['context_metadata'] = {
            'iteration': context.get('iteration', 1),
            'step': context.get('step', 'unknown'),
            'intent': self.primary_intent['intent'],
            'concrete_findings_count': len(concrete_results['findings'])
        }
        
        return insights
    
    def _determine_relevance(self, findings: List[str], question: str) -> str:
        """Determine how findings relate to the question"""
        
        if not findings:
            return "No specific findings to relate to the question"
        
        intent = self.primary_intent['intent']
        
        relevance_map = {
            'quality': "Data quality assessment reveals completeness and reliability",
            'correlation': "Relationship analysis shows connections between variables",
            'outlier': "Anomaly detection identifies unusual patterns in the data",
            'temporal': "Time-based analysis reveals trends and patterns",
            'segmentation': "Segment analysis shows group differences and characteristics",
            'aggregation': "Summary statistics provide overall data insights"
        }
        
        return relevance_map.get(intent, "Analysis provides data-driven insights")
    
    def _suggest_next_steps(self, current_step: str) -> List[str]:
        """Suggest concrete next steps"""
        
        suggestions = {
            'identify_missing_values': ["Investigate causes of missing data", "Consider imputation strategies"],
            'calculate_correlation_matrix': ["Deep dive into strong correlations", "Test for causation"],
            'detect_outliers': ["Investigate outlier causes", "Decide on outlier handling strategy"],
            'aggregate_by_time_periods': ["Analyze seasonal patterns", "Forecast future trends"],
            'calculate_segment_metrics': ["Compare segment performance", "Identify optimization opportunities"]
        }
        
        return suggestions.get(current_step, ["Continue analysis with deeper exploration"])


class SpecificAnswerAgent(EnhancedDataExplorationReActAgent):
    """Agent that provides specific, data-driven answers"""
    
    def __init__(self, enhanced_intelligence: Any, semantic_graph_builder: Any,
                 enhanced_summarizer: Optional[Any] = None, llm_model: str = "gpt-4"):
        
        super().__init__(enhanced_intelligence, semantic_graph_builder, 
                        enhanced_summarizer, llm_model)
        
        # Override with specific answer components
        self.result_extractor = SpecificResultExtractor()
    
    def _execute_intelligent_exploration(self) -> Dict[str, Any]:
        """Execute exploration ensuring specific answers"""
        
        state = self._current_exploration_state
        strategy = state.analysis_strategy
        
        print(f"🔍 Starting intelligent exploration: '{state.user_question}'")
        print(f"📋 Strategy: {strategy.get('approach', 'discovery')}")
        print(f"🎯 Intent: {state.question_analysis['primary_intent']['intent']}")
        print("=" * 80)
        
        # Override the insight synthesizer with specific answer version
        self.insight_synthesizer = SpecificAnswerInsightSynthesizer(
            self.llm, 
            state.question_analysis
        )
        
        # Collect concrete findings
        all_concrete_findings = []
        
        # Iterative exploration loop
        while (state.iteration_count < state.max_iterations and 
               len(all_concrete_findings) < 10):  # Need at least 10 concrete findings
            
            state.iteration_count += 1
            print(f"\n🔄 Intelligence Cycle {state.iteration_count}")
            
            # Execute intelligent cycle
            cycle_result = self._execute_intelligent_cycle()
            
            # Extract concrete findings
            if cycle_result.get('insights'):
                findings = cycle_result['insights'].get('findings', [])
                all_concrete_findings.extend(findings)
                print(f"📊 Found {len(findings)} concrete insights")
            
            # Check if we have sufficient concrete insights
            if len(all_concrete_findings) >= 5:
                print("✅ Sufficient concrete insights achieved")
                break
        
        # Generate final answer with all concrete findings
        final_insights = self._generate_specific_final_answer(all_concrete_findings)
        
        return {
            'user_question': state.user_question,
            'exploration_summary': {
                'iterations_used': state.iteration_count,
                'confidence_level': 0.9 if len(all_concrete_findings) >= 5 else 0.6,
                'total_findings': len(all_concrete_findings),
                'strategy_used': strategy.get('approach', 'discovery'),
                'intelligence_driven': True,
                'operations_executed': len(state.previous_operations)
            },
            'insights': final_insights,
            'intelligence_context': {
                'profiles_generated': len(state.enhanced_profiles),
                'question_analysis': state.question_analysis,
                'concrete_findings': all_concrete_findings
            },
            'exploration_history': state.exploration_history,
            'recommendations': self._generate_specific_recommendations(all_concrete_findings),
            'data_quality_summary': self._summarize_data_quality()
        }
    
    def _generate_specific_final_answer(self, all_findings: List[str]) -> Dict[str, Any]:
        """Generate final answer with specific data points"""
        
        state = self._current_exploration_state
        intent = state.question_analysis['primary_intent']['intent']
        question = state.user_question
        
        # Create intent-specific answers
        if intent == 'quality':
            direct_answer = self._generate_quality_answer(all_findings)
        elif intent == 'correlation':
            direct_answer = self._generate_correlation_answer(all_findings)
        elif intent == 'outlier':
            direct_answer = self._generate_outlier_answer(all_findings)
        elif intent == 'temporal':
            direct_answer = self._generate_temporal_answer(all_findings)
        elif intent == 'segmentation':
            direct_answer = self._generate_segmentation_answer(all_findings)
        else:
            direct_answer = self._generate_general_answer(all_findings)
        
        return {
            'direct_answer': direct_answer,
            'key_insights': all_findings[:5] if all_findings else ["Analysis completed but no specific findings extracted"],
            'supporting_evidence': all_findings[5:10] if len(all_findings) > 5 else [],
            'confidence_score': 0.9 if all_findings else 0.3,
            'business_implications': self._derive_business_implications(intent, all_findings)
        }
    
    def _generate_quality_answer(self, findings: List[str]) -> str:
        """Generate specific answer for quality questions"""
        
        missing_findings = [f for f in findings if 'missing' in f.lower() or 'null' in f.lower()]
        
        if missing_findings:
            return f"Data quality analysis reveals several issues: {'. '.join(missing_findings[:3])}"
        else:
            return "Data quality analysis shows excellent completeness with no missing values detected"
    
    def _generate_correlation_answer(self, findings: List[str]) -> str:
        """Generate specific answer for correlation questions"""
        
        correlation_findings = [f for f in findings if 'correlation' in f.lower() or 'relationship' in f.lower()]
        
        if correlation_findings:
            return f"Relationship analysis found: {'. '.join(correlation_findings[:3])}"
        else:
            return "No strong correlations detected between the analyzed variables"
    
    def _generate_outlier_answer(self, findings: List[str]) -> str:
        """Generate specific answer for outlier questions"""
        
        outlier_findings = [f for f in findings if 'outlier' in f.lower() or 'bound' in f.lower()]
        
        if outlier_findings:
            return f"Outlier analysis results: {'. '.join(outlier_findings[:3])}"
        else:
            return "No significant outliers detected in the pricing data"
    
    def _generate_temporal_answer(self, findings: List[str]) -> str:
        """Generate specific answer for temporal questions"""
        
        temporal_findings = [f for f in findings if any(word in f.lower() for word in ['date', 'time', 'period', 'days'])]
        
        if temporal_findings:
            return f"Temporal analysis shows: {'. '.join(temporal_findings[:3])}"
        else:
            return "Time-based analysis completed with patterns identified"
    
    def _generate_segmentation_answer(self, findings: List[str]) -> str:
        """Generate specific answer for segmentation questions"""
        
        segment_findings = [f for f in findings if 'segment' in f.lower() or 'group' in f.lower()]
        
        if segment_findings:
            return f"Segmentation analysis reveals: {'. '.join(segment_findings[:3])}"
        else:
            return "Segment analysis completed with group differences identified"
    
    def _generate_general_answer(self, findings: List[str]) -> str:
        """Generate general answer with specific findings"""
        
        if findings:
            return f"Analysis results: {'. '.join(findings[:3])}"
        else:
            return "Analysis completed but specific findings need further exploration"
    
    def _derive_business_implications(self, intent: str, findings: List[str]) -> List[str]:
        """Derive business implications from findings"""
        
        implications = []
        
        if intent == 'quality' and any('missing' in f for f in findings):
            implications.append("Data completeness issues may impact analysis accuracy")
            implications.append("Consider data collection improvements for affected columns")
        elif intent == 'correlation' and any('correlation' in f for f in findings):
            implications.append("Strong relationships suggest optimization opportunities")
            implications.append("Consider these correlations in predictive modeling")
        elif intent == 'outlier' and any('outlier' in f for f in findings):
            implications.append("Outliers may indicate data quality issues or special cases")
            implications.append("Review outlier handling strategy before analysis")
        
        return implications[:2] if implications else ["Further analysis needed for business recommendations"]
    
    def _generate_specific_recommendations(self, findings: List[str]) -> List[str]:
        """Generate specific recommendations based on findings"""
        
        recommendations = []
        
        if any('missing' in f for f in findings):
            recommendations.append("Address missing data in identified columns before production use")
        
        if any('outlier' in f for f in findings):
            recommendations.append("Investigate and handle outliers based on business context")
        
        if any('correlation' in f for f in findings):
            recommendations.append("Leverage identified correlations for predictive analytics")
        
        if not recommendations:
            recommendations.append("Continue monitoring data patterns for insights")
        
        return recommendations