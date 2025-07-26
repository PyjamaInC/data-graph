"""
Error Handling and Recovery for ReAct Agents

Provides comprehensive error handling with stage-specific recovery strategies,
fallback mechanisms, and graceful degradation.
"""

import time
import re
from typing import Dict, Any, List

try:
    from .state_manager import ReActQueryState
except ImportError:
    from state_manager import ReActQueryState


class ReActErrorHandler:
    """Comprehensive error handling and recovery for ReAct agents"""
    
    def __init__(self):
        self.recovery_strategies = {
            'intent_recognition': self._recover_intent_failure,
            'schema_validation': self._recover_schema_failure,  
            'relationship_exploration': self._recover_relationship_failure
        }
        
    def handle_stage_error(self, state: ReActQueryState, error: Exception, stage: str) -> Dict[str, Any]:
        """Handle errors with stage-specific recovery strategies"""
        
        error_info = {
            'stage': stage,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time(),
            'recovery_attempt': state.get('recovery_attempts', 0) + 1
        }
        
        # Update error tracking in state
        state['error_count'] += 1
        state['last_error'] = error_info['error_message']
        state['recovery_attempts'] = error_info['recovery_attempt']
        
        # Apply recovery strategy
        if error_info['recovery_attempt'] <= 2 and stage in self.recovery_strategies:
            recovery_result = self.recovery_strategies[stage](state, error_info)
            return {
                'recovery_applied': True,
                'recovery_result': recovery_result,
                'should_retry': recovery_result.get('should_retry', False),
                'fallback_data': recovery_result.get('fallback_data', {}),
                'error_info': error_info
            }
        else:
            # Max retries exceeded or no recovery strategy
            return self._final_fallback(state, error_info)
    
    def _recover_intent_failure(self, state: ReActQueryState, error_info: Dict) -> Dict[str, Any]:
        """Recovery strategy for intent recognition failures"""
        
        error_msg = error_info['error_message'].lower()
        
        if 'json' in error_msg or 'parse' in error_msg:
            # JSON parsing error - try with more structured prompt
            return {
                'should_retry': True,
                'modified_prompt': True,
                'prompt_adjustment': 'Use more structured JSON format with examples',
                'fallback_data': self._generate_basic_intent(state)
            }
        elif 'api' in error_msg or 'rate' in error_msg or 'timeout' in error_msg:
            # API issues - wait and retry with fallback
            return {
                'should_retry': True, 
                'wait_time': 2 ** error_info['recovery_attempt'],  # Exponential backoff
                'fallback_data': self._generate_basic_intent(state)
            }
        else:
            # Use basic intent extraction without LLM
            return {
                'should_retry': False,
                'fallback_data': self._generate_basic_intent(state)
            }
    
    def _recover_schema_failure(self, state: ReActQueryState, error_info: Dict) -> Dict[str, Any]:
        """Recovery strategy for schema validation failures"""
        
        error_msg = error_info['error_message'].lower()
        
        if 'no matches found' in error_msg or 'no mapping' in error_msg:
            # Expand search criteria
            return {
                'should_retry': True,
                'search_expansion': True,
                'fallback_data': self._generate_broad_schema_mapping(state)
            }
        elif 'schema' in error_msg or 'attribute' in error_msg:
            # Schema access issues - use all available tables
            return {
                'should_retry': False,
                'fallback_data': self._generate_fallback_schema_mapping(state)
            }
        else:
            return {
                'should_retry': False,
                'fallback_data': self._generate_fallback_schema_mapping(state)
            }
    
    def _recover_relationship_failure(self, state: ReActQueryState, error_info: Dict) -> Dict[str, Any]:
        """Recovery strategy for relationship exploration failures"""
        
        error_msg = error_info['error_message'].lower()
        
        if 'no path found' in error_msg or 'no join' in error_msg:
            # Skip complex joins, use simple approach
            return {
                'should_retry': False,
                'fallback_data': {
                    'join_strategy': {
                        'join_path': state.get('validated_mapping', {}).get('relevant_tables', []),
                        'join_conditions': [],
                        'path_confidence': 0.3,
                        'strategy_type': 'simple_joins',
                        'estimated_performance': 'moderate'
                    },
                    'confidence': 0.3
                }
            }
        elif 'graph' in error_msg or 'knowledge' in error_msg:
            # Knowledge graph issues - use table-level joins only
            return {
                'should_retry': False,
                'fallback_data': {
                    'join_strategy': {
                        'join_path': state.get('validated_mapping', {}).get('relevant_tables', []),
                        'join_conditions': [],
                        'path_confidence': 0.4,
                        'strategy_type': 'table_level_only',
                        'estimated_performance': 'slow'
                    },
                    'confidence': 0.4
                }
            }
        else:
            return {
                'should_retry': False,
                'fallback_data': {
                    'join_strategy': {
                        'join_path': [],
                        'join_conditions': [],
                        'path_confidence': 0.2,
                        'strategy_type': 'no_joins',
                        'estimated_performance': 'unknown'
                    },
                    'confidence': 0.2
                }
            }
    
    def _generate_basic_intent(self, state: ReActQueryState) -> Dict[str, Any]:
        """Generate basic intent without LLM using rule-based approach"""
        query = state['user_query'].lower()
        
        # Simple keyword-based intent detection
        if any(word in query for word in ['sum', 'total', 'count', 'average', 'group']):
            action_type = 'aggregation'
        elif any(word in query for word in ['location', 'city', 'state', 'country', 'region']):
            action_type = 'geographical_analysis'
        elif any(word in query for word in ['time', 'date', 'trend', 'over time', 'month', 'year']):
            action_type = 'trend_analysis'
        elif any(word in query for word in ['compare', 'vs', 'versus', 'between']):
            action_type = 'comparison'
        else:
            action_type = 'general_analysis'
        
        # Extract basic concepts (nouns from query)
        words = re.findall(r'\b[a-zA-Z]+\b', query)
        concepts = [word for word in words 
                   if len(word) > 3 and word not in ['show', 'find', 'what', 'where', 'when', 'how']][:3]
        
        return {
            'intent_profile': {
                'action_type': action_type,
                'target_concepts': concepts or ['data'],
                'analysis_scope': 'multi_table' if len(concepts) > 1 else 'single_table',
                'complexity': 'moderate',
                'confidence': 0.4
            },
            'confidence': 0.4,
            'stage_status': 'completed_with_basic_fallback'
        }
    
    def _generate_broad_schema_mapping(self, state: ReActQueryState) -> Dict[str, Any]:
        """Generate broad schema mapping when specific matching fails"""
        intent = state.get('intent_profile', {})
        concepts = intent.get('target_concepts', ['data'])
        
        # Create broad mappings for concepts
        concept_mappings = {}
        for concept in concepts:
            concept_mappings[concept] = [{
                'table': 'main_table',
                'column': concept,
                'semantic_role': 'dimension',
                'match_type': 'broad_search',
                'confidence': 0.5
            }]
        
        return {
            'validated_mapping': {
                'relevant_tables': ['main_table', 'secondary_table'],
                'concept_mappings': concept_mappings,
                'joins_needed': True,
                'required_roles': ['measure', 'dimension'],
                'mapping_confidence': 0.5,
                'mapping_strategy': 'broad_search'
            },
            'confidence': 0.5,
            'stage_status': 'completed_with_broad_mapping'
        }
    
    def _generate_fallback_schema_mapping(self, state: ReActQueryState) -> Dict[str, Any]:
        """Generate minimal fallback schema mapping"""
        return {
            'validated_mapping': {
                'relevant_tables': ['default_table'],
                'concept_mappings': {'data': [{
                    'table': 'default_table',
                    'column': 'value',
                    'semantic_role': 'measure',
                    'match_type': 'fallback',
                    'confidence': 0.3
                }]},
                'joins_needed': False,
                'required_roles': ['measure'],
                'mapping_confidence': 0.3,
                'mapping_strategy': 'fallback'
            },
            'confidence': 0.3,
            'stage_status': 'completed_with_fallback'
        }
    
    def _final_fallback(self, state: ReActQueryState, error_info: Dict) -> Dict[str, Any]:
        """Final fallback when all recovery attempts fail"""
        return {
            'recovery_applied': False,
            'should_retry': False,
            'final_fallback': True,
            'fallback_data': {
                'basic_analysis': True,
                'error_summary': f"Failed at {error_info['stage']} after {error_info['recovery_attempt']} attempts",
                'confidence': 0.1,
                'stage_status': 'failed_with_final_fallback'
            },
            'error_info': error_info
        }


class ContextCompressor:
    """Advanced context compression utilities for token optimization"""
    
    @staticmethod
    def compress_schema_context(intent_concepts: List[str], schema_manager) -> str:
        """Build ultra-compressed schema focusing on relevant concepts"""
        if not hasattr(schema_manager, 'schema') or not schema_manager.schema:
            return "Schema: No schema available"
        
        compressed_tables = []
        
        for table_name, table_schema in schema_manager.schema.tables.items():
            # Get semantic role summary
            roles = set()
            relevant_cols = []
            
            for col_name, col_schema in table_schema.columns.items():
                role_str = col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role)
                roles.add(role_str[:3].upper())  # Abbreviate role
                
                # Include if matches concepts
                for concept in intent_concepts:
                    if (concept.lower() in col_name.lower() or 
                        col_name.lower() in concept.lower()):
                        relevant_cols.append(col_name)
            
            # Compressed notation: T(table_name):role1+role2[relevant_cols]
            role_str = '+'.join(sorted(roles))
            col_str = f"[{','.join(relevant_cols[:3])}]" if relevant_cols else ""
            
            compressed_tables.append(f"{table_name[0].upper()}({table_name}):{role_str}{col_str}")
        
        return f"Schema: {', '.join(compressed_tables)}"
    
    @staticmethod
    def compress_relationship_context(tables: List[str], relationship_context) -> str:
        """Build compressed relationship context for specific tables"""
        if not tables or len(tables) <= 1:
            return "Single table: no joins needed"
        
        if not relationship_context or not hasattr(relationship_context, 'strong_relationships'):
            return "No strong relationships found"
        
        relevant_rels = []
        table_set = set(tables)
        
        for rel in relationship_context.strong_relationships[:5]:  # Limit to top 5
            # Check if relationship involves our target tables
            rel_tables = set()
            if '.' in rel.get('from', ''):
                rel_tables.add(rel['from'].split('.')[0])
            if '.' in rel.get('to', ''):
                rel_tables.add(rel['to'].split('.')[0])
            
            if rel_tables.intersection(table_set):
                # Compressed notation: T1↔T2(weight,type)
                from_table = rel['from'].split('.')[0] if '.' in rel.get('from', '') else 'T1'
                to_table = rel['to'].split('.')[0] if '.' in rel.get('to', '') else 'T2'
                weight = rel.get('weight', 0.5)
                rel_type = rel.get('type', 'UNK')[:2]  # Abbreviate type
                
                relevant_rels.append(f"{from_table}↔{to_table}({weight:.2f},{rel_type})")
        
        return f"Joins: {', '.join(relevant_rels[:3])}" if relevant_rels else "No strong joins found"