"""
Relationship Explorer Agent - Stage 3 of ReAct Query Planning

Finds optimal join paths using existing KnowledgeGraphContextExtractor.
Provides intelligent relationship discovery with confidence scoring.
"""

from typing import Dict, Any, List, Optional

try:
    from .base_agent import BaseReActAgent
    from .state_manager import ReActQueryState
except ImportError:
    pass

# Mock for testing
class KnowledgeGraphContextExtractor:
    def extract_relationship_context(self):
        return type('obj', (object,), {
            'strong_relationships': [],
            'join_paths': []
        })()


class RelationshipExplorerAgent(BaseReActAgent):
    """Find optimal join paths using existing KnowledgeGraphContextExtractor"""
    
    def __init__(self, kg_extractor):
        super().__init__()
        self.kg_extractor = kg_extractor
        self.stage_name = "relationship_exploration"
        
    def _generate_thought(self, state: ReActQueryState) -> str:
        """Plan relationship discovery strategy"""
        mapping = state.get('validated_mapping', {})
        tables = mapping.get('relevant_tables', [])
        joins_needed = mapping.get('joins_needed', False)
        
        if not joins_needed:
            return f"Single table analysis on {tables[0] if tables else 'unknown'}, no joins needed"
        else:
            return f"Need to find optimal join path connecting {len(tables)} tables: {tables}"
    
    def _take_action(self, state: ReActQueryState, thought: str) -> Dict[str, Any]:
        """Leverage existing KG relationship extraction"""
        mapping = state.get('validated_mapping', {})
        
        if not mapping.get('joins_needed', False):
            return {
                'join_strategy': 'no_joins_needed',
                'target_table': mapping.get('relevant_tables', ['unknown'])[0],
                'relationship_context': None,
                'path_confidence': 1.0
            }
        
        try:
            # Use existing relationship extraction capabilities
            relationship_context = self.kg_extractor.extract_relationship_context()
            
            # Filter for relevant tables only
            relevant_joins = self._filter_relevant_joins(
                relationship_context.join_paths if hasattr(relationship_context, 'join_paths') else [], 
                mapping.get('relevant_tables', [])
            )
            
            # Find best join strategy
            join_strategy = self._select_optimal_join_strategy(relevant_joins, mapping.get('relevant_tables', []))
            
            return {
                'relationship_context': self._serialize_relationship_context(relationship_context),
                'relevant_joins': relevant_joins,
                'join_strategy': join_strategy,
                'table_count': len(mapping.get('relevant_tables', []))
            }
            
        except Exception as e:
            # Fallback to simple join strategy
            return self._generate_fallback_join_strategy(mapping.get('relevant_tables', []), str(e))
    
    def _make_observation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of discovered relationships"""
        
        if action.get('join_strategy') == 'no_joins_needed':
            return {
                'status': 'success',
                'confidence': 1.0,
                'join_complexity': 'none'
            }
        
        join_strategy = action.get('join_strategy', {})
        path_confidence = join_strategy.get('path_confidence', 0.0)
        
        if path_confidence > 0.8:
            return {
                'status': 'success',
                'confidence': path_confidence,
                'join_complexity': 'optimal'
            }
        elif path_confidence > 0.5:
            return {
                'status': 'partial',
                'confidence': path_confidence,
                'join_complexity': 'moderate'
            }
        else:
            return {
                'status': 'weak',
                'confidence': path_confidence,
                'join_complexity': 'complex'
            }
    
    def _synthesize_result(self, thought: str, action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured join strategy result"""
        
        join_strategy = action.get('join_strategy', {})
        confidence = observation.get('confidence', 0.5)
        
        return {
            'join_strategy': join_strategy,
            'confidence': confidence,
            'stage_status': observation['status'],
            'reasoning_chain': [
                thought,
                f"Analyzed {action.get('table_count', 0)} tables for join opportunities",
                f"Selected {observation.get('join_complexity', 'unknown')} join strategy"
            ]
        }
    
    def _filter_relevant_joins(self, all_joins: List[Dict], target_tables: List[str]) -> List[Dict]:
        """Filter join paths to only include target tables"""
        if not all_joins or not target_tables:
            return []
        
        relevant = []
        target_set = set(target_tables)
        
        for join in all_joins:
            from_table = join.get('from_table', '')
            to_table = join.get('to_table', '')
            
            if from_table in target_set and to_table in target_set:
                relevant.append(join)
        
        return sorted(relevant, key=lambda x: x.get('total_weight', 0), reverse=True)
    
    def _select_optimal_join_strategy(self, relevant_joins: List[Dict], tables: List[str]) -> Dict[str, Any]:
        """Select the best join strategy from available options"""
        
        if not relevant_joins:
            return self._generate_simple_join_strategy(tables)
        
        # Select join with highest confidence
        best_join = max(relevant_joins, key=lambda x: x.get('total_weight', 0))
        
        return {
            'join_path': best_join.get('path', tables),
            'join_conditions': self._extract_join_conditions(best_join),
            'path_confidence': best_join.get('total_weight', 0.5),
            'estimated_performance': self._estimate_performance(best_join),
            'strategy_type': 'knowledge_graph_optimized'
        }
    
    def _generate_simple_join_strategy(self, tables: List[str]) -> Dict[str, Any]:
        """Generate simple join strategy when KG relationships are not available"""
        
        if len(tables) <= 1:
            return {
                'join_path': tables,
                'join_conditions': [],
                'path_confidence': 1.0,
                'estimated_performance': 'fast',
                'strategy_type': 'single_table'
            }
        
        # Assume simple sequential joins
        join_conditions = []
        for i in range(len(tables) - 1):
            join_conditions.append({
                'from_table': tables[i],
                'to_table': tables[i + 1],
                'condition': f"{tables[i]}.id = {tables[i + 1]}.{tables[i]}_id",
                'confidence': 0.5
            })
        
        return {
            'join_path': tables,
            'join_conditions': join_conditions,
            'path_confidence': 0.5,
            'estimated_performance': 'moderate',
            'strategy_type': 'simple_sequential'
        }
    
    def _extract_join_conditions(self, join_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract join conditions from KG join information"""
        conditions = []
        
        # Try to extract from existing join info
        if 'join_recommendation' in join_info:
            recommendation = join_info['join_recommendation']
            # Parse recommendation string like "JOIN via: table1.col1 → table2.col2"
            if '→' in recommendation:
                parts = recommendation.split('→')
                if len(parts) >= 2:
                    from_col = parts[0].split(':')[-1].strip()
                    to_col = parts[1].strip()
                    conditions.append({
                        'from_column': from_col,
                        'to_column': to_col,
                        'join_type': 'inner',
                        'confidence': join_info.get('total_weight', 0.5)
                    })
        
        return conditions
    
    def _estimate_performance(self, join_info: Dict[str, Any]) -> str:
        """Estimate query performance based on join characteristics"""
        confidence = join_info.get('total_weight', 0.5)
        path_length = len(join_info.get('path', []))
        
        if confidence > 0.8 and path_length <= 2:
            return 'fast'
        elif confidence > 0.6 and path_length <= 3:
            return 'moderate'
        else:
            return 'slow'
    
    def _generate_fallback_join_strategy(self, tables: List[str], error_msg: str) -> Dict[str, Any]:
        """Generate fallback strategy when KG extraction fails"""
        
        return {
            'join_strategy': {
                'join_path': tables,
                'join_conditions': [],
                'path_confidence': 0.3,
                'estimated_performance': 'unknown',
                'strategy_type': 'fallback',
                'error': error_msg
            },
            'relationship_context': None,
            'relevant_joins': [],
            'table_count': len(tables)
        }
    
    def _serialize_relationship_context(self, context) -> Optional[Dict[str, Any]]:
        """Serialize relationship context for storage in state"""
        if not context:
            return None
        
        try:
            return {
                'strong_relationships_count': len(getattr(context, 'strong_relationships', [])),
                'join_paths_count': len(getattr(context, 'join_paths', [])),
                'concept_clusters': getattr(context, 'concept_clusters', {}),
                'has_temporal_relationships': bool(getattr(context, 'temporal_relationships', []))
            }
        except Exception:
            return {'serialization_error': True}