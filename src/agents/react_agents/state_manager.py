"""
State Management for ReAct Query Planning

Manages workflow state with performance tracking, confidence accumulation,
and stage progression logic.
"""

import time
from typing import TypedDict, Literal, List, Dict, Any, Optional


class ReActQueryState(TypedDict):
    """Type-safe state definition for ReAct workflow"""
    # Input
    user_query: str
    business_context: str
    timestamp: float
    
    # Stage outputs
    intent_profile: Dict[str, Any]
    validated_mapping: Dict[str, Any]
    join_strategy: Dict[str, Any]
    
    # Workflow control
    current_stage: Literal["intent", "schema", "relationship", "complete"]
    stage_confidence: float
    accumulated_confidence: float
    should_skip_next: bool
    
    # Performance tracking
    tokens_per_stage: List[int]
    total_tokens: int
    baseline_tokens: int  # For comparison
    efficiency_ratio: float
    execution_times: List[float]
    
    # Error handling
    error_count: int
    last_error: str
    recovery_attempts: int


class StateManager:
    """Manage ReAct workflow state with performance tracking"""
    
    def __init__(self):
        self.current_state: Optional[ReActQueryState] = None
        
    def initialize_state(self, user_query: str, business_context: str = "") -> ReActQueryState:
        """Initialize fresh state for new query"""
        state = {
            'user_query': user_query,
            'business_context': business_context,
            'timestamp': time.time(),
            
            'intent_profile': {},
            'validated_mapping': {},
            'join_strategy': {},
            
            'current_stage': 'intent',
            'stage_confidence': 0.0,
            'accumulated_confidence': 0.0,
            'should_skip_next': False,
            
            'tokens_per_stage': [],
            'total_tokens': 0,
            'baseline_tokens': 0,
            'efficiency_ratio': 0.0,
            'execution_times': [],
            
            'error_count': 0,
            'last_error': '',
            'recovery_attempts': 0
        }
        
        self.current_state = state
        return state
    
    def update_stage_result(self, state: ReActQueryState, stage_result: Dict[str, Any]) -> ReActQueryState:
        """Update state with stage results and performance metrics"""
        stage_name = state['current_stage']
        
        # Update stage-specific data
        if stage_name == 'intent':
            state['intent_profile'] = stage_result.get('intent_profile', {})
        elif stage_name == 'schema':
            state['validated_mapping'] = stage_result.get('validated_mapping', {})
        elif stage_name == 'relationship':
            state['join_strategy'] = stage_result.get('join_strategy', {})
        
        # Update performance metrics
        if 'performance_metrics' in stage_result:
            metrics = stage_result['performance_metrics']
            state['tokens_per_stage'].append(metrics.get('total_tokens', 0))
            state['total_tokens'] += metrics.get('total_tokens', 0)
            state['execution_times'].append(metrics.get('execution_time', 0))
            state['stage_confidence'] = metrics.get('confidence', 0.5)
        
        # Update accumulated confidence
        if state['stage_confidence'] > 0:
            if state['accumulated_confidence'] == 0:
                state['accumulated_confidence'] = state['stage_confidence']
            else:
                # Geometric mean for accumulated confidence
                state['accumulated_confidence'] = (state['accumulated_confidence'] * state['stage_confidence']) ** 0.5
        
        # Update error tracking if stage failed
        if stage_result.get('error', False):
            state['error_count'] += 1
            state['last_error'] = stage_result.get('error_message', 'Unknown error')
        
        # Determine next stage
        state = self._advance_stage(state)
        
        return state
    
    def _advance_stage(self, state: ReActQueryState) -> ReActQueryState:
        """Advance to next stage with conditional skipping logic"""
        current = state['current_stage']
        
        if current == 'intent':
            state['current_stage'] = 'schema'
        elif current == 'schema':
            # Skip relationship stage if high confidence + no joins needed
            if (state['stage_confidence'] > 0.95 and 
                state['validated_mapping'].get('joins_needed', False) == False):
                state['should_skip_next'] = True
                state['current_stage'] = 'complete'
            else:
                state['current_stage'] = 'relationship'
        elif current == 'relationship':
            state['current_stage'] = 'complete'
        
        return state
    
    def get_stage_summary(self, state: ReActQueryState) -> Dict[str, Any]:
        """Get summary of current stage execution"""
        return {
            'current_stage': state['current_stage'],
            'stage_confidence': state['stage_confidence'],
            'accumulated_confidence': state['accumulated_confidence'],
            'total_tokens': state['total_tokens'],
            'total_time': sum(state['execution_times']),
            'stages_completed': len(state['tokens_per_stage']),
            'error_count': state['error_count']
        }
    
    def should_continue(self, state: ReActQueryState) -> bool:
        """Determine if workflow should continue"""
        return (
            state['current_stage'] != 'complete' and
            state['error_count'] < 3 and  # Max 3 errors
            state['total_tokens'] < 1000   # Token budget
        )
    
    def calculate_efficiency_metrics(self, state: ReActQueryState, baseline_tokens: int) -> Dict[str, Any]:
        """Calculate comprehensive efficiency metrics"""
        actual_tokens = state['total_tokens']
        
        if baseline_tokens > 0:
            efficiency_ratio = actual_tokens / baseline_tokens
            token_savings = baseline_tokens - actual_tokens
            savings_percentage = (token_savings / baseline_tokens) * 100
        else:
            efficiency_ratio = 1.0
            token_savings = 0
            savings_percentage = 0
        
        return {
            'baseline_tokens': baseline_tokens,
            'actual_tokens': actual_tokens,
            'token_savings': token_savings,
            'savings_percentage': savings_percentage,
            'efficiency_ratio': efficiency_ratio,
            'tokens_per_stage': state['tokens_per_stage'],
            'average_tokens_per_stage': sum(state['tokens_per_stage']) / len(state['tokens_per_stage']) if state['tokens_per_stage'] else 0
        }