"""
ReAct Query Orchestrator - Main coordination for multi-stage query planning

Coordinates all ReAct agents, manages workflow progression, tracks performance,
and provides comprehensive reporting.
"""

import time
from typing import Dict, Any, List

try:
    from .state_manager import ReActQueryState, StateManager
    from .intent_recognizer import IntentRecognitionAgent
    from .schema_validator import SchemaValidationAgent
    from .relationship_explorer import RelationshipExplorerAgent
    from .error_handler import ReActErrorHandler, ContextCompressor
except ImportError:
    from state_manager import ReActQueryState, StateManager
    from intent_recognizer import IntentRecognitionAgent
    from schema_validator import SchemaValidationAgent
    from relationship_explorer import RelationshipExplorerAgent
    from error_handler import ReActErrorHandler, ContextCompressor


class TokenEfficiencyTracker:
    """Track and optimize token usage across ReAct stages"""
    
    def __init__(self):
        self.baseline_estimator = BaselineTokenEstimator()
        
    def calculate_efficiency_metrics(self, state: ReActQueryState, schema_table_count: int = 3) -> Dict[str, Any]:
        """Calculate comprehensive efficiency metrics"""
        
        # Estimate baseline tokens (traditional approach)
        baseline_tokens = self.baseline_estimator.estimate_baseline_tokens(
            state['user_query'], 
            schema_table_count
        )
        
        # Actual tokens used
        actual_tokens = state['total_tokens']
        
        # Calculate metrics
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


class BaselineTokenEstimator:
    """Estimate token usage for traditional schema-dump approach"""
    
    def estimate_baseline_tokens(self, query: str, table_count: int) -> int:
        """Estimate tokens for raw schema approach"""
        
        # Based on existing test_token_efficiency.py observations
        base_schema_tokens = 200 * table_count  # ~200 tokens per table
        query_tokens = len(query.split()) * 1.3  # Account for tokenization
        prompt_overhead = 100  # System prompts, formatting
        
        return int(base_schema_tokens + query_tokens + prompt_overhead)


class ReActQueryOrchestrator:
    """Main orchestrator for ReAct multi-stage query planning"""
    
    def __init__(self, schema_manager, kg_extractor):
        self.schema_manager = schema_manager
        self.kg_extractor = kg_extractor
        
        # Initialize agents
        self.agents = {
            'intent': IntentRecognitionAgent(),
            'schema': SchemaValidationAgent(schema_manager),
            'relationship': RelationshipExplorerAgent(kg_extractor)
        }
        
        # Supporting components
        self.state_manager = StateManager()
        self.error_handler = ReActErrorHandler()
        self.token_tracker = TokenEfficiencyTracker()
        self.context_compressor = ContextCompressor()
        
    def execute_react_planning(self, user_query: str, business_context: str = "") -> Dict[str, Any]:
        """Execute complete ReAct multi-stage query planning"""
        
        # Initialize state
        state = self.state_manager.initialize_state(user_query, business_context)
        
        print(f"🚀 Starting ReAct Query Planning for: '{user_query}'")
        print("=" * 80)
        
        # Execute stages sequentially
        stage_sequence = ['intent', 'schema', 'relationship']
        
        for stage_name in stage_sequence:
            if state['current_stage'] == 'complete':
                break
                
            print(f"\n🔄 Stage: {stage_name.upper()}")
            print("-" * 40)
            
            try:
                # Get stage-specific context
                stage_context = self._get_stage_context(stage_name, state)
                if stage_context:
                    print(f"📝 Context: {stage_context[:100]}...")
                
                # Execute agent
                agent = self.agents[stage_name]
                stage_result = agent.execute(state)
                
                # Update state
                state = self.state_manager.update_stage_result(state, stage_result)
                
                # Print results
                self._print_stage_results(stage_name, stage_result, state)
                
                # Check for skipping
                if state.get('should_skip_next', False):
                    print("⏭️  Skipping next stage (high confidence + simple case)")
                    break
                    
            except Exception as e:
                print(f"❌ Error in {stage_name} stage: {e}")
                
                # Apply error recovery
                recovery_result = self.error_handler.handle_stage_error(state, e, stage_name)
                
                if recovery_result['should_retry']:
                    print(f"🔄 Applying recovery strategy, retrying...")
                    # In a full implementation, we would retry the stage here
                    # For now, we'll use fallback data
                    if recovery_result.get('fallback_data'):
                        state = self.state_manager.update_stage_result(state, recovery_result['fallback_data'])
                else:
                    print(f"⚠️  Using fallback data")
                    # Apply fallback data to state
                    if recovery_result.get('fallback_data'):
                        state = self.state_manager.update_stage_result(state, recovery_result['fallback_data'])
        
        # Calculate final efficiency metrics
        table_count = self._get_table_count()
        efficiency_metrics = self.token_tracker.calculate_efficiency_metrics(state, table_count)
        
        # Build final result
        final_result = {
            'query': user_query,
            'business_context': business_context,
            'execution_summary': {
                'stages_completed': [s for s in stage_sequence if s != state['current_stage']],
                'total_execution_time': sum(state['execution_times']),
                'accumulated_confidence': state['accumulated_confidence'],
                'error_count': state['error_count']
            },
            'planning_results': {
                'intent_profile': state['intent_profile'],
                'validated_mapping': state['validated_mapping'], 
                'join_strategy': state['join_strategy']
            },
            'efficiency_metrics': efficiency_metrics,
            'ready_for_query_generation': state['current_stage'] == 'complete' and state['accumulated_confidence'] > 0.3
        }
        
        self._print_final_summary(final_result)
        return final_result
    
    def _get_table_count(self) -> int:
        """Get number of tables in schema"""
        try:
            if hasattr(self.schema_manager, 'schema') and self.schema_manager.schema:
                return len(self.schema_manager.schema.tables)
        except:
            pass
        return 3  # Default assumption
    
    def _get_stage_context(self, stage: str, state: ReActQueryState) -> str:
        """Get compressed context specific to current stage"""
        
        if stage == 'intent':
            return ""  # Intent stage gets no additional context
        elif stage == 'schema':
            intent = state.get('intent_profile', {})
            concepts = intent.get('target_concepts', [])
            return self.context_compressor.compress_schema_context(concepts, self.schema_manager)
        elif stage == 'relationship':
            mapping = state.get('validated_mapping', {})
            tables = mapping.get('relevant_tables', [])
            if len(tables) <= 1:
                return "Single table: no joins needed"
            return f"Target tables: {', '.join(tables)}"
        else:
            return ""
    
    def _print_stage_results(self, stage_name: str, stage_result: Dict[str, Any], state: ReActQueryState):
        """Print stage execution results"""
        
        if 'performance_metrics' in stage_result:
            metrics = stage_result['performance_metrics']
            print(f"⚡ Tokens: {metrics['total_tokens']}, Time: {metrics['execution_time']:.2f}s, Confidence: {metrics['confidence']:.2f}")
        
        if 'reasoning_chain' in stage_result:
            print(f"🧠 Reasoning: {' → '.join(stage_result['reasoning_chain'])}")
        
        # Stage-specific output
        if stage_name == 'intent' and 'intent_profile' in stage_result:
            intent = stage_result['intent_profile']
            print(f"🎯 Intent: {intent.get('action_type', 'unknown')} | Concepts: {intent.get('target_concepts', [])} | Scope: {intent.get('analysis_scope', 'unknown')}")
            
        elif stage_name == 'schema' and 'validated_mapping' in stage_result:
            mapping = stage_result['validated_mapping']
            print(f"🗂️  Tables: {mapping.get('relevant_tables', [])} | Joins: {mapping.get('joins_needed', False)}")
            
        elif stage_name == 'relationship' and 'join_strategy' in stage_result:
            strategy = stage_result['join_strategy']
            if isinstance(strategy, dict):
                strategy_type = strategy.get('strategy_type', 'unknown')
                confidence = strategy.get('path_confidence', 0)
                print(f"🔗 Join Strategy: {strategy_type} (confidence: {confidence:.2f})")
    
    def _print_final_summary(self, result: Dict[str, Any]):
        """Print comprehensive execution summary"""
        
        print("\n" + "=" * 80)
        print("📊 REACT PLANNING SUMMARY")
        print("=" * 80)
        
        exec_summary = result['execution_summary']
        efficiency = result['efficiency_metrics']
        
        print(f"\n✅ Execution Status:")
        print(f"  Stages completed: {', '.join(exec_summary['stages_completed'])}")
        print(f"  Total time: {exec_summary['total_execution_time']:.2f}s")
        print(f"  Confidence: {exec_summary['accumulated_confidence']:.2f}")
        print(f"  Errors: {exec_summary['error_count']}")
        
        print(f"\n💰 Token Efficiency:")
        print(f"  Baseline tokens: {efficiency['baseline_tokens']:,}")
        print(f"  Actual tokens: {efficiency['actual_tokens']:,}")
        print(f"  Savings: {efficiency['savings_percentage']:.1f}% ({efficiency['token_savings']:,} tokens)")
        print(f"  Efficiency ratio: {efficiency['efficiency_ratio']:.2f}")
        
        print(f"\n🎯 Planning Results:")
        intent = result['planning_results']['intent_profile']
        mapping = result['planning_results']['validated_mapping']
        
        if intent:
            print(f"  Action: {intent.get('action_type', 'unknown')}")
            print(f"  Concepts: {intent.get('target_concepts', [])}")
            
        if mapping:
            print(f"  Tables: {mapping.get('relevant_tables', [])}")
            print(f"  Joins needed: {mapping.get('joins_needed', False)}")
        
        print(f"\n🚀 Ready for query generation: {result['ready_for_query_generation']}")