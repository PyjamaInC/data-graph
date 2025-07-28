"""
Data Exploration ReAct Agent

Implements iterative data exploration using pandas operations with the ReAct pattern.
Discovers insights by reasoning about data and executing appropriate analyses.
"""

import pandas as pd
import numpy as np
import json
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage

from .base_agent import BaseReActAgent
from .state_manager import ReActQueryState


@dataclass
class ExplorationState:
    """State for data exploration with accumulated findings"""
    user_question: str
    tables: Dict[str, pd.DataFrame]
    table_profiles: Dict[str, Any] = field(default_factory=dict)
    exploration_history: List[Dict[str, Any]] = field(default_factory=list)
    current_findings: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    max_iterations: int = 8
    iteration_count: int = 0
    available_columns: Dict[str, List[str]] = field(default_factory=dict)
    data_types: Dict[str, Dict[str, str]] = field(default_factory=dict)


class DataExplorationToolkit:
    """Toolkit of pandas operations the agent can execute"""
    
    @staticmethod
    def validate_operation(operation: str, tables: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        """Validate operation before execution"""
        # Check for dangerous operations
        dangerous_patterns = ['exec', 'eval', '__', 'import', 'open', 'file', 'os.', 'sys.']
        for pattern in dangerous_patterns:
            if pattern in operation and pattern not in ['eval']:  # Allow eval for pandas operations
                return False, f"Operation contains potentially dangerous pattern: {pattern}"
        
        # Check if referenced tables exist
        for table_name in tables.keys():
            if table_name in operation or f"tables['{table_name}']" in operation:
                return True, "Operation appears valid"
        
        return True, "Operation validation passed"
    
    @staticmethod
    def execute_pandas_operation(code: str, tables: Dict[str, pd.DataFrame], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safely execute pandas code and return results"""
        
        # Validate operation first
        is_valid, validation_msg = DataExplorationToolkit.validate_operation(code, tables)
        if not is_valid:
            return {
                'success': False,
                'error': {'type': 'ValidationError', 'message': validation_msg},
                'code': code
            }
        
        # Create safe execution environment
        safe_globals = {
            'pd': pd,
            'np': np,
            'tables': tables,
            **tables,  # Make tables directly accessible
            **(context or {})
        }
        
        # Add common pandas functions to namespace
        safe_globals.update({
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            'concat': pd.concat,
            'merge': pd.merge,
            'pivot_table': pd.pivot_table,
            'to_datetime': pd.to_datetime,
        })
        
        result = {'success': False, 'output': None, 'error': None, 'code': code}
        
        try:
            # Handle multi-line operations vs single expressions
            if '\n' in code.strip() or '=' in code and not any(op in code for op in ['==', '!=', '<=', '>=']):
                # Multi-line or assignment operation - use exec
                local_namespace = {}
                exec(code, safe_globals, local_namespace)
                
                # Try to find the result from the last line or variable assignment
                exec_result = None
                lines = [line.strip() for line in code.strip().split('\n') if line.strip()]
                
                if lines:
                    last_line = lines[-1]
                    # If last line is a variable name, get its value
                    if last_line in local_namespace:
                        exec_result = local_namespace[last_line]
                    else:
                        # Try to evaluate the last line as an expression
                        try:
                            exec_result = eval(last_line, safe_globals, local_namespace)
                        except:
                            # If that fails, just return the entire namespace
                            exec_result = local_namespace
            else:
                # Single expression - use eval
                exec_result = eval(code, safe_globals, {})
            
            # Convert result to serializable format
            if isinstance(exec_result, pd.DataFrame):
                result['output'] = DataExplorationToolkit._serialize_dataframe(exec_result)
            elif isinstance(exec_result, pd.Series):
                result['output'] = DataExplorationToolkit._serialize_series(exec_result)
            elif isinstance(exec_result, (int, float, str, bool, type(None))):
                result['output'] = {'type': 'scalar', 'value': exec_result}
            elif isinstance(exec_result, (list, dict)):
                result['output'] = {'type': 'collection', 'data': exec_result}
            elif isinstance(exec_result, np.ndarray):
                result['output'] = {'type': 'array', 'data': exec_result.tolist(), 'shape': exec_result.shape}
            else:
                result['output'] = {'type': 'other', 'value': str(exec_result)}
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            
        return result
    
    @staticmethod
    def _serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Serialize DataFrame for JSON output"""
        if len(df) > 20:  # Limit output size
            return {
                'type': 'dataframe',
                'shape': df.shape,
                'head': df.head(10).to_dict('records'),
                'tail': df.tail(5).to_dict('records'),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'summary': f"Showing first 10 and last 5 rows of {len(df)} total rows"
            }
        else:
            return {
                'type': 'dataframe',
                'shape': df.shape,
                'data': df.to_dict('records'),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
    
    @staticmethod
    def _serialize_series(series: pd.Series) -> Dict[str, Any]:
        """Serialize Series for JSON output"""
        if len(series) > 20:
            return {
                'type': 'series',
                'length': len(series),
                'head': series.head(10).to_dict(),
                'tail': series.tail(5).to_dict(),
                'name': series.name,
                'dtype': str(series.dtype),
                'summary': f"Showing first 10 and last 5 values of {len(series)} total"
            }
        else:
            return {
                'type': 'series',
                'data': series.to_dict(),
                'name': series.name,
                'dtype': str(series.dtype)
            }
    
    @staticmethod
    def suggest_contextual_operations(state: ExplorationState, question_type: str) -> List[str]:
        """Suggest operations based on current exploration context"""
        suggestions = []
        tables = state.tables
        
        # Use available columns info
        for table_name, columns in state.available_columns.items():
            if 'date' in question_type.lower() or 'time' in question_type.lower():
                date_cols = [col for col in columns if any(d in col.lower() for d in ['date', 'time', 'timestamp'])]
                if date_cols:
                    suggestions.append(f"tables['{table_name}']['{date_cols[0]}'].describe()")
                    suggestions.append(f"tables['{table_name}'].groupby(pd.Grouper(key='{date_cols[0]}', freq='M')).size()")
            
            if 'correlation' in question_type.lower():
                numeric_cols = [col for col, dtype in state.data_types.get(table_name, {}).items() 
                               if 'int' in dtype or 'float' in dtype]
                if len(numeric_cols) > 1:
                    suggestions.append(f"tables['{table_name}'][{numeric_cols[:5]}].corr()")
        
        # General exploration based on findings
        if not state.current_findings:
            # Initial exploration
            for table in tables.keys():
                suggestions.extend([
                    f"tables['{table}'].shape",
                    f"tables['{table}'].info()",
                    f"tables['{table}'].describe(include='all')"
                ])
        else:
            # Deeper exploration based on previous findings
            for table in tables.keys():
                suggestions.extend([
                    f"tables['{table}'].isnull().sum()",
                    f"tables['{table}'].nunique()",
                    f"tables['{table}'].select_dtypes(include=[np.number]).quantile([0.25, 0.5, 0.75])"
                ])
        
        return suggestions[:10]  # Limit suggestions


class DataExplorationReActAgent(BaseReActAgent):
    """ReAct agent for iterative data exploration and insight discovery"""
    
    def __init__(self, 
                 enhanced_intelligence: Optional[Any] = None,
                 semantic_graph_builder: Optional[Any] = None,
                 llm_model: str = "gpt-4"):
        super().__init__(llm_model)
        
        self.enhanced_intelligence = enhanced_intelligence
        self.semantic_graph_builder = semantic_graph_builder
        self.toolkit = DataExplorationToolkit()
        self._current_exploration_state: Optional[ExplorationState] = None
        
        # Common pandas operations templates
        self.operation_templates = {
            'basic_info': "tables['{table}'].info()",
            'shape': "tables['{table}'].shape",
            'columns': "tables['{table}'].columns.tolist()",
            'describe': "tables['{table}'].describe(include='all')",
            'null_counts': "tables['{table}'].isnull().sum()",
            'value_counts': "tables['{table}']['{column}'].value_counts()",
            'correlation': "tables['{table}'].select_dtypes(include=[np.number]).corr()",
            'groupby_count': "tables['{table}'].groupby('{column}').size()",
            'groupby_agg': "tables['{table}'].groupby('{group_col}')['{agg_col}'].agg(['count', 'mean', 'sum'])",
        }
    
    def explore_for_insights(self, user_question: str, 
                           tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Main method to explore data and discover insights"""
        
        # Initialize exploration state
        self._current_exploration_state = ExplorationState(
            user_question=user_question,
            tables=tables
        )
        
        # Extract column information
        for name, df in tables.items():
            self._current_exploration_state.available_columns[name] = list(df.columns)
            self._current_exploration_state.data_types[name] = {
                col: str(dtype) for col, dtype in df.dtypes.items()
            }
        
        # Generate table profiles if intelligence layer available
        if self.enhanced_intelligence:
            for name, df in tables.items():
                try:
                    profile = self.enhanced_intelligence.analyze_table_comprehensive(name, df)
                    self._current_exploration_state.table_profiles[name] = profile
                except Exception as e:
                    print(f"Warning: Failed to profile table {name}: {e}")
        
        print(f"🔍 Starting data exploration for: '{user_question}'")
        print("=" * 80)
        
        # Iterative exploration loop
        while (self._current_exploration_state.iteration_count < self._current_exploration_state.max_iterations and 
               self._current_exploration_state.confidence_level < 0.8):
            
            self._current_exploration_state.iteration_count += 1
            print(f"\n🔄 Exploration Iteration {self._current_exploration_state.iteration_count}")
            
            # Execute ReAct cycle
            cycle_result = self._execute_exploration_cycle()
            
            if cycle_result.get('sufficient_insights', False):
                break
        
        # Generate final insights report
        final_insights = self._generate_final_insights()
        
        return {
            'user_question': user_question,
            'exploration_summary': {
                'iterations_used': self._current_exploration_state.iteration_count,
                'confidence_level': self._current_exploration_state.confidence_level,
                'total_findings': len(self._current_exploration_state.current_findings)
            },
            'insights': final_insights,
            'exploration_history': self._current_exploration_state.exploration_history,
            'recommendations': self._generate_recommendations()
        }
    
    def _execute_exploration_cycle(self) -> Dict[str, Any]:
        """Execute one ReAct exploration cycle"""
        
        # Convert to ReActQueryState for base class compatibility
        react_state: ReActQueryState = {
            'user_query': self._current_exploration_state.user_question,
            'business_context': '',
            'timestamp': 0.0,
            'intent_profile': {},
            'validated_mapping': {},
            'join_strategy': {},
            'current_stage': 'intent',
            'stage_confidence': self._current_exploration_state.confidence_level,
            'accumulated_confidence': self._current_exploration_state.confidence_level,
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
        
        # Execute base ReAct pattern
        result = self.execute(react_state)
        
        # Extract findings and update exploration state
        if 'findings' in result:
            self._current_exploration_state.current_findings.extend(result['findings'])
        
        if 'confidence' in result:
            self._current_exploration_state.confidence_level = result['confidence']
        
        # Record in history
        self._current_exploration_state.exploration_history.append({
            'iteration': self._current_exploration_state.iteration_count,
            'action': result.get('action', {}),
            'observation': result.get('observation', {}),
            'findings': result.get('findings', [])
        })
        
        return result
    
    def _generate_thought(self, state: ReActQueryState) -> str:
        """Generate reasoning about what to explore next"""
        
        if not self._current_exploration_state:
            return "I need to understand the data structure first."
        
        exploration_state = self._current_exploration_state
        iteration = exploration_state.iteration_count
        findings = exploration_state.current_findings[-3:] if exploration_state.current_findings else []
        
        # Build context for thought generation
        context = f"""
Iteration: {iteration}
User Question: "{state['user_query']}"
Available Tables: {list(exploration_state.tables.keys())}
Recent Findings: {findings}
Current Confidence: {exploration_state.confidence_level:.2f}
"""
        
        # Generate contextual thought
        if iteration == 1:
            thought = f"This is my first exploration. I need to understand the basic structure of the {len(exploration_state.tables)} available tables. Let me start with overview information."
        elif exploration_state.confidence_level < 0.3:
            thought = f"I still need more information to answer the question. Based on {findings}, I should explore specific columns or relationships."
        elif exploration_state.confidence_level < 0.7:
            thought = f"I'm making progress. The findings show {findings}. Now I need to dig deeper into specific patterns related to the question."
        else:
            thought = f"I have good understanding now. Let me verify my findings and ensure I can fully answer: '{state['user_query']}'"
        
        return thought
    
    def _take_action(self, state: ReActQueryState, thought: str) -> Dict[str, Any]:
        """Decide what pandas operation to execute"""
        
        exploration_state = self._current_exploration_state
        if not exploration_state:
            return {'operation': 'tables.keys()', 'reasoning': thought}
        
        # Get contextual suggestions
        suggestions = self.toolkit.suggest_contextual_operations(
            exploration_state, 
            state['user_query']
        )
        
        # Create prompt for the LLM
        prompt = f"""
Based on this exploration context, generate ONE pandas operation to execute:

USER QUESTION: "{state['user_query']}"
CURRENT THOUGHT: {thought}

AVAILABLE TABLES AND COLUMNS:
{json.dumps(exploration_state.available_columns, indent=2)}

DATA TYPES:
{json.dumps(exploration_state.data_types, indent=2)}

RECENT FINDINGS: {exploration_state.current_findings[-3:]}

SUGGESTED OPERATIONS:
{chr(10).join(f"- {s}" for s in suggestions[:5])}

Generate a SINGLE pandas operation that will help answer the user's question.
The operation must be valid Python code that can be executed with eval().
Use the format: tables['table_name'] to access tables.

PANDAS OPERATION:
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        operation = response.content.strip()
        
        # Clean up the operation
        if '```python' in operation:
            operation = operation.split('```python')[1].split('```')[0].strip()
        elif '```' in operation:
            operation = operation.split('```')[1].strip()
        
        # Ensure operation is a single line
        operation = operation.split('\n')[0].strip()
        
        return {
            'operation': operation,
            'reasoning': thought,
            'suggestions_provided': suggestions[:3]
        }
    
    def _make_observation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pandas operation and observe results"""
        
        operation = action['operation']
        exploration_state = self._current_exploration_state
        
        if not exploration_state:
            return {'error': True, 'message': 'No exploration state available'}
        
        print(f"🔧 Executing: {operation}")
        
        # Execute the operation
        execution_result = self.toolkit.execute_pandas_operation(
            operation, 
            exploration_state.tables
        )
        
        observation = {
            'operation_executed': operation,
            'execution_success': execution_result['success'],
            'result_summary': self._summarize_result(execution_result),
            'raw_result': execution_result
        }
        
        if execution_result['success']:
            print(f"✅ Success: {observation['result_summary']}")
        else:
            print(f"❌ Error: {execution_result['error']['message']}")
            
        return observation
    
    def _synthesize_result(self, thought: str, action: Dict[str, Any], 
                          observation: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings and determine next steps"""
        
        exploration_state = self._current_exploration_state
        
        # Create synthesis prompt
        result_detail = ""
        if observation['execution_success'] and 'raw_result' in observation:
            output = observation['raw_result'].get('output', {})
            if output.get('type') == 'dataframe':
                result_detail = f"DataFrame with shape {output.get('shape', 'unknown')}"
                if 'head' in output:
                    result_detail += f"\nFirst few rows: {json.dumps(output['head'][:3], indent=2)}"
            elif output.get('type') == 'series':
                result_detail = f"Series with {len(output.get('data', {}))} values"
                if 'data' in output:
                    sample = dict(list(output['data'].items())[:5])
                    result_detail += f"\nSample: {json.dumps(sample, indent=2)}"
            else:
                result_detail = json.dumps(output, indent=2)
        
        prompt = f"""
Analyze this data exploration step and extract insights:

THOUGHT: {thought}
ACTION: {action['operation']}
RESULT: {result_detail}
USER QUESTION: "{exploration_state.user_question if exploration_state else ''}"

Based on this exploration step:
1. What specific insights or findings can be extracted?
2. How confident are you that these findings help answer the user's question (0.0-1.0)?
3. What should be explored next to better answer the question?
4. Do we have sufficient insights to answer the original question?

Respond in JSON format:
{{
    "findings": ["specific finding 1", "specific finding 2"],
    "confidence": 0.6,
    "next_exploration_ideas": ["next step 1", "next step 2"],
    "sufficient_insights": false,
    "reasoning": "explanation of why these findings are relevant"
}}
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Try to parse JSON response
            content = response.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            result = json.loads(content)
        except Exception as e:
            # Fallback parsing
            result = {
                "findings": [f"Executed {action['operation']} successfully"],
                "confidence": 0.5,
                "next_exploration_ideas": ["Continue exploration"],
                "sufficient_insights": False,
                "reasoning": f"Failed to parse LLM response: {str(e)}"
            }
        
        # Update result with ReAct components
        result.update({
            'thought': thought,
            'action': action,
            'observation': observation,
            'stage_name': 'data_exploration'
        })
        
        return result
    
    def _summarize_result(self, execution_result: Dict[str, Any]) -> str:
        """Create human-readable summary of execution result"""
        
        if not execution_result['success']:
            return f"Error: {execution_result['error']['message']}"
        
        output = execution_result['output']
        
        if output['type'] == 'dataframe':
            shape = output['shape']
            cols = len(output.get('columns', []))
            return f"DataFrame with {shape[0]} rows and {cols} columns"
        elif output['type'] == 'series':
            length = len(output.get('data', {}))
            return f"Series '{output.get('name', 'unnamed')}' with {length} values"
        elif output['type'] == 'scalar':
            return f"Value: {output['value']}"
        elif output['type'] == 'collection':
            data = output['data']
            if isinstance(data, dict):
                return f"Dictionary with {len(data)} items: {list(data.keys())[:5]}"
            elif isinstance(data, list):
                return f"List with {len(data)} items"
        elif output['type'] == 'array':
            return f"Array with shape {output.get('shape', 'unknown')}"
        
        return "Operation completed successfully"
    
    def _generate_final_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights from all exploration"""
        
        exploration_state = self._current_exploration_state
        if not exploration_state:
            return {'error': 'No exploration state available'}
        
        # Summarize all findings
        all_findings = '\n'.join(f"- {f}" for f in exploration_state.current_findings)
        
        prompt = f"""
Based on this complete data exploration session, provide comprehensive insights:

ORIGINAL QUESTION: "{exploration_state.user_question}"

ALL FINDINGS FROM EXPLORATION:
{all_findings}

TABLES EXPLORED: {list(exploration_state.tables.keys())}
ITERATIONS COMPLETED: {exploration_state.iteration_count}

Generate a comprehensive analysis that directly answers the user's question.
Include:
1. Direct answer to the question
2. Key insights discovered from the data
3. Supporting evidence (specific numbers, patterns, etc.)
4. Any data quality issues noticed
5. Recommendations for further analysis

Respond in JSON format:
{{
    "direct_answer": "Clear answer to the user's question",
    "key_insights": ["insight 1 with specific details", "insight 2"],
    "supporting_evidence": ["specific evidence 1", "evidence 2"],
    "data_patterns": ["pattern 1", "pattern 2"],
    "data_quality_notes": ["quality observation 1"],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "confidence_score": 0.8
}}
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            content = response.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            insights = json.loads(content)
        except:
            # Fallback insights
            insights = {
                "direct_answer": f"Completed {exploration_state.iteration_count} exploration iterations",
                "key_insights": exploration_state.current_findings[:5],
                "supporting_evidence": ["See exploration history for details"],
                "data_patterns": ["Patterns identified through exploration"],
                "data_quality_notes": ["Data quality assessed during exploration"],
                "recommendations": ["Review exploration history for detailed findings"],
                "confidence_score": exploration_state.confidence_level
            }
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for further analysis"""
        
        exploration_state = self._current_exploration_state
        if not exploration_state:
            return ["Unable to generate recommendations without exploration state"]
        
        recommendations = []
        
        # Based on confidence level
        if exploration_state.confidence_level < 0.7:
            recommendations.append("Consider more targeted exploration with specific column analysis")
        
        # Based on iterations
        if exploration_state.iteration_count >= exploration_state.max_iterations:
            recommendations.append("Break down the question into smaller, more specific queries")
        
        # Based on findings
        if len(exploration_state.current_findings) < 3:
            recommendations.append("Explore relationships between tables using merge operations")
        
        # Based on data characteristics
        if exploration_state.available_columns:
            total_cols = sum(len(cols) for cols in exploration_state.available_columns.values())
            if total_cols > 50:
                recommendations.append("Focus on specific columns most relevant to your question")
        
        # General recommendations
        recommendations.extend([
            "Validate findings with domain knowledge",
            "Consider creating visualizations for better insight communication",
            "Document analysis steps for reproducibility"
        ])
        
        return recommendations[:5]  # Limit to top 5 recommendations