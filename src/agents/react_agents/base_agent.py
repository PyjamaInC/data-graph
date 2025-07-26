"""
BaseReActAgent: Foundation class for all ReAct agents

Provides common functionality for token tracking, confidence scoring,
and the ReAct pattern execution (Thought -> Action -> Observation -> Result).
"""

import time
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class BaseReActAgent(ABC):
    """Base class for ReAct agents with token tracking and confidence scoring"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.stage_name = self.__class__.__name__
        
    def execute(self, state: 'ReActQueryState') -> Dict[str, Any]:
        """Execute agent with ReAct pattern: Thought -> Action -> Observation -> Result"""
        start_time = time.time()
        input_tokens = self._estimate_input_tokens(state)
        
        try:
            # ReAct execution pattern
            thought = self._generate_thought(state)
            action = self._take_action(state, thought)
            observation = self._make_observation(action)
            result = self._synthesize_result(thought, action, observation)
            
            # Track performance
            execution_time = time.time() - start_time
            output_tokens = self._estimate_output_tokens(result)
            
            return {
                **result,
                'performance_metrics': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'execution_time': execution_time,
                    'confidence': result.get('confidence', 0.5)
                }
            }
        except Exception as e:
            return self._handle_error(state, e)
    
    @abstractmethod
    def _generate_thought(self, state: 'ReActQueryState') -> str:
        """Generate reasoning thought for current state"""
        pass
    
    @abstractmethod
    def _take_action(self, state: 'ReActQueryState', thought: str) -> Dict[str, Any]:
        """Take action based on thought and state"""
        pass
    
    @abstractmethod
    def _make_observation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Make observation about action results"""
        pass
    
    @abstractmethod
    def _synthesize_result(self, thought: str, action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final result from ReAct components"""
        pass
    
    def _estimate_input_tokens(self, state: 'ReActQueryState') -> int:
        """Estimate token count for input state"""
        # Simple estimation based on string length
        input_text = str(state.get('user_query', ''))
        if state.get('current_stage') == 'schema':
            input_text += str(state.get('intent_profile', {}))
        elif state.get('current_stage') == 'relationship':
            input_text += str(state.get('validated_mapping', {}))
        
        # Rough approximation: 1 token ≈ 4 characters
        return len(input_text) // 4
    
    def _estimate_output_tokens(self, result: Dict[str, Any]) -> int:
        """Estimate token count for output result"""
        # Simple estimation based on string length of result
        output_text = str(result)
        return len(output_text) // 4
    
    def _handle_error(self, state: 'ReActQueryState', error: Exception) -> Dict[str, Any]:
        """Handle errors gracefully with fallback result"""
        return {
            'error': True,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stage_status': 'failed',
            'confidence': 0.1,
            'performance_metrics': {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'execution_time': 0,
                'confidence': 0.1
            }
        }
    
    def _safe_json_parse(self, text: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safely parse JSON with fallback"""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return fallback or {}
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from text that might contain additional content"""
        # Find JSON-like content between braces
        start = text.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
        
        return None