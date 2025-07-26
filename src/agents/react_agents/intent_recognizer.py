"""
Intent Recognition Agent - Stage 1 of ReAct Query Planning

Extracts structured intent from natural language queries using minimal context.
Provides fallback mechanisms for robust operation.
"""

import re
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage

try:
    from .base_agent import BaseReActAgent
    from .state_manager import ReActQueryState
except ImportError:
    from base_agent import BaseReActAgent
    from state_manager import ReActQueryState


class IntentRecognitionAgent(BaseReActAgent):
    """Extract structured intent from natural language query using existing LLM"""
    
    def __init__(self):
        super().__init__()
        self.stage_name = "intent_recognition"
        
    def _generate_thought(self, state: ReActQueryState) -> str:
        """Analyze what the user is trying to accomplish"""
        query = state['user_query']
        business_context = state.get('business_context', '')
        
        return f"I need to extract the core intent from: '{query}' in context '{business_context}'"
    
    def _take_action(self, state: ReActQueryState, thought: str) -> Dict[str, Any]:
        """Use compressed prompt to extract intent (reuse existing LLM patterns)"""
        
        # Minimal context prompt (targeting 50 tokens)
        prompt = f"""Query: "{state['user_query']}"
        
Extract intent as JSON:
{{
    "action_type": "aggregation|filtering|comparison|trend_analysis|geographical_analysis",
    "target_concepts": ["concept1", "concept2"],
    "analysis_scope": "single_table|multi_table|time_series", 
    "complexity": "simple|moderate|complex",
    "confidence": 0.9
}}"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {'llm_response': response.content, 'prompt': prompt}
    
    def _make_observation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Try to extract JSON from response
            json_text = self._extract_json_from_text(action['llm_response'])
            if json_text:
                intent_data = json.loads(json_text)
            else:
                intent_data = json.loads(action['llm_response'])
            
            # Validate required fields
            required_fields = ['action_type', 'target_concepts', 'analysis_scope', 'confidence']
            if all(field in intent_data for field in required_fields):
                return {'status': 'success', 'intent_data': intent_data}
            else:
                missing_fields = [f for f in required_fields if f not in intent_data]
                return {'status': 'partial', 'intent_data': intent_data, 'missing_fields': missing_fields}
                
        except json.JSONDecodeError:
            return {'status': 'failed', 'error': 'Invalid JSON response', 'raw_response': action['llm_response']}
    
    def _synthesize_result(self, thought: str, action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured intent profile"""
        if observation['status'] == 'success':
            return {
                'intent_profile': observation['intent_data'],
                'confidence': observation['intent_data'].get('confidence', 0.5),
                'stage_status': 'completed',
                'reasoning_chain': [
                    thought, 
                    f"Extracted {len(observation['intent_data']['target_concepts'])} concepts", 
                    "Intent successfully parsed"
                ]
            }
        elif observation['status'] == 'partial':
            # Use partial data with lower confidence
            intent_data = observation['intent_data']
            # Fill missing fields with defaults
            intent_data.setdefault('action_type', 'general_analysis')
            intent_data.setdefault('target_concepts', self._extract_basic_concepts(thought))
            intent_data.setdefault('analysis_scope', 'unknown')
            intent_data.setdefault('confidence', 0.4)
            
            return {
                'intent_profile': intent_data,
                'confidence': 0.4,
                'stage_status': 'completed_with_partial_data',
                'reasoning_chain': [
                    thought, 
                    f"Partial parsing: missing {observation.get('missing_fields', [])}", 
                    "Using partial intent with defaults"
                ]
            }
        else:
            # Fallback to rule-based intent extraction
            fallback_intent = self._generate_fallback_intent(thought)
            return {
                'intent_profile': fallback_intent,
                'confidence': 0.3,
                'stage_status': 'completed_with_fallback',
                'reasoning_chain': [
                    thought, 
                    f"LLM parsing failed: {observation.get('error', 'unknown')}", 
                    "Using rule-based fallback intent"
                ]
            }
    
    def _generate_fallback_intent(self, thought: str) -> Dict[str, Any]:
        """Generate basic intent using rule-based approach when LLM fails"""
        # Extract query from thought
        query_start = thought.find("'") + 1
        query_end = thought.find("'", query_start)
        query = thought[query_start:query_end] if query_start > 0 and query_end > query_start else ""
        
        return {
            'action_type': self._detect_action_type(query),
            'target_concepts': self._extract_basic_concepts(query),
            'analysis_scope': self._detect_analysis_scope(query),
            'complexity': 'moderate',
            'confidence': 0.3
        }
    
    def _detect_action_type(self, query: str) -> str:
        """Detect action type using keyword matching"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sum', 'total', 'count', 'average', 'group', 'aggregate']):
            return 'aggregation'
        elif any(word in query_lower for word in ['location', 'city', 'state', 'country', 'region', 'geography']):
            return 'geographical_analysis'
        elif any(word in query_lower for word in ['time', 'date', 'trend', 'over time', 'month', 'year', 'temporal']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'between', 'comparison']):
            return 'comparison'
        elif any(word in query_lower for word in ['filter', 'where', 'select', 'find']):
            return 'filtering'
        else:
            return 'general_analysis'
    
    def _extract_basic_concepts(self, query: str) -> List[str]:
        """Extract basic concepts using simple NLP"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'show', 'find', 'what', 'where', 'when', 'how', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words that might be concepts
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        concepts = []
        
        # Business domain concepts
        business_concepts = {
            'customer', 'customers', 'client', 'clients',
            'order', 'orders', 'purchase', 'purchases', 'sale', 'sales',
            'product', 'products', 'item', 'items', 'goods',
            'price', 'prices', 'cost', 'costs', 'revenue', 'profit',
            'location', 'locations', 'city', 'cities', 'region', 'regions',
            'date', 'dates', 'time', 'period', 'month', 'year',
            'category', 'categories', 'type', 'types', 'brand', 'brands'
        }
        
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                (word in business_concepts or any(bc in word for bc in business_concepts))):
                concepts.append(word)
        
        # Return top 3 concepts
        return concepts[:3] if concepts else ['data']
    
    def _detect_analysis_scope(self, query: str) -> str:
        """Detect if analysis involves single or multiple tables"""
        query_lower = query.lower()
        
        # Indicators of multi-table queries
        multi_table_indicators = [
            'join', 'with', 'and', 'by', 'from', 'across',
            'customer order', 'order item', 'product sale',
            'location sales', 'category revenue'
        ]
        
        if any(indicator in query_lower for indicator in multi_table_indicators):
            return 'multi_table'
        elif any(word in query_lower for word in ['time', 'date', 'trend', 'over']):
            return 'time_series'
        else:
            return 'single_table'