"""
Schema Validation Agent - Stage 2 of ReAct Query Planning

Maps intent concepts to actual schema entities using SchemaManager capabilities.
Provides semantic role-based matching and confidence scoring.
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage

try:
    from .base_agent import BaseReActAgent
    from .state_manager import ReActQueryState
except ImportError:
    pass

# Fallback for testing
from enum import Enum
class SemanticRole(Enum):
    IDENTIFIER = "identifier"
    MEASURE = "measure"
    DIMENSION = "dimension"
    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    DESCRIPTIVE = "descriptive"


class SchemaValidationAgent(BaseReActAgent):
    """Map intent concepts to actual schema entities using SchemaManager"""
    
    def __init__(self, schema_manager):
        super().__init__()
        self.schema_manager = schema_manager
        self.stage_name = "schema_validation"
        
    def _generate_thought(self, state: ReActQueryState) -> str:
        """Plan how to map concepts to schema"""
        intent = state['intent_profile']
        concepts = intent.get('target_concepts', [])
        action_type = intent.get('action_type', 'unknown')
        return f"Need to map concepts {concepts} to schema entities, focusing on {action_type}"
    
    def _take_action(self, state: ReActQueryState, thought: str) -> Dict[str, Any]:
        """Use LLM-enhanced schema mapping for intelligent concept matching"""
        intent = state['intent_profile']
        concepts = intent.get('target_concepts', [])
        action_type = intent.get('action_type', 'general_analysis')
        
        # First, try rule-based matching for exact matches
        relevant_tables = []
        concept_mappings = {}
        required_roles = self._determine_required_roles(action_type)
        
        # Search through schema for matching entities
        for concept in concepts:
            matches = self._find_concept_matches(concept, required_roles)
            
            # If no good matches found, use LLM for semantic matching
            if not matches or (matches and matches[0]['confidence'] < 0.8):  # Increased threshold to 0.8
                print(f"   🧠 Using LLM semantic matching for '{concept}' (current best: {matches[0]['confidence'] if matches else 'none'})")
                llm_matches = self._llm_semantic_concept_matching(concept, action_type, state)
                if llm_matches:
                    print(f"   ✅ LLM found {len(llm_matches)} semantic matches")
                    matches.extend(llm_matches)
                    # Re-sort by confidence
                    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)[:5]
                else:
                    print(f"   ⚠️ LLM semantic matching returned no results")
            
            concept_mappings[concept] = matches
            
            # Collect relevant tables
            for match in matches:
                table = match['table']
                if table not in relevant_tables:
                    relevant_tables.append(table)
        
        # Determine if joins are needed
        joins_needed = len(relevant_tables) > 1
        
        return {
            'concept_mappings': concept_mappings,
            'relevant_tables': relevant_tables,
            'joins_needed': joins_needed,
            'required_roles': [role.value for role in required_roles],
            'mapping_confidence': self._calculate_mapping_confidence(concept_mappings)
        }
    
    def _make_observation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and assess the quality of schema mappings"""
        concept_mappings = action['concept_mappings']
        relevant_tables = action['relevant_tables']
        
        # Check mapping quality
        total_concepts = len(concept_mappings)
        mapped_concepts = sum(1 for matches in concept_mappings.values() if matches)
        
        if mapped_concepts == 0:
            return {
                'status': 'failed',
                'error': 'No schema mappings found for any concepts',
                'coverage': 0.0
            }
        elif mapped_concepts < total_concepts:
            return {
                'status': 'partial',
                'coverage': mapped_concepts / total_concepts,
                'missing_concepts': [concept for concept, matches in concept_mappings.items() if not matches]
            }
        else:
            return {
                'status': 'success',
                'coverage': 1.0,
                'table_count': len(relevant_tables)
            }
    
    def _synthesize_result(self, thought: str, action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured validation result"""
        
        if observation['status'] == 'success':
            confidence = action['mapping_confidence']
        elif observation['status'] == 'partial':
            confidence = action['mapping_confidence'] * observation['coverage']
        else:
            # Generate fallback mapping
            action = self._generate_fallback_mapping(thought)
            confidence = 0.3
        
        return {
            'validated_mapping': {
                'relevant_tables': action['relevant_tables'],
                'concept_mappings': action['concept_mappings'],
                'joins_needed': action['joins_needed'],
                'required_roles': action['required_roles'],
                'mapping_confidence': confidence
            },
            'confidence': confidence,
            'stage_status': observation['status'],
            'reasoning_chain': [
                thought,
                f"Found {len(action['relevant_tables'])} relevant tables",
                f"Mapped {observation.get('coverage', 0)} of concepts successfully"
            ]
        }
    
    def _find_concept_matches(self, concept: str, required_roles: List) -> List[Dict[str, Any]]:
        """Find schema entities matching the concept"""
        matches = []
        
        if not hasattr(self.schema_manager, 'schema') or not self.schema_manager.schema:
            return matches
        
        # First, check for table name matches (very important!)
        for table_name, table_schema in self.schema_manager.schema.tables.items():
            # Table name exact match
            if concept.lower() == table_name.lower():
                # Find the best representative column for this table
                best_col = self._find_best_column_for_table(table_schema, required_roles)
                matches.append({
                    'table': table_name,
                    'column': best_col['name'],
                    'semantic_role': best_col['role'],
                    'match_type': 'table_exact_match',
                    'confidence': 0.95
                })
            # Table name partial match (e.g., "product" matches "products")
            elif (concept.lower() in table_name.lower() or 
                  table_name.lower() in concept.lower() or
                  # Handle plural/singular variations
                  (concept.lower().rstrip('s') == table_name.lower().rstrip('s'))):
                best_col = self._find_best_column_for_table(table_schema, required_roles)
                matches.append({
                    'table': table_name,
                    'column': best_col['name'],
                    'semantic_role': best_col['role'],
                    'match_type': 'table_partial_match',
                    'confidence': 0.85
                })
        
        # Then search through columns
        for table_name, table_schema in self.schema_manager.schema.tables.items():
            for col_name, col_schema in table_schema.columns.items():
                
                # Exact column name match (highest confidence)
                if concept.lower() == col_name.lower():
                    matches.append({
                        'table': table_name,
                        'column': col_name,
                        'semantic_role': col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role),
                        'match_type': 'column_exact_match',
                        'confidence': 0.9
                    })
                
                # Partial column name match
                elif concept.lower() in col_name.lower() or col_name.lower() in concept.lower():
                    matches.append({
                        'table': table_name,
                        'column': col_name,
                        'semantic_role': col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role),
                        'match_type': 'column_partial_match',
                        'confidence': 0.7
                    })
                
                # Semantic role match for required roles
                elif col_schema.semantic_role in required_roles:
                    matches.append({
                        'table': table_name,
                        'column': col_name,
                        'semantic_role': col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role),
                        'match_type': 'role_match',
                        'confidence': 0.6
                    })
                
                # Business domain match
                elif (hasattr(col_schema, 'business_domain') and 
                      col_schema.business_domain and 
                      concept.lower() in col_schema.business_domain.lower()):
                    matches.append({
                        'table': table_name,
                        'column': col_name,
                        'semantic_role': col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role),
                        'match_type': 'domain_match',
                        'confidence': 0.5
                    })
        
        # Remove duplicates and sort by confidence
        unique_matches = []
        seen_keys = set()
        for match in matches:
            key = (match['table'], match['column'])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_matches.append(match)
        
        return sorted(unique_matches, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _find_best_column_for_table(self, table_schema, required_roles: List) -> Dict[str, str]:
        """Find the best representative column for a table"""
        # Priority order: required roles > identifiers > measures > others
        
        # First, look for columns with required roles
        for col_name, col_schema in table_schema.columns.items():
            if col_schema.semantic_role in required_roles:
                return {
                    'name': col_name,
                    'role': col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role)
                }
        
        # Then look for identifier columns (primary keys)
        for col_name, col_schema in table_schema.columns.items():
            role_str = col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role)
            if role_str == 'identifier' and col_name.endswith('_id'):
                return {'name': col_name, 'role': role_str}
        
        # Then look for measure columns
        for col_name, col_schema in table_schema.columns.items():
            role_str = col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role)
            if role_str == 'measure':
                return {'name': col_name, 'role': role_str}
        
        # Finally, just return the first column
        if table_schema.columns:
            first_col_name = list(table_schema.columns.keys())[0]
            first_col_schema = table_schema.columns[first_col_name]
            return {
                'name': first_col_name,
                'role': first_col_schema.semantic_role.value if hasattr(first_col_schema.semantic_role, 'value') else str(first_col_schema.semantic_role)
            }
        
        # Fallback
        return {'name': 'unknown', 'role': 'unknown'}
    
    def _llm_semantic_concept_matching(self, concept: str, action_type: str, state: ReActQueryState) -> List[Dict[str, Any]]:
        """Use LLM to find semantic matches for concepts that don't have exact name matches"""
        
        if not hasattr(self.schema_manager, 'schema') or not self.schema_manager.schema:
            return []
        
        # Build schema context for LLM
        schema_context = self._build_schema_context_for_llm()
        
        # Create semantic matching prompt
        prompt = f"""
Given the user query concept "{concept}" in the context of {action_type}, find the most semantically relevant database columns.

Available schema:
{schema_context}

Task: Identify which columns best represent the concept "{concept}" for {action_type} analysis.

Consider semantic relationships like:
- "revenue" or "sales" → price, amount, value columns
- "location" → city, state, geographical columns  
- "time" or "temporal" → date, timestamp columns
- "customer" → customer_id, customer_name columns

Return your analysis as a JSON list with this format:
[
  {{
    "table": "table_name",
    "column": "column_name", 
    "semantic_role": "role",
    "reasoning": "why this column matches the concept",
    "confidence": 0.8
  }}
]

Focus on the top 3 most relevant matches. Confidence should be 0.6-0.9 for semantic matches.
"""

        try:
            # Use the LLM to find semantic matches
            print(f"      🔄 Calling LLM for concept '{concept}'...")
            response = self.llm.predict(prompt)
            print(f"      📝 LLM response length: {len(response)} characters")
            
            # Parse LLM response 
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                matches_data = json.loads(json_match.group())
                
                # Convert to our format
                llm_matches = []
                for match_data in matches_data:
                    # Validate that table and column exist
                    table_name = match_data.get('table')
                    column_name = match_data.get('column')
                    
                    if (table_name in self.schema_manager.schema.tables and 
                        column_name in self.schema_manager.schema.tables[table_name].columns):
                        
                        llm_matches.append({
                            'table': table_name,
                            'column': column_name,
                            'semantic_role': match_data.get('semantic_role', 'unknown'),
                            'match_type': 'llm_semantic_match',
                            'confidence': min(match_data.get('confidence', 0.7), 0.85),  # Cap at 0.85 for LLM matches
                            'reasoning': match_data.get('reasoning', 'LLM semantic match')
                        })
                
                return llm_matches[:3]  # Return top 3
                
        except Exception as e:
            # Fallback: log error but don't fail
            print(f"LLM semantic matching failed for '{concept}': {e}")
            return []
        
        return []
    
    def _build_schema_context_for_llm(self) -> str:
        """Build a concise schema description for LLM"""
        schema_lines = []
        
        for table_name, table_schema in self.schema_manager.schema.tables.items():
            columns_info = []
            for col_name, col_schema in table_schema.columns.items():
                role = col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role)
                columns_info.append(f"{col_name}({role})")
            
            schema_lines.append(f"{table_name}: {', '.join(columns_info)}")
        
        return '\n'.join(schema_lines)
    
    def _determine_required_roles(self, action_type: str) -> List:
        """Determine required semantic roles based on action type"""
        role_mappings = {
            'aggregation': [SemanticRole.MEASURE, SemanticRole.DIMENSION],
            'geographical_analysis': [SemanticRole.GEOGRAPHICAL, SemanticRole.MEASURE],
            'trend_analysis': [SemanticRole.TEMPORAL, SemanticRole.MEASURE],
            'comparison': [SemanticRole.DIMENSION, SemanticRole.MEASURE],
            'filtering': [SemanticRole.DIMENSION, SemanticRole.IDENTIFIER]
        }
        return role_mappings.get(action_type, [SemanticRole.MEASURE, SemanticRole.DIMENSION])
    
    def _calculate_mapping_confidence(self, concept_mappings: Dict[str, List]) -> float:
        """Calculate overall confidence in schema mappings"""
        if not concept_mappings:
            return 0.0
        
        total_confidence = 0.0
        total_concepts = 0
        
        for concept, matches in concept_mappings.items():
            if matches:
                # Use highest confidence match for each concept
                max_confidence = max(match['confidence'] for match in matches)
                total_confidence += max_confidence
            total_concepts += 1
        
        return total_confidence / total_concepts if total_concepts > 0 else 0.0
    
    def _generate_fallback_mapping(self, thought: str) -> Dict[str, Any]:
        """Generate fallback mapping when primary mapping fails"""
        # Extract concepts from thought
        if "map concepts" in thought:
            concepts_start = thought.find('[') + 1
            concepts_end = thought.find(']')
            if concepts_start > 0 and concepts_end > concepts_start:
                concepts_str = thought[concepts_start:concepts_end]
                concepts = [c.strip().strip("'\"") for c in concepts_str.split(',')]
            else:
                concepts = ['data']
        else:
            concepts = ['data']
        
        # Use all available tables as fallback
        all_tables = []
        if hasattr(self.schema_manager, 'schema') and self.schema_manager.schema:
            all_tables = list(self.schema_manager.schema.tables.keys())[:3]  # Limit to 3 tables
        
        fallback_mappings = {}
        for concept in concepts:
            fallback_mappings[concept] = [{
                'table': table,
                'column': 'unknown',
                'semantic_role': 'unknown',
                'match_type': 'fallback',
                'confidence': 0.3
            } for table in all_tables[:1]]  # Map to first table only
        
        return {
            'concept_mappings': fallback_mappings,
            'relevant_tables': all_tables,
            'joins_needed': len(all_tables) > 1,
            'required_roles': ['measure', 'dimension'],
            'mapping_confidence': 0.3
        }