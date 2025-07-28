# 🚀 ReAct Query Planning: Implementation Guide with Current Codebase

## 📋 Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Implementation Roadmap](#implementation-roadmap)
3. [Stage-by-Stage Implementation](#stage-by-stage-implementation)
4. [Code Examples](#code-examples)
5. [Token Optimization Strategy](#token-optimization-strategy)
6. [Testing & Validation](#testing--validation)

---

## 📊 Current State Analysis

### What We Have
1. **✅ Schema Management** (`src/schema/schema_manager.py`)
   - Auto-discovery with semantic roles
   - Statistical analysis
   - Business domain detection

2. **✅ Knowledge Graph** (`src/knowledge_graph/`)
   - ML-powered relationship detection
   - Confidence-weighted edges
   - Multiple relationship types

3. **✅ Basic Query Planning** (`src/agents/`)
   - Intent analysis
   - Column selection
   - Path finding

### What's Missing for Full ReAct
1. **❌ Multi-stage ReAct workflow**
2. **❌ Progressive context loading**
3. **❌ Query generation (SQL/Pandas/DAX)**
4. **❌ LangGraph state management**
5. **❌ Token tracking per stage**

---

## 🗺️ Implementation Roadmap

### Phase 1: Foundation (Current Focus)
1. Create ReAct agent structure
2. Implement stage-specific agents
3. Add state management

### Phase 2: Optimization
1. Progressive context loading
2. Context compression
3. Stage skipping logic

### Phase 3: Query Generation
1. SQL builder
2. Pandas generator
3. DAX support

---

## 🔧 Stage-by-Stage Implementation

### Stage 1: Intent Recognition Agent

**Current Code Base**: Partially in `LLMQueryAnalyzer.analyze_query()`

**New Implementation**:
```python
# src/agents/react_agents/intent_recognizer.py

from typing import Dict, Any, List
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json

@dataclass
class IntentProfile:
    action_type: str  # geographical_aggregation, trend_analysis, etc.
    target_concepts: List[str]
    analysis_scope: str  # single_table, multi_table
    complexity: str  # simple, moderate, complex
    confidence: float
    reasoning_trace: List[str]  # ReAct reasoning steps

class IntentRecognitionAgent:
    """ReAct agent for intent recognition - Stage 1"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
    def recognize_intent(self, query: str, business_context: str = "") -> IntentProfile:
        """Extract structured intent using ReAct reasoning"""
        
        prompt = f"""Analyze this query to extract structured intent.

Query: "{query}"
Business Context: {business_context or "E-commerce analysis"}

Use ReAct reasoning:
1. Thought: What is the user trying to achieve?
2. Action: Identify key concepts and analysis type
3. Observation: Note any specific requirements

Provide JSON response:
{{
    "action_type": "type of analysis",
    "target_concepts": ["concept1", "concept2"],
    "analysis_scope": "single_table or multi_table",
    "complexity": "simple/moderate/complex",
    "confidence": 0.0-1.0,
    "reasoning_trace": ["thought1", "action1", "observation1"]
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            result = json.loads(response.content)
            return IntentProfile(**result)
        except:
            # Fallback
            return IntentProfile(
                action_type="general_analysis",
                target_concepts=self._extract_concepts(query),
                analysis_scope="unknown",
                complexity="moderate",
                confidence=0.5,
                reasoning_trace=["Failed to parse LLM response"]
            )
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Simple concept extraction fallback"""
        # Use existing concept extraction logic
        keywords = ['customer', 'order', 'product', 'price', 'location', 'date']
        return [k for k in keywords if k in query.lower()]
```

### Stage 2: Schema Validation Agent

**Current Code Base**: Logic in `_find_relevant_columns()`

**New Implementation**:
```python
# src/agents/react_agents/schema_validator.py

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ValidatedMapping:
    target_tables: List[str]
    concept_mappings: Dict[str, str]  # concept -> table.column
    required_joins: bool
    confidence: float
    validation_trace: List[str]

class SchemaValidationAgent:
    """ReAct agent for schema validation - Stage 2"""
    
    def __init__(self, schema_manager):
        self.schema_manager = schema_manager
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
    def validate_and_map(self, 
                        intent: IntentProfile, 
                        compressed_schema: str) -> ValidatedMapping:
        """Map intent concepts to actual schema entities"""
        
        # Build compressed schema using existing schema manager
        schema_context = self._build_compressed_schema(intent.target_concepts)
        
        prompt = f"""Map these concepts to actual schema entities.

Intent: {intent.action_type} of {intent.target_concepts}
Compressed Schema: {schema_context}

ReAct Process:
Thought: Which tables/columns match each concept?
Action: Map each concept to specific columns
Observation: Note if joins are needed

Return JSON:
{{
    "target_tables": ["table1", "table2"],
    "concept_mappings": {{"concept": "table.column"}},
    "required_joins": true/false,
    "confidence": 0.0-1.0,
    "validation_trace": ["reasoning steps"]
}}"""

        # Use LLM for mapping
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse and return
        result = json.loads(response.content)
        return ValidatedMapping(**result)
    
    def _build_compressed_schema(self, concepts: List[str]) -> str:
        """Build compressed schema focused on concepts"""
        # Leverage existing schema manager
        compressed = []
        
        for table_name, table_schema in self.schema_manager.schema.tables.items():
            relevant_cols = []
            
            for col_name, col_schema in table_schema.columns.items():
                # Check relevance to concepts
                if any(concept in col_name.lower() or 
                      concept in table_name.lower() 
                      for concept in concepts):
                    
                    # Compressed notation
                    role_abbrev = col_schema.semantic_role.value[:3].upper()
                    relevant_cols.append(f"{col_name}:{role_abbrev}")
            
            if relevant_cols:
                # Table notation: T(name):cols
                compressed.append(f"{table_name[:1].upper()}({table_name}):{'+'.join(relevant_cols[:5])}")
        
        return ", ".join(compressed)
```

### Stage 3: Relationship Discovery Agent

**Current Code Base**: Well implemented in `KnowledgeGraphTraverser`

**New Implementation** (enhancing existing):
```python
# src/agents/react_agents/relationship_explorer.py

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass 
class JoinStrategy:
    join_path: List[str]  # [table1, table2, table3]
    join_conditions: List[Dict[str, str]]  # [{from: col1, to: col2}]
    path_confidence: float
    estimated_performance: str
    discovery_trace: List[str]

class RelationshipDiscoveryAgent:
    """ReAct agent for relationship discovery - Stage 3"""
    
    def __init__(self, kg_traverser, kg_context_extractor):
        self.traverser = kg_traverser  # Reuse existing
        self.extractor = kg_context_extractor  # Reuse existing
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
    def discover_relationships(self,
                             validated_mapping: ValidatedMapping,
                             kg_context: str) -> JoinStrategy:
        """Find optimal join paths using KG intelligence"""
        
        # Use existing traverser to find paths
        tables = validated_mapping.target_tables
        
        if len(tables) < 2:
            return JoinStrategy(
                join_path=[tables[0]] if tables else [],
                join_conditions=[],
                path_confidence=1.0,
                estimated_performance="fast",
                discovery_trace=["Single table - no joins needed"]
            )
        
        # Find paths using existing logic
        paths = []
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                table_paths = self._find_table_paths(table1, table2)
                paths.extend(table_paths)
        
        # Select optimal path
        if paths:
            best_path = max(paths, key=lambda p: p['weight'])
            
            return JoinStrategy(
                join_path=best_path['tables'],
                join_conditions=best_path['conditions'],
                path_confidence=best_path['weight'],
                estimated_performance=self._estimate_performance(best_path),
                discovery_trace=[
                    f"Found {len(paths)} possible paths",
                    f"Selected path with confidence {best_path['weight']:.2f}",
                    f"Path: {' → '.join(best_path['tables'])}"
                ]
            )
        
        return self._fallback_strategy(tables)
    
    def _find_table_paths(self, table1: str, table2: str) -> List[Dict]:
        """Wrapper around existing path finding"""
        # This would use the existing KnowledgeGraphTraverser
        # Returns paths with weights and conditions
        pass
```

### Stage 4: Query Builder Agent

**Currently Missing** - New Implementation:
```python
# src/agents/react_agents/query_builder.py

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class GeneratedQueries:
    sql: str
    pandas: str
    dax: Optional[str] = None
    generation_trace: List[str] = None

class MultiLanguageQueryBuilder:
    """ReAct agent for query generation - Stage 4"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
    def build_queries(self,
                     intent: IntentProfile,
                     mapping: ValidatedMapping,
                     join_strategy: JoinStrategy) -> GeneratedQueries:
        """Generate executable queries in multiple languages"""
        
        # Consolidate context
        context = self._build_generation_context(intent, mapping, join_strategy)
        
        prompt = f"""Generate queries based on this plan:

Intent: {intent.action_type}
Tables: {mapping.target_tables}
Joins: {self._format_joins(join_strategy)}
Measures: {self._extract_measures(mapping)}
Dimensions: {self._extract_dimensions(mapping)}

Generate:
1. SQL query with proper joins and aggregation
2. Pandas code using merge() and groupby()
3. DAX formula (if applicable)

Return JSON:
{{
    "sql": "SELECT ...",
    "pandas": "df = ...",
    "dax": "SUMMARIZE(...)",
    "generation_trace": ["step1", "step2"]
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        
        return GeneratedQueries(**result)
    
    def _build_generation_context(self, intent, mapping, join_strategy):
        """Build consolidated context for generation"""
        return {
            "intent": intent.action_type,
            "concepts": intent.target_concepts,
            "tables": mapping.target_tables,
            "columns": mapping.concept_mappings,
            "joins": join_strategy.join_conditions,
            "confidence": min(intent.confidence, 
                            mapping.confidence, 
                            join_strategy.path_confidence)
        }
```

---

## 🔄 Orchestrator with State Management

```python
# src/agents/react_orchestrator.py

from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
import time

class ReActQueryState(TypedDict):
    # User input
    user_query: str
    business_context: str
    
    # Progressive intelligence building
    intent_profile: Dict[str, Any]      # Stage 1 output
    validated_mapping: Dict[str, Any]   # Stage 2 output  
    join_strategy: Dict[str, Any]       # Stage 3 output
    generated_queries: Dict[str, Any]   # Stage 4 output
    
    # Token tracking
    tokens_per_stage: List[int]
    total_tokens: int
    
    # Workflow control
    current_stage: Literal["intent", "schema", "relationship", "query", "complete"]
    stage_confidence: float
    should_skip_next: bool

class ReActQueryPlanner:
    """Main orchestrator using LangGraph state management"""
    
    def __init__(self, schema_manager, knowledge_graph):
        self.schema_manager = schema_manager
        self.knowledge_graph = knowledge_graph
        
        # Initialize agents
        self.intent_agent = IntentRecognitionAgent()
        self.schema_agent = SchemaValidationAgent(schema_manager)
        self.relationship_agent = RelationshipDiscoveryAgent(
            KnowledgeGraphTraverser(knowledge_graph),
            KnowledgeGraphContextExtractor(knowledge_graph, schema_manager.schema)
        )
        self.query_agent = MultiLanguageQueryBuilder()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with conditional routing"""
        
        workflow = StateGraph(ReActQueryState)
        
        # Add nodes
        workflow.add_node("intent_recognizer", self._intent_node)
        workflow.add_node("schema_validator", self._schema_node)
        workflow.add_node("relationship_explorer", self._relationship_node)
        workflow.add_node("query_generator", self._query_node)
        
        # Conditional routing
        def route_after_schema(state: ReActQueryState) -> str:
            if state.get("should_skip_next", False):
                return "query_generator"  # Skip relationship discovery
            return "relationship_explorer"
        
        # Add edges
        workflow.add_edge("START", "intent_recognizer")
        workflow.add_edge("intent_recognizer", "schema_validator")
        workflow.add_conditional_edges("schema_validator", route_after_schema)
        workflow.add_edge("relationship_explorer", "query_generator")
        workflow.add_edge("query_generator", "END")
        
        return workflow.compile()
    
    def _intent_node(self, state: ReActQueryState) -> Dict:
        """Stage 1: Intent Recognition"""
        start_tokens = self._count_tokens(state["user_query"])
        
        intent = self.intent_agent.recognize_intent(
            state["user_query"], 
            state.get("business_context", "")
        )
        
        end_tokens = self._count_tokens(str(intent))
        
        return {
            "intent_profile": intent.__dict__,
            "current_stage": "schema",
            "stage_confidence": intent.confidence,
            "tokens_per_stage": [end_tokens - start_tokens]
        }
    
    def _schema_node(self, state: ReActQueryState) -> Dict:
        """Stage 2: Schema Validation"""
        # Build compressed schema based on intent
        compressed_schema = self._build_compressed_schema(
            state["intent_profile"]["target_concepts"]
        )
        
        start_tokens = self._count_tokens(compressed_schema)
        
        mapping = self.schema_agent.validate_and_map(
            IntentProfile(**state["intent_profile"]),
            compressed_schema
        )
        
        # Check if we can skip relationship discovery
        should_skip = (
            not mapping.required_joins or 
            mapping.confidence > 0.95
        )
        
        tokens_used = self._count_tokens(str(mapping)) - start_tokens
        
        return {
            "validated_mapping": mapping.__dict__,
            "current_stage": "relationship",
            "stage_confidence": mapping.confidence,
            "should_skip_next": should_skip,
            "tokens_per_stage": state["tokens_per_stage"] + [tokens_used]
        }
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """Execute the ReAct planning workflow"""
        
        initial_state = {
            "user_query": query,
            "business_context": "E-commerce analysis",
            "tokens_per_stage": [],
            "current_stage": "intent"
        }
        
        # Run workflow
        start_time = time.time()
        result = self.workflow.invoke(initial_state)
        end_time = time.time()
        
        # Calculate totals
        result["total_tokens"] = sum(result.get("tokens_per_stage", []))
        result["execution_time"] = end_time - start_time
        result["baseline_comparison"] = self._calculate_baseline_tokens()
        result["efficiency_ratio"] = (
            result["total_tokens"] / result["baseline_comparison"]
        )
        
        return result
```

---

## 💡 Token Optimization Strategy

### Current Implementation in `kg_enriched_query_planner.py`

```python
# Context compression example
def _build_enriched_context(self, context: RelationshipContext) -> str:
    """Current implementation builds enriched context"""
    sections = []
    
    # Only strong relationships
    if context.strong_relationships:
        sections.append("STRONG RELATIONSHIPS (ML confidence > 0.7):")
        for rel in context.strong_relationships[:5]:  # Limit to top 5
            sections.append(f"• {rel['business_meaning']} (weight: {rel['weight']:.2f})")
```

### Enhanced Compression Techniques

```python
# src/utils/context_compression.py

class ContextCompressor:
    """Advanced context compression utilities"""
    
    @staticmethod
    def compress_schema(table_schema: TableSchema) -> str:
        """Ultra-compressed schema representation"""
        # Table: C(customers):geo+id+desc
        roles = set()
        for col in table_schema.columns.values():
            role_abbrev = col.semantic_role.value[:3].upper()
            roles.add(role_abbrev)
        
        return f"{table_schema.name[0].upper()}({table_schema.name}):{'+'.join(roles)}"
    
    @staticmethod
    def compress_relationship(rel: Dict[str, Any]) -> str:
        """Compressed relationship notation"""
        # C.id↔O.customer_id(0.98,FK)
        from_parts = rel['from'].split('.')
        to_parts = rel['to'].split('.')
        
        from_short = f"{from_parts[-2][0].upper()}.{from_parts[-1]}"
        to_short = f"{to_parts[-2][0].upper()}.{to_parts[-1]}"
        
        return f"{from_short}↔{to_short}({rel['weight']:.2f},{rel['type'][:2]})"
    
    @staticmethod
    def compress_join_path(path: List[str]) -> str:
        """Compressed join path"""
        # C→O→I
        return "→".join([table[0].upper() for table in path])
```

---

## 🧪 Testing & Validation

### Test the Complete Pipeline

```python
# test_react_pipeline.py

def test_react_query_planning():
    """Test the complete ReAct pipeline"""
    
    # Setup
    from src.schema.schema_manager import SchemaManager
    from src.knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
    from src.agents.react_orchestrator import ReActQueryPlanner
    
    # Load data
    data = load_test_data()
    
    # Build schema and KG
    schema_manager = SchemaManager()
    schema = schema_manager.discover_schema_from_data(data, "ecommerce")
    
    kg_builder = EnhancedKnowledgeGraphBuilder()
    kg_builder.add_dataset(data, "ecommerce")
    
    # Initialize planner
    planner = ReActQueryPlanner(schema_manager, kg_builder.graph)
    
    # Test queries
    test_queries = [
        "Show customer orders with total prices by location",
        "What are the top selling products by revenue?",
        "Analyze order trends over time by category"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        result = planner.plan_query(query)
        
        print(f"\nStages:")
        print(f"1. Intent: {result['intent_profile']['action_type']}")
        print(f"2. Schema: {len(result['validated_mapping']['target_tables'])} tables")
        print(f"3. Relationships: confidence {result['join_strategy']['path_confidence']:.2f}")
        print(f"4. Queries: SQL, Pandas, DAX generated")
        
        print(f"\nToken Usage:")
        print(f"  Per stage: {result['tokens_per_stage']}")
        print(f"  Total: {result['total_tokens']}")
        print(f"  Baseline: {result['baseline_comparison']}")
        print(f"  Efficiency: {result['efficiency_ratio']:.1%} of baseline")
        
        print(f"\nGenerated SQL:")
        print(result['generated_queries']['sql'])
```

---

## 🎯 Next Steps for Implementation

1. **Create the ReAct agent files**:
   ```
   src/agents/react_agents/
   ├── __init__.py
   ├── intent_recognizer.py
   ├── schema_validator.py
   ├── relationship_explorer.py
   └── query_builder.py
   ```

2. **Implement the orchestrator**:
   ```
   src/agents/react_orchestrator.py
   ```

3. **Add context compression utilities**:
   ```
   src/utils/context_compression.py
   ```

4. **Create comprehensive tests**:
   ```
   tests/test_react_pipeline.py
   tests/test_token_efficiency.py
   ```

5. **Integrate with existing codebase**:
   - Reuse `SchemaManager` for schema operations
   - Leverage `KnowledgeGraphTraverser` for path finding
   - Utilize `MLRelationshipDetector` results
   - Build on `KnowledgeGraphContextExtractor`

This implementation plan builds on your existing strong foundation while adding the missing ReAct components for a complete, token-efficient query planning system.