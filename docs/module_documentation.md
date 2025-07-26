# Data Graph System - Module Documentation

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
3. [Module Relationships](#module-relationships)
4. [Integration with ReAct Guide](#integration-with-react-guide)

---

## 🏗️ Architecture Overview

The Data Graph system implements an ML-enhanced knowledge graph for intelligent query planning. The architecture consists of three main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Agents    │  │     API      │  │    Frontend      │  │
│  │  (Planners) │  │  (Future)    │  │    (Future)      │  │
│  └──────┬──────┘  └──────────────┘  └──────────────────┘  │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│         │          Knowledge Layer                           │
│  ┌──────▼──────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Schema    │  │  Knowledge   │  │  Relationship    │  │
│  │  Manager    │  │    Graph     │  │   Detector       │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼────────────────┼───────────────────┼─────────────┘
          │                │                   │
┌─────────▼────────────────▼───────────────────▼─────────────┐
│                     Data Layer                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Data     │  │    Data      │  │   CSV/Future     │  │
│  │   Models    │  │  Connectors  │  │   Connectors     │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Core Modules

### 1. Schema Manager (`src/schema/schema_manager.py`)

**Purpose**: Flexible schema discovery and management system that supports both auto-discovery and manual configuration.

**Key Features**:
- **Auto-discovery** with ydata-profiling integration
- **Semantic role detection** (IDENTIFIER, MEASURE, DIMENSION, etc.)
- **Business domain inference**
- **Statistical property extraction**
- **Relationship hints**

**Core Classes**:
```python
- DataType(Enum)         # Standardized data types
- SemanticRole(Enum)     # Column semantic roles
- ColumnSchema           # Single column metadata
- TableSchema            # Table-level schema
- DatasetSchema          # Complete dataset schema
- SchemaAutoDiscovery    # Auto-discovery engine
- SchemaManager          # Main interface
```

**Integration Points**:
- Provides schema context to agents
- Feeds semantic roles to knowledge graph
- Enables ML relationship detection

---

### 2. Knowledge Graph Builder (`src/knowledge_graph/`)

#### 2.1 Base Graph Builder (`graph_builder.py`)
**Purpose**: Constructs the foundational knowledge graph structure.

**Key Features**:
- Creates hierarchical node structure (dataset → table → column)
- Basic relationship detection
- Graph visualization
- Relationship querying

#### 2.2 Enhanced Graph Builder (`enhanced_graph_builder.py`)
**Purpose**: Extends base builder with ML-powered relationship detection.

**Key Features**:
- **ML relationship discovery** using enhanced detector
- **Confidence-weighted edges**
- **Multi-type relationships** (FOREIGN_KEY, CORRELATED, SAME_DOMAIN, etc.)
- **Rich visualization** with relationship types

**Core Methods**:
```python
- _discover_inter_table_relationships()  # Between tables
- _discover_intra_table_relationships()  # Within tables
- get_relationships_by_type()            # Query by type
- get_relationship_summary()             # Statistics
- visualize_enhanced_graph()             # Rich visualization
```

---

### 3. Relationship Detectors (`src/knowledge_graph/`)

#### 3.1 ML Relationship Detector (`relationship_detector.py`)
**Purpose**: Uses machine learning to detect relationships between columns.

**Detection Methods**:
- **Foreign Key Detection**: Pattern matching + cardinality analysis
- **Correlation Analysis**: Statistical correlation for numeric columns
- **Categorical Similarity**: Jaccard similarity for categorical data
- **Value Distribution**: KL divergence for distribution comparison
- **Mutual Information**: Information theory metrics

**Confidence Scoring**: Weighted ensemble of multiple signals

#### 3.2 Enhanced Relationship Detector (`enhanced_relationship_detector.py`)
**Purpose**: Production-ready detector with better handling of edge cases.

**Improvements**:
- Robust handling of missing data
- Datetime column support
- Better memory management
- More nuanced relationship types

---

### 4. Query Planning Agents (`src/agents/`)

#### 4.1 Schema-Driven Query Planner (`schema_driven_query_planner.py`)
**Purpose**: Traditional approach sending full schema to LLM.

**Components**:
- `KnowledgeGraphTraverser`: Finds paths between concepts
- `LLMQueryAnalyzer`: Analyzes query intent with full schema context
- `SchemaDrivenQueryPlanner`: Main orchestrator

**Token Usage**: ~800+ tokens (full schema dump)

#### 4.2 KG-Enriched Query Planner (`kg_enriched_query_planner.py`)
**Purpose**: Optimized approach using enriched context from knowledge graph.

**Components**:
- `KnowledgeGraphContextExtractor`: Extracts intelligent context
- `EnrichedContextLLMAnalyzer`: Analyzes with minimal context
- `KGEnrichedQueryPlanner`: Main orchestrator

**Key Innovation**:
```python
RelationshipContext:
    - strong_relationships     # High-confidence relationships only
    - concept_clusters        # Business concept groupings
    - join_paths             # Pre-computed optimal paths
    - measure_dimension_pairs # Analysis-ready combinations
    - temporal_relationships  # Time-based opportunities
```

**Token Usage**: ~250-300 tokens (70% reduction)

---

### 5. Data Layer (`src/data/`)

#### 5.1 Data Models (`models/data_models.py`)
**Purpose**: Core data structures for the system.

**Key Classes**:
- `Relationship`: Represents discovered relationships
- Future: Additional data models

#### 5.2 Connectors (`connectors/`)
**Purpose**: Abstract data source connections.

**Current**:
- `BaseConnector`: Abstract interface
- `CSVConnector`: CSV file support

**Future**: Database connectors, API connectors

---

## 🔗 Module Relationships

### Data Flow
```
CSV Files → CSVConnector → SchemaManager → KnowledgeGraphBuilder
                                ↓                    ↓
                         SchemaAutoDiscovery    MLRelationshipDetector
                                ↓                    ↓
                            DatasetSchema      Enhanced Graph with ML edges
                                ↓                    ↓
                        Query Planner Agents ← Enriched Context
```

### Key Integration Points

1. **Schema → Knowledge Graph**
   - Semantic roles guide relationship detection
   - Business domains inform clustering
   - Statistical properties feed ML features

2. **Knowledge Graph → Query Planners**
   - Graph provides relationship intelligence
   - ML weights guide join path selection
   - Concept clusters inform query understanding

3. **Schema + KG → LLM Context**
   - Original: Raw schema dump (800+ tokens)
   - Optimized: Enriched relationships (250 tokens)

---

## 🚀 Integration with ReAct Guide

### Current Implementation vs Guide Stages

#### Stage 1: Intent Recognition ✅ Partially Implemented
**Guide**: Extract structured intent from query
**Current**: 
- `LLMQueryAnalyzer.analyze_query()` in schema_driven_query_planner.py
- Extracts concepts, analysis type, relationships needed
- **Gap**: Not yet a separate ReAct agent with reasoning traces

#### Stage 2: Schema Validation ✅ Partially Implemented  
**Guide**: Map intent concepts to schema entities
**Current**:
- `_find_relevant_columns()` in schema_driven_query_planner.py
- Maps concepts to actual columns
- **Gap**: No compressed schema representation yet

#### Stage 3: Relationship Discovery ✅ Well Implemented
**Guide**: Find optimal join paths using KG
**Current**:
- `KnowledgeGraphTraverser` finds paths
- `_find_concept_connections()` discovers relationships
- ML-weighted path selection
- **Enhancement**: Already uses ML confidence scores

#### Stage 4: Query Generation ❌ Not Implemented
**Guide**: Generate SQL/Pandas/DAX queries
**Current**:
- Query planning complete, but no query generation
- **Gap**: Need query builder component

### Token Optimization Implementation

#### Current Optimizations
1. **Enriched Context** (kg_enriched_query_planner.py):
   - Extracts only relevant relationships
   - Pre-computes join paths
   - Groups by business concepts

2. **Context Compression**:
   - Relationship notation: `C.id↔O.customer_id(0.98,FK)`
   - Concept clusters: Grouped columns by domain
   - Analysis pairs: Pre-matched measure-dimension combinations

#### Missing Optimizations from Guide
1. **Progressive Context Loading**: Currently loads all context at once
2. **Stage Skipping**: No conditional skipping based on confidence
3. **Compressed Schema Format**: Not using abbreviated notation
4. **Multi-stage Workflow**: Not using LangGraph state management

### Recommended Next Steps

1. **Implement ReAct Agents**:
   - Separate agents for each stage
   - Add reasoning traces
   - Implement state management

2. **Add Query Generation**:
   - SQL builder
   - Pandas query builder
   - DAX support (optional)

3. **Implement Progressive Loading**:
   - Stage-specific context
   - Conditional skipping
   - Token tracking per stage

4. **Add LangGraph Integration**:
   - State management
   - Conditional routing
   - Error recovery

5. **Optimize Context Compression**:
   - Implement abbreviated notation
   - Better compression algorithms
   - Context caching