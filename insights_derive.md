Looking at your comprehensive knowledge graph module, you've built an excellent foundation with sophisticated table-level intelligence. To achieve your goal of building an agent that can derive analytic insights from entire databases, you need to expand from **table-centric** to **database-centric** intelligence. Here's what I suggest you need to add:
Ran tool

Based on your current excellent table intelligence foundation, here's what you need to build to achieve database-wide analytic insights:

## 🎯 **Missing Components for Database-Wide Analytics**

### 1. **Database-Wide Analysis Engine** 
You need a central orchestrator that can:

```python
# New: src/knowledge_graph/database_analytics_engine.py

class DatabaseAnalyticsEngine:
    """
    Central engine for database-wide analytics and insights
    """
    
    def __init__(self, table_intelligence: EnhancedTableIntelligenceLayer):
        self.table_intelligence = table_intelligence
        self.change_detector = DatabaseChangeDetector()
        self.causal_analyzer = CausalAnalysisEngine()
        self.anomaly_detector = DatabaseWideAnomalyDetector()
        self.trend_analyzer = CrossTableTrendAnalyzer()
        
    def analyze_database_comprehensive(self, tables: Dict[str, pd.DataFrame]) -> DatabaseInsights:
        """Perform comprehensive database-wide analysis"""
        # 1. Analyze each table
        # 2. Find cross-table patterns
        # 3. Detect system-wide anomalies
        # 4. Generate causal insights
        # 5. Predict trends and changes
```

### 2. **Change Detection & Temporal Analysis**
Currently missing - essential for "detecting changes":

```python
# New: src/knowledge_graph/change_detection_engine.py

class DatabaseChangeDetector:
    """
    Detect and analyze changes across database over time
    """
    
    def detect_schema_changes(self, current_snapshot, previous_snapshot):
        """Detect structural changes in tables"""
        
    def detect_data_drift(self, current_data, historical_data):
        """Detect statistical drift in data distributions"""
        
    def detect_relationship_changes(self, current_graph, previous_graph):
        """Detect changes in table relationships"""
        
    def generate_change_insights(self, changes):
        """Generate business insights from detected changes"""
```

### 3. **Causal Analysis Engine**
For "finding causes" - currently missing:

```python
# New: src/knowledge_graph/causal_analysis_engine.py

class CausalAnalysisEngine:
    """
    Perform causal analysis across database tables
    """
    
    def identify_root_causes(self, effect_table, metric_change):
        """Find root causes of changes in key metrics"""
        
    def analyze_impact_propagation(self, source_table, change_event):
        """Analyze how changes propagate through the database"""
        
    def discover_causal_chains(self, database_snapshot):
        """Discover causal relationships between tables"""
```

### 4. **Cross-Table Pattern Discovery**
Extend your community detection to find deeper patterns:

```python
# Enhance: src/knowledge_graph/pattern_discovery_engine.py

class CrossTablePatternEngine:
    """
    Discover complex patterns across multiple tables
    """
    
    def discover_business_processes(self, table_profiles):
        """Map business processes across tables"""
        
    def find_metric_dependencies(self, fact_tables):
        """Find how metrics in different tables depend on each other"""
        
    def detect_cyclic_relationships(self, graph):
        """Detect circular dependencies and feedback loops"""
```

### 5. **Intelligent Query Agent**
Enhance your `SmartQueryAssistant` to be more analytical:
Read file: src/knowledge_graph/smart_query_assistant.py

## 🚀 **Recommended Implementation Plan**

Based on your excellent foundation, here's what I suggest you build:

### **Phase 1: Database-Wide Intelligence (Core)**
Read file: src/knowledge_graph/table_intelligence.py

Here's my comprehensive suggestion for building a database-wide analytic insights agent:

## 🏗️ **1. Database Intelligence Engine** 
Build a central orchestrator that extends your table intelligence:

```python
# New: src/knowledge_graph/database_intelligence_engine.py

@dataclass
class DatabaseSnapshot:
    """Complete database snapshot for analysis"""
    timestamp: str
    table_profiles: Dict[str, EnhancedTableProfile]
    cross_table_relationships: Dict[str, Any]
    system_metrics: Dict[str, Any]
    business_context: Dict[str, Any]

@dataclass 
class DatabaseInsights:
    """Comprehensive database insights"""
    executive_summary: str
    data_health_score: float
    trend_analysis: Dict[str, Any]
    anomaly_alerts: List[Dict[str, Any]]
    causal_insights: List[Dict[str, Any]]
    business_opportunities: List[str]
    recommended_actions: List[str]
    cross_table_patterns: Dict[str, Any]

class DatabaseIntelligenceEngine:
    """Central engine for database-wide analytics"""
    
    def __init__(self, table_intelligence: EnhancedTableIntelligenceLayer):
        self.table_intelligence = table_intelligence
        self.relationship_detector = self._init_relationship_detector()
        self.change_detector = self._init_change_detector()
        self.anomaly_detector = self._init_anomaly_detector()
        self.causal_engine = self._init_causal_engine()
        
    def analyze_database(self, tables: Dict[str, pd.DataFrame], 
                        previous_snapshot: Optional[DatabaseSnapshot] = None) -> DatabaseInsights:
        """Perform comprehensive database analysis"""
        
        # 1. Individual table analysis
        table_profiles = self._analyze_all_tables(tables)
        
        # 2. Cross-table relationship analysis  
        relationships = self._analyze_cross_table_relationships(tables, table_profiles)
        
        # 3. Database-wide patterns
        patterns = self._discover_database_patterns(table_profiles, relationships)
        
        # 4. Change detection (if previous snapshot available)
        changes = self._detect_changes(table_profiles, previous_snapshot) if previous_snapshot else {}
        
        # 5. Anomaly detection at database level
        anomalies = self._detect_database_anomalies(table_profiles, relationships)
        
        # 6. Causal analysis
        causal_insights = self._perform_causal_analysis(table_profiles, changes, relationships)
        
        # 7. Generate insights
        return self._generate_database_insights(
            table_profiles, relationships, patterns, changes, anomalies, causal_insights
        )
```

## 🔍 **2. Change Detection & Temporal Intelligence**
This is crucial for "detecting changes":

```python
# New: src/knowledge_graph/change_detection_engine.py

class DatabaseChangeDetector:
    """Detect and analyze changes across database"""
    
    def detect_schema_evolution(self, current_profiles: Dict, historical_profiles: Dict):
        """Detect schema changes over time"""
        changes = {
            'new_tables': [],
            'removed_tables': [],
            'schema_changes': [],
            'column_additions': [],
            'column_removals': []
        }
        # Implementation for tracking schema evolution
        
    def detect_data_distribution_drift(self, current_data: Dict[str, pd.DataFrame], 
                                     historical_stats: Dict):
        """Detect statistical drift in data distributions"""
        drift_alerts = []
        for table_name, df in current_data.items():
            # Compare distributions, means, variances
            # Flag significant changes
            pass
            
    def detect_volume_changes(self, current_profiles: Dict, historical_profiles: Dict):
        """Detect significant changes in data volume"""
        volume_changes = []
        for table_name, profile in current_profiles.items():
            if table_name in historical_profiles:
                prev_count = historical_profiles[table_name].row_count
                curr_count = profile.row_count
                change_rate = (curr_count - prev_count) / prev_count if prev_count > 0 else 0
                
                if abs(change_rate) > 0.2:  # 20% change threshold
                    volume_changes.append({
                        'table': table_name,
                        'change_rate': change_rate,
                        'severity': 'high' if abs(change_rate) > 0.5 else 'medium'
                    })
        return volume_changes
        
    def detect_quality_degradation(self, current_profiles: Dict, historical_profiles: Dict):
        """Detect data quality degradation over time"""
        # Compare quality scores, alert on degradation
        pass
```

## 🎯 **3. Causal Analysis Engine**
For "finding causes" - the heart of analytic insights:

```python
# New: src/knowledge_graph/causal_analysis_engine.py

class CausalAnalysisEngine:
    """Advanced causal analysis for database insights"""
    
    def analyze_metric_causality(self, effect_table: str, effect_metric: str,
                               candidate_tables: List[str], 
                               time_window: int = 30) -> List[Dict]:
        """Find what causes changes in key business metrics"""
        
        causal_relationships = []
        
        for candidate_table in candidate_tables:
            # Analyze temporal precedence
            # Check correlation strength
            # Test causal hypotheses using statistical methods
            
            causal_score = self._calculate_causal_strength(
                effect_table, effect_metric, candidate_table
            )
            
            if causal_score > 0.7:  # Strong causal indication
                causal_relationships.append({
                    'cause_table': candidate_table,
                    'effect_table': effect_table,
                    'effect_metric': effect_metric,
                    'causal_strength': causal_score,
                    'mechanism': self._infer_causal_mechanism(candidate_table, effect_table),
                    'time_lag': self._estimate_time_lag(candidate_table, effect_table)
                })
                
        return causal_relationships
    
    def trace_impact_propagation(self, source_change: Dict, database_graph: nx.MultiDiGraph):
        """Trace how changes propagate through the database"""
        impact_chain = []
        # Use graph traversal to find impact paths
        # Calculate impact strength at each step
        return impact_chain
    
    def identify_root_causes(self, symptom: Dict, database_snapshot: DatabaseSnapshot):
        """Find root causes of observed symptoms"""
        # Symptom: {'table': 'orders', 'metric': 'total_value', 'change': 'decrease_20%'}
        
        potential_causes = []
        
        # 1. Check upstream tables (suppliers, products, customers)
        # 2. Check related metrics in same table
        # 3. Check temporal patterns
        # 4. Check external factors (seasonality, etc.)
        
        return self._rank_causes_by_likelihood(potential_causes)
```

## 📊 **4. Enhanced Analytics Agent**
Transform your `SmartQueryAssistant` into a proactive analytics agent:

```python
# Enhanced: src/knowledge_graph/analytics_agent.py

class DatabaseAnalyticsAgent:
    """Intelligent agent for database-wide analytics"""
    
    def __init__(self, database_engine: DatabaseIntelligenceEngine):
        self.db_engine = database_engine
        self.llm_summarizer = database_engine.llm_summarizer
        self.knowledge_base = {}
        
    def analyze_business_question(self, question: str, 
                                context: Optional[Dict] = None) -> AnalyticsResponse:
        """Answer complex business questions across the database"""
        
        # Parse question intent
        intent = self._parse_question_intent(question)
        
        # Route to appropriate analysis method
        if intent['type'] == 'causal_analysis':
            return self._perform_causal_investigation(question, intent)
        elif intent['type'] == 'trend_analysis':
            return self._perform_trend_analysis(question, intent)
        elif intent['type'] == 'anomaly_investigation':
            return self._investigate_anomaly(question, intent)
        elif intent['type'] == 'impact_analysis':
            return self._perform_impact_analysis(question, intent)
        else:
            return self._perform_general_analysis(question, intent)
    
    def proactive_insights_discovery(self) -> List[ProactiveInsight]:
        """Automatically discover insights without specific questions"""
        
        insights = []
        
        # 1. Detect unusual patterns
        unusual_patterns = self._detect_unusual_patterns()
        
        # 2. Find hidden correlations
        hidden_correlations = self._find_hidden_correlations()
        
        # 3. Identify optimization opportunities
        optimizations = self._identify_optimization_opportunities()
        
        # 4. Predict future issues
        predictions = self._predict_future_issues()
        
        return self._prioritize_insights(insights)
    
    def continuous_monitoring(self) -> List[Alert]:
        """Continuously monitor database for important changes"""
        
        alerts = []
        
        # Monitor key metrics
        metric_alerts = self._monitor_key_metrics()
        
        # Monitor data quality
        quality_alerts = self._monitor_data_quality()
        
        # Monitor relationship changes
        relationship_alerts = self._monitor_relationships()
        
        return self._prioritize_alerts(alerts)
```

## 🎯 **5. Specific Implementation Steps**

### **Step 1: Start with Cross-Table Pattern Discovery**
Enhance your existing `community_detection.py`:

```python
# Enhance existing community_detection.py with:

class BusinessProcessDetector:
    """Detect business processes spanning multiple tables"""
    
    def map_customer_journey(self, tables: Dict[str, pd.DataFrame]):
        """Map customer journey across tables"""
        # customers -> orders -> payments -> reviews
        
    def identify_supply_chain_flow(self, tables: Dict[str, pd.DataFrame]):
        """Identify supply chain processes"""
        # suppliers -> products -> inventory -> orders -> shipping
        
    def detect_revenue_cycle(self, tables: Dict[str, pd.DataFrame]):
        """Detect revenue generation processes"""
        # leads -> customers -> orders -> payments -> renewals
```

### **Step 2: Build Change Detection**
Start simple with volume and quality monitoring:

```python
# New: src/knowledge_graph/simple_change_detector.py

class SimpleChangeDetector:
    """Simple change detection to get started"""
    
    def __init__(self):
        self.baseline_stats = {}
    
    def establish_baseline(self, tables: Dict[str, pd.DataFrame]):
        """Establish baseline statistics"""
        for name, df in tables.items():
            self.baseline_stats[name] = {
                'row_count': len(df),
                'null_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)),
                'numeric_means': df.select_dtypes(include=[np.number]).mean().to_dict()
            }
    
    def detect_changes(self, tables: Dict[str, pd.DataFrame]):
        """Detect simple changes from baseline"""
        changes = []
        for name, df in tables.items():
            if name in self.baseline_stats:
                # Check volume change
                baseline_count = self.baseline_stats[name]['row_count']
                current_count = len(df)
                volume_change = (current_count - baseline_count) / baseline_count
                
                if abs(volume_change) > 0.1:  # 10% change
                    changes.append({
                        'table': name,
                        'type': 'volume_change',
                        'change_rate': volume_change,
                        'description': f"Row count changed by {volume_change:.1%}"
                    })
        
        return changes
```

### **Step 3: Enhanced Query Intelligence**
Upgrade your `SmartQueryAssistant` for complex analysis:

```python
# Enhanced query types to handle:

ANALYTICAL_QUERY_TYPES = {
    'causal': [
        "What caused the decrease in sales?",
        "Why did customer satisfaction drop?",
        "What's driving the increase in costs?"
    ],
    'predictive': [
        "What will happen to revenue next quarter?",
        "Which customers are likely to churn?",
        "When will inventory run out?"
    ],
    'diagnostic': [
        "Why is data quality declining?",
        "What's wrong with our order processing?",
        "Where are we losing customers?"
    ],
    'optimization': [
        "How can we improve efficiency?",
        "What's the best pricing strategy?",
        "Which products should we promote?"
    ]
}
```

## 📈 **Implementation Priority**

**Phase 1 (Start Here):**
1. Build `DatabaseIntelligenceEngine` with basic cross-table analysis
2. Enhance `community_detection.py` to find business processes
3. Create simple change detection for volume/quality monitoring
4. Upgrade `SmartQueryAssistant` to handle analytical questions

**Phase 2:**
1. Add causal analysis capabilities
2. Build proactive insight discovery
3. Implement continuous monitoring and alerting
4. Add predictive analytics

**Phase 3:**
1. Advanced ML-based causality detection
2. Natural language interface for complex queries
3. Automated report generation
4. Integration with business intelligence tools

Would you like me to help you implement any specific component first? I'd recommend starting with the `DatabaseIntelligenceEngine` as it will orchestrate everything else.