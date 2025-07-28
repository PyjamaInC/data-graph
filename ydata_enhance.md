I'll provide a comprehensive analysis of how to incorporate ydata-profiling's ML abilities to advance your `table_intelligence.py` capabilities. Let me first check the latest ydata-profiling documentation to ensure I'm using the most current features.
Ran tool
Now let me get more specific information about the latest ML features in ydata-profiling:
Ran tool
Based on my comprehensive research of ydata-profiling's latest capabilities and your current `table_intelligence.py` implementation, I'll provide a detailed enhancement strategy incorporating ydata-profiling's advanced ML features.

# **Comprehensive Enhancement Strategy: Incorporating ydata-profiling ML Capabilities into table_intelligence.py**

## **1. Advanced Data Quality Intelligence System**

### **1.1 Enhanced Alert-Based Quality Assessment**

**Current Implementation:** Basic completeness ratio for data quality scoring.

**Enhancement:** Leverage ydata-profiling's 20+ automatic alert system for comprehensive quality assessment:

```python
@dataclass
class EnhancedDataQualityProfile:
    """Comprehensive data quality profile using ydata-profiling alerts"""
    table_name: str
    overall_quality_score: float
    alert_summary: Dict[str, int]
    critical_alerts: List[Dict[str, Any]]
    warning_alerts: List[Dict[str, Any]]
    info_alerts: List[Dict[str, Any]]
    quality_recommendations: List[str]
    trend_alerts: List[Dict[str, Any]]  # NEW: Trend-based alerts
    distribution_alerts: List[Dict[str, Any]]  # NEW: Distribution-based alerts
    correlation_alerts: List[Dict[str, Any]]  # NEW: Correlation-based alerts

class AdvancedDataQualityAnalyzer:
    """
    Advanced data quality analyzer using ydata-profiling's ML-powered alerts
    """
    
    # Alert categorization mapping
    ALERT_CATEGORIES = {
        'critical': ['constant', 'zeros', 'infinite', 'rejected', 'unsupported', 'empty'],
        'warning': ['high_correlation', 'high_cardinality', 'imbalance', 'skewness', 'missing', 'duplicates'],
        'info': ['unique', 'uniform', 'constant_length', 'date', 'seasonal', 'non_stationary']
    }
    
    # Quality scoring weights
    ALERT_WEIGHTS = {
        'critical': -10,
        'warning': -3,
        'info': -1
    }

    def analyze_data_quality(self, df: pd.DataFrame, table_name: str) -> EnhancedDataQualityProfile:
        """Generate comprehensive data quality analysis using ydata-profiling alerts"""
        
        # Generate enhanced profile with comprehensive alerts
        profile = ProfileReport(
            df,
            title=f"Quality Analysis: {table_name}",
            minimal=False,
            explorative=True,
            # Enable all alert types
            vars={
                'num': {'low_categorical_threshold': 0},  # Detect categorical in numeric
                'cat': {'cardinality_threshold': 50}      # Adjust cardinality threshold
            },
            # Advanced correlation detection
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True}, 
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},     # Categorical correlations
                "cramers": {"calculate": True}    # Categorical associations
            },
            # Missing data pattern analysis
            missing_diagrams={
                "heatmap": True,
                "dendrogram": True,
                "matrix": True
            },
            # Interaction detection
            interactions={
                "continuous": True,
                "targets": []
            }
        )
        
        profile_data = profile.get_description()
        alerts = profile_data.get('alerts', [])
        
        # Categorize alerts
        categorized_alerts = self._categorize_alerts(alerts)
        
        # Calculate quality score
        quality_score = self._calculate_enhanced_quality_score(alerts, df)
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(categorized_alerts, df)
        
        return EnhancedDataQualityProfile(
            table_name=table_name,
            overall_quality_score=quality_score,
            alert_summary=self._get_alert_summary(categorized_alerts),
            critical_alerts=categorized_alerts['critical'],
            warning_alerts=categorized_alerts['warning'],
            info_alerts=categorized_alerts['info'],
            quality_recommendations=recommendations,
            trend_alerts=self._extract_trend_alerts(alerts),
            distribution_alerts=self._extract_distribution_alerts(alerts),
            correlation_alerts=self._extract_correlation_alerts(alerts)
        )
    
    def _categorize_alerts(self, alerts: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize alerts by severity"""
        categorized = {'critical': [], 'warning': [], 'info': []}
        
        for alert in alerts:
            alert_type = alert.get('alert_type', '').lower()
            
            # Enhanced alert categorization
            if any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['critical']):
                categorized['critical'].append(alert)
            elif any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['warning']):
                categorized['warning'].append(alert)
            else:
                categorized['info'].append(alert)
        
        return categorized
    
    def _calculate_enhanced_quality_score(self, alerts: List[Dict], df: pd.DataFrame) -> float:
        """Calculate advanced quality score using ML-detected patterns"""
        base_score = 100.0
        
        # Deduct points based on alert severity
        for alert in alerts:
            alert_type = alert.get('alert_type', '').lower()
            
            if any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['critical']):
                base_score += self.ALERT_WEIGHTS['critical']
            elif any(keyword in alert_type for keyword in self.ALERT_CATEGORIES['warning']):
                base_score += self.ALERT_WEIGHTS['warning']
            else:
                base_score += self.ALERT_WEIGHTS['info']
        
        # Additional ML-based quality metrics
        completeness_bonus = self._calculate_completeness_bonus(df)
        consistency_bonus = self._calculate_consistency_bonus(df)
        
        final_score = max(0, min(100, base_score + completeness_bonus + consistency_bonus))
        return round(final_score, 2)
    
    def _generate_quality_recommendations(self, alerts: Dict, df: pd.DataFrame) -> List[str]:
        """Generate actionable quality improvement recommendations"""
        recommendations = []
        
        # Critical issues
        if alerts['critical']:
            recommendations.append("🚨 CRITICAL: Address constant columns, infinite values, or data type issues immediately")
        
        # High correlation recommendations
        correlation_alerts = [a for a in alerts['warning'] if 'correlation' in a.get('alert_type', '').lower()]
        if correlation_alerts:
            recommendations.append("⚠️ Consider feature selection to reduce high correlations between variables")
        
        # Missing data recommendations
        missing_alerts = [a for a in alerts['warning'] if 'missing' in a.get('alert_type', '').lower()]
        if missing_alerts:
            recommendations.append("📊 Implement data imputation strategy for missing values")
        
        # Skewness recommendations
        skew_alerts = [a for a in alerts['warning'] if 'skew' in a.get('alert_type', '').lower()]
        if skew_alerts:
            recommendations.append("📈 Consider data transformation to address skewed distributions")
        
        return recommendations
```

### **1.2 Advanced Outlier Detection and Anomaly Analysis**

**Enhancement:** Replace manual IQR-based outlier detection with ydata-profiling's advanced outlier detection:

```python
class MLOutlierAnalyzer:
    """Advanced outlier detection using ydata-profiling's ML capabilities"""
    
    def detect_outliers_comprehensive(self, df: pd.DataFrame, column_insights: List[ColumnInsight]) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple ML methods"""
        
        # Generate profile with outlier focus
        profile = ProfileReport(
            df,
            # Enable advanced outlier detection
            vars={
                'num': {
                    'low_categorical_threshold': 0,
                    'skewness_threshold': 0.5,  # Detect distribution skewness
                }
            },
            # Enable comprehensive statistical analysis
            explorative=True
        )
        
        profile_data = profile.get_description()
        variables = profile_data.get('variables', {})
        
        outlier_insights = {}
        
        for var_name, var_data in variables.items():
            if var_name in [insight.column_name for insight in column_insights if insight.semantic_role == SemanticRole.MEASURE]:
                
                outlier_info = {
                    'column_name': var_name,
                    'outlier_method': 'ydata_profiling_advanced',
                    'outlier_count': 0,
                    'outlier_percentage': 0,
                    'outlier_bounds': {},
                    'distribution_type': var_data.get('type', 'unknown'),
                    'skewness': var_data.get('skewness', 0),
                    'kurtosis': var_data.get('kurtosis', 0),
                    'outlier_impact': 'low'
                }
                
                # Extract outlier information from ydata-profiling
                if 'outliers' in var_data:
                    outlier_data = var_data['outliers']
                    outlier_info.update({
                        'outlier_count': outlier_data.get('count', 0),
                        'outlier_percentage': outlier_data.get('percentage', 0),
                        'outlier_bounds': {
                            'lower': outlier_data.get('lower_bound'),
                            'upper': outlier_data.get('upper_bound')
                        }
                    })
                
                # Determine outlier impact using ML-based analysis
                outlier_info['outlier_impact'] = self._assess_outlier_impact(
                    outlier_info['outlier_percentage'],
                    outlier_info['skewness'],
                    var_data
                )
                
                outlier_insights[var_name] = outlier_info
        
        return {
            'outlier_summary': outlier_insights,
            'high_impact_outliers': self._identify_high_impact_outliers(outlier_insights),
            'outlier_recommendations': self._generate_outlier_recommendations(outlier_insights)
        }
    
    def _assess_outlier_impact(self, outlier_percentage: float, skewness: float, var_data: Dict) -> str:
        """Assess the impact of outliers using ML insights"""
        if outlier_percentage > 10:
            return 'high'
        elif outlier_percentage > 5 or abs(skewness) > 2:
            return 'medium'
        else:
            return 'low'
```

## **2. Advanced Correlation and Interaction Analysis**

**Enhancement:** Leverage multiple correlation methods and interaction detection:

```python
class AdvancedCorrelationAnalyzer:
    """Advanced correlation analysis using ydata-profiling's multiple correlation methods"""
    
    def analyze_advanced_correlations(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Comprehensive correlation analysis using all available methods"""
        
        profile = ProfileReport(
            df,
            title=f"Correlation Analysis: {table_name}",
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},       # Linear relationships
                "spearman": {"calculate": True},      # Monotonic relationships  
                "kendall": {"calculate": True},       # Ordinal relationships
                "phi_k": {"calculate": True},         # Categorical correlations
                "cramers": {"calculate": True}        # Categorical associations
            },
            interactions={
                "continuous": True,                   # Detect non-linear interactions
                "targets": []
            }
        )
        
        profile_data = profile.get_description()
        correlations = profile_data.get('correlations', {})
        interactions = profile_data.get('interactions', {})
        
        # Analyze correlation patterns
        correlation_insights = {
            'linear_relationships': self._analyze_linear_correlations(correlations.get('pearson', {})),
            'monotonic_relationships': self._analyze_monotonic_correlations(correlations.get('spearman', {})),
            'categorical_associations': self._analyze_categorical_correlations(
                correlations.get('phi_k', {}), 
                correlations.get('cramers', {})
            ),
            'non_linear_interactions': self._analyze_interactions(interactions),
            'correlation_strength_matrix': self._build_correlation_matrix(correlations),
            'feature_redundancy': self._detect_feature_redundancy(correlations),
            'relationship_recommendations': []
        }
        
        # Generate insights and recommendations
        correlation_insights['relationship_recommendations'] = self._generate_correlation_recommendations(
            correlation_insights
        )
        
        return correlation_insights
    
    def _analyze_linear_correlations(self, pearson_corr: Dict) -> Dict[str, Any]:
        """Analyze linear relationships using Pearson correlations"""
        strong_correlations = []
        moderate_correlations = []
        
        for pair, corr_value in pearson_corr.items():
            if isinstance(corr_value, (int, float)) and not np.isnan(corr_value):
                abs_corr = abs(corr_value)
                
                if abs_corr >= 0.8:
                    strong_correlations.append({
                        'variables': pair,
                        'correlation': corr_value,
                        'strength': 'strong',
                        'type': 'positive' if corr_value > 0 else 'negative'
                    })
                elif abs_corr >= 0.5:
                    moderate_correlations.append({
                        'variables': pair,
                        'correlation': corr_value,
                        'strength': 'moderate',
                        'type': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return {
            'strong_linear': strong_correlations,
            'moderate_linear': moderate_correlations,
            'linear_relationship_count': len(strong_correlations) + len(moderate_correlations)
        }
    
    def _detect_feature_redundancy(self, correlations: Dict) -> List[Dict[str, Any]]:
        """Detect redundant features using correlation analysis"""
        redundant_groups = []
        
        # Check Pearson correlations for high linear redundancy
        pearson_corr = correlations.get('pearson', {})
        
        for pair, corr_value in pearson_corr.items():
            if isinstance(corr_value, (int, float)) and abs(corr_value) >= 0.95:
                redundant_groups.append({
                    'variables': pair,
                    'correlation_type': 'pearson',
                    'correlation_value': corr_value,
                    'redundancy_level': 'high',
                    'recommendation': 'Consider removing one variable to reduce multicollinearity'
                })
        
        return redundant_groups
```

## **3. Enhanced Time-Series Intelligence**

**Enhancement:** Leverage ydata-profiling's time-series capabilities:

```python
class TimeSeriesIntelligenceLayer:
    """Advanced time-series analysis using ydata-profiling's temporal capabilities"""
    
    def analyze_temporal_patterns(self, df: pd.DataFrame, temporal_columns: List[str], table_name: str) -> Dict[str, Any]:
        """Comprehensive temporal pattern analysis"""
        
        if not temporal_columns:
            return {}
        
        # Configure for time-series analysis
        primary_time_col = temporal_columns[0]
        
        # Ensure datetime format
        if primary_time_col in df.columns:
            df = df.copy()
            df[primary_time_col] = pd.to_datetime(df[primary_time_col], errors='coerce')
        
        profile = ProfileReport(
            df,
            title=f"Time-Series Analysis: {table_name}",
            tsmode=True,                          # Enable time-series mode
            sortby=primary_time_col,              # Primary temporal column
            explorative=True,
            # Enhanced time-series analysis
            vars={
                'timeseries': {
                    'active': True,
                    'autocorrelation': {'calculate': True},
                    'pacf': {'calculate': True}
                }
            }
        )
        
        profile_data = profile.get_description()
        
        temporal_insights = {
            'temporal_overview': self._extract_temporal_overview(profile_data, primary_time_col),
            'seasonality_patterns': self._analyze_seasonality(profile_data),
            'trend_analysis': self._analyze_trends(profile_data),
            'autocorrelation_insights': self._analyze_autocorrelation(profile_data),
            'time_gaps': self._analyze_time_gaps(profile_data),
            'temporal_data_quality': self._assess_temporal_quality(profile_data),
            'forecasting_readiness': self._assess_forecasting_readiness(profile_data)
        }
        
        return temporal_insights
    
    def _analyze_seasonality(self, profile_data: Dict) -> Dict[str, Any]:
        """Analyze seasonal patterns using ydata-profiling insights"""
        seasonality_info = {
            'has_seasonality': False,
            'seasonal_periods': [],
            'seasonal_strength': 0.0,
            'seasonal_patterns': {}
        }
        
        # Extract seasonality information from profile
        variables = profile_data.get('variables', {})
        
        for var_name, var_data in variables.items():
            if 'timeseries' in var_data:
                ts_data = var_data['timeseries']
                
                if 'seasonality' in ts_data:
                    seasonality_info['has_seasonality'] = True
                    seasonality_info['seasonal_periods'].append(ts_data['seasonality'])
                    seasonality_info['seasonal_patterns'][var_name] = ts_data['seasonality']
        
        return seasonality_info
    
    def _assess_forecasting_readiness(self, profile_data: Dict) -> Dict[str, Any]:
        """Assess readiness for forecasting models"""
        readiness_score = 0
        readiness_factors = []
        
        # Check stationarity
        variables = profile_data.get('variables', {})
        non_stationary_count = 0
        
        for var_name, var_data in variables.items():
            if 'timeseries' in var_data:
                if var_data['timeseries'].get('is_stationary', True) == False:
                    non_stationary_count += 1
        
        if non_stationary_count == 0:
            readiness_score += 30
            readiness_factors.append("All time series are stationary")
        else:
            readiness_factors.append(f"{non_stationary_count} non-stationary series detected")
        
        # Check for missing values
        missing_percentage = profile_data.get('missing', {}).get('percentage', 0)
        if missing_percentage < 5:
            readiness_score += 25
            readiness_factors.append("Low missing data percentage")
        
        # Check for outliers
        # Add outlier analysis here
        
        return {
            'forecasting_readiness_score': readiness_score,
            'readiness_factors': readiness_factors,
            'recommendations': self._generate_forecasting_recommendations(readiness_score, readiness_factors)
        }
```

## **4. Enhanced Table Classification System**

**Enhancement:** Use ML-detected patterns for smarter table classification:

```python
class MLTableClassifier:
    """ML-powered table classification using ydata-profiling insights"""
    
    def classify_table_advanced(self, df: pd.DataFrame, table_name: str, column_insights: List[ColumnInsight]) -> Dict[str, Any]:
        """Advanced table classification using ML-detected patterns"""
        
        # Generate comprehensive profile
        profile = ProfileReport(
            df,
            title=f"Classification Analysis: {table_name}",
            explorative=True,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True}
            }
        )
        
        profile_data = profile.get_description()
        
        # Extract ML features for classification
        classification_features = self._extract_classification_features(profile_data, column_insights)
        
        # Perform multi-dimensional classification
        classification_result = {
            'primary_type': self._classify_primary_type(classification_features),
            'business_domain': self._classify_business_domain(classification_features, table_name),
            'data_pattern': self._classify_data_pattern(classification_features),
            'complexity_level': self._assess_complexity(classification_features),
            'ml_readiness': self._assess_ml_readiness(classification_features),
            'classification_confidence': 0.0,
            'classification_evidence': []
        }
        
        # Calculate confidence score
        classification_result['classification_confidence'] = self._calculate_classification_confidence(
            classification_features, classification_result
        )
        
        return classification_result
    
    def _extract_classification_features(self, profile_data: Dict, column_insights: List[ColumnInsight]) -> Dict[str, Any]:
        """Extract features for ML-based table classification"""
        
        variables = profile_data.get('variables', {})
        correlations = profile_data.get('correlations', {})
        
        features = {
            # Structural features
            'row_count': len(profile_data.get('sample', {})),
            'column_count': len(variables),
            'unique_ratio_avg': np.mean([var.get('n_distinct', 0) / max(var.get('count', 1), 1) 
                                       for var in variables.values()]),
            
            # Data type distribution
            'numeric_column_ratio': len([col for col in column_insights 
                                       if col.data_type in [DataType.INTEGER, DataType.FLOAT]]) / len(column_insights),
            'categorical_column_ratio': len([col for col in column_insights 
                                           if col.data_type == DataType.CATEGORICAL]) / len(column_insights),
            'identifier_column_ratio': len([col for col in column_insights 
                                          if col.semantic_role == SemanticRole.IDENTIFIER]) / len(column_insights),
            
            # Semantic role distribution
            'measure_column_count': len([col for col in column_insights 
                                       if col.semantic_role == SemanticRole.MEASURE]),
            'dimension_column_count': len([col for col in column_insights 
                                         if col.semantic_role == SemanticRole.DIMENSION]),
            
            # Correlation patterns
            'high_correlation_count': self._count_high_correlations(correlations),
            'correlation_density': self._calculate_correlation_density(correlations),
            
            # Data quality indicators
            'missing_data_percentage': profile_data.get('missing', {}).get('percentage', 0),
            'duplicate_percentage': profile_data.get('duplicates', {}).get('percentage', 0)
        }
        
        return features
    
    def _classify_primary_type(self, features: Dict[str, Any]) -> str:
        """Classify primary table type using ML-detected features"""
        
        # Fact table indicators
        if (features['measure_column_count'] >= features['dimension_column_count'] and
            features['correlation_density'] > 0.3 and
            features['row_count'] > 1000):
            return 'fact'
        
        # Dimension table indicators  
        elif (features['dimension_column_count'] > features['measure_column_count'] and
              features['identifier_column_ratio'] >= 0.1 and
              features['unique_ratio_avg'] < 0.5):
            return 'dimension'
        
        # Bridge table indicators
        elif (features['identifier_column_ratio'] >= 0.4 and
              features['measure_column_count'] == 0 and
              features['correlation_density'] > 0.5):
            return 'bridge'
        
        # Temporal/log table indicators
        elif features['row_count'] > 10000 and features['correlation_density'] < 0.2:
            return 'temporal'
        
        else:
            return 'analytical'
```

## **5. Integration with Enhanced TableProfile**

**Enhancement:** Update the TableProfile dataclass to include all new insights:

```python
@dataclass
class EnhancedTableProfile:
    """Enhanced table profile with comprehensive ML insights"""
    # Existing fields
    table_name: str
    row_count: int
    column_count: int
    semantic_summary: str
    key_concepts: List[str]
    business_domain: Optional[str]
    table_type: str
    measure_columns: List[str]
    dimension_columns: List[str]
    identifier_columns: List[str]
    temporal_columns: List[str]
    data_quality_score: float
    embedding: Optional[np.ndarray] = None
    
    # NEW ENHANCED FIELDS
    # Data Quality Intelligence
    quality_profile: Optional[EnhancedDataQualityProfile] = None
    
    # Outlier Analysis
    outlier_analysis: Optional[Dict[str, Any]] = None
    
    # Advanced Correlations
    correlation_analysis: Optional[Dict[str, Any]] = None
    
    # Time-Series Intelligence (if applicable)
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # ML Classification Results
    classification_results: Optional[Dict[str, Any]] = None
    
    # Duplicate Analysis
    duplicate_analysis: Optional[Dict[str, Any]] = None
    
    # Missing Data Pattern Analysis
    missing_pattern_analysis: Optional[Dict[str, Any]] = None
    
    # Distribution Analysis
    distribution_analysis: Optional[Dict[str, Any]] = None
    
    # Interaction Analysis
    interaction_analysis: Optional[Dict[str, Any]] = None
    
    # ML Readiness Assessment
    ml_readiness_score: Optional[float] = None
    ml_readiness_factors: Optional[List[str]] = None
```

## **6. Enhanced TableIntelligenceLayer Integration**

**Enhancement:** Update the main class to incorporate all new capabilities:

```python
class EnhancedTableIntelligenceLayer(TableIntelligenceLayer):
    """Enhanced table intelligence with comprehensive ML capabilities"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 enable_profiling: bool = True,
                 cache_embeddings: bool = True,
                 use_llm_summaries: bool = False,
                 llm_config: Optional[LLMConfig] = None,
                 # NEW PARAMETERS
                 enable_advanced_quality: bool = True,
                 enable_outlier_detection: bool = True,
                 enable_correlation_analysis: bool = True,
                 enable_temporal_analysis: bool = True,
                 enable_ml_classification: bool = True):
        
        super().__init__(model_name, enable_profiling, cache_embeddings, use_llm_summaries, llm_config)
        
        # Initialize new analyzers
        self.quality_analyzer = AdvancedDataQualityAnalyzer() if enable_advanced_quality else None
        self.outlier_analyzer = MLOutlierAnalyzer() if enable_outlier_detection else None
        self.correlation_analyzer = AdvancedCorrelationAnalyzer() if enable_correlation_analysis else None
        self.temporal_analyzer = TimeSeriesIntelligenceLayer() if enable_temporal_analysis else None
        self.ml_classifier = MLTableClassifier() if enable_ml_classification else None
    
    def analyze_table_comprehensive(self, 
                                  table_name: str, 
                                  df: pd.DataFrame,
                                  schema_info: Optional[Dict] = None) -> EnhancedTableProfile:
        """Comprehensive table analysis with all ML capabilities"""
        
        self.logger.info(f"Starting comprehensive analysis for table: {table_name}")
        
        # Perform base analysis
        base_profile = super().analyze_table(table_name, df, schema_info)
        
        # Initialize enhanced profile
        enhanced_profile = EnhancedTableProfile(
            **asdict(base_profile)
        )
        
        # Advanced Data Quality Analysis
        if self.quality_analyzer:
            self.logger.info("Performing advanced data quality analysis...")
            enhanced_profile.quality_profile = self.quality_analyzer.analyze_data_quality(df, table_name)
        
        # Outlier Detection
        if self.outlier_analyzer:
            self.logger.info("Performing ML-based outlier detection...")
            enhanced_profile.outlier_analysis = self.outlier_analyzer.detect_outliers_comprehensive(
                df, enhanced_profile.column_insights
            )
        
        # Advanced Correlation Analysis
        if self.correlation_analyzer:
            self.logger.info("Performing advanced correlation analysis...")
            enhanced_profile.correlation_analysis = self.correlation_analyzer.analyze_advanced_correlations(
                df, table_name
            )
        
        # Temporal Analysis (if applicable)
        if self.temporal_analyzer and enhanced_profile.temporal_columns:
            self.logger.info("Performing time-series analysis...")
            enhanced_profile.temporal_analysis = self.temporal_analyzer.analyze_temporal_patterns(
                df, enhanced_profile.temporal_columns, table_name
            )
        
        # ML Classification
        if self.ml_classifier:
            self.logger.info("Performing ML-based table classification...")
            enhanced_profile.classification_results = self.ml_classifier.classify_table_advanced(
                df, table_name, enhanced_profile.column_insights
            )
        
        # ML Readiness Assessment
        enhanced_profile.ml_readiness_score, enhanced_profile.ml_readiness_factors = self._assess_ml_readiness(
            enhanced_profile
        )
        
        self.logger.info(f"Comprehensive analysis completed for table: {table_name}")
        return enhanced_profile
    
    def _assess_ml_readiness(self, profile: EnhancedTableProfile) -> Tuple[float, List[str]]:
        """Assess ML readiness based on comprehensive analysis"""
        readiness_score = 0.0
        factors = []
        
        # Data quality factor (30%)
        if profile.quality_profile:
            quality_score = profile.quality_profile.overall_quality_score
            readiness_score += (quality_score / 100) * 30
            
            if quality_score >= 80:
                factors.append("✅ High data quality score")
            elif quality_score >= 60:
                factors.append("⚠️ Moderate data quality - may need cleaning")
            else:
                factors.append("❌ Low data quality - requires significant cleaning")
        
        # Missing data factor (20%)
        missing_percentage = profile.data_quality_score  # Assuming this reflects missing data
        if missing_percentage >= 0.9:
            readiness_score += 20
            factors.append("✅ Low missing data percentage")
        elif missing_percentage >= 0.7:
            readiness_score += 10
            factors.append("⚠️ Moderate missing data")
        else:
            factors.append("❌ High missing data percentage")
        
        # Feature diversity factor (25%)
        if len(profile.measure_columns) >= 3 and len(profile.dimension_columns) >= 2:
            readiness_score += 25
            factors.append("✅ Good feature diversity")
        elif len(profile.measure_columns) >= 1 and len(profile.dimension_columns) >= 1:
            readiness_score += 15
            factors.append("⚠️ Moderate feature diversity")
        else:
            factors.append("❌ Limited feature diversity")
        
        # Data volume factor (15%)
        if profile.row_count >= 10000:
            readiness_score += 15
            factors.append("✅ Sufficient data volume for ML")
        elif profile.row_count >= 1000:
            readiness_score += 10
            factors.append("⚠️ Moderate data volume")
        else:
            factors.append("❌ Limited data volume for ML")
        
        # Correlation structure factor (10%)
        if profile.correlation_analysis:
            moderate_corr_count = len(profile.correlation_analysis.get('linear_relationships', {}).get('moderate_linear', []))
            if moderate_corr_count >= 3:
                readiness_score += 10
                factors.append("✅ Good feature relationships detected")
            elif moderate_corr_count >= 1:
                readiness_score += 5
                factors.append("⚠️ Some feature relationships detected")
            else:
                factors.append("❌ Limited feature relationships")
        
        return round(readiness_score, 2), factors
```

## **7. Implementation Strategy**

### **Phase 1: Core Enhancement (Week 1-2)**
1. Implement `AdvancedDataQualityAnalyzer` 
2. Enhance alert categorization and scoring
3. Update `TableProfile` dataclass

### **Phase 2: Advanced Analytics (Week 3-4)**
1. Implement `MLOutlierAnalyzer` and `AdvancedCorrelationAnalyzer`
2. Add time-series capabilities
3. Implement ML-based table classification

### **Phase 3: Integration (Week 5-6)**
1. Create `EnhancedTableIntelligenceLayer`
2. Integrate all analyzers
3. Add comprehensive testing

### **Phase 4: LLM Enhancement (Week 7-8)**
1. Enhance LLM prompts with new insights
2. Generate business-friendly explanations
3. Add recommendation systems

This comprehensive enhancement strategy transforms your `table_intelligence.py` from a basic profiling tool into a sophisticated ML-powered data intelligence system that fully leverages ydata-profiling's advanced capabilities while maintaining backward compatibility and adding significant new value for data analysis and ML preparation.

====================================================================================

Based on your test results, I can see that the enhanced table intelligence is working well and generating rich profiling data. This presents an excellent opportunity to create sophisticated LLM-powered metadata using Ollama! Here are my thoughts on how to leverage this profiling data:

## **1. Enhanced LLM Prompt Engineering with Rich Profiling Context**

The current `LLMSemanticSummarizer` can be significantly enhanced by incorporating the detailed profiling data:

```python
class EnhancedLLMSemanticSummarizer(LLMSemanticSummarizer):
    """Enhanced LLM summarizer using comprehensive profiling insights"""
    
    def generate_rich_metadata(self, 
                              table_name: str,
                              enhanced_profile: EnhancedTableProfile) -> Dict[str, Any]:
        """Generate comprehensive metadata using all profiling insights"""
        
        # Create rich context from profiling data
        profiling_context = self._build_profiling_context(enhanced_profile)
        
        # Generate different types of metadata
        metadata = {
            'business_summary': self._generate_business_summary(table_name, profiling_context),
            'data_quality_narrative': self._generate_quality_narrative(enhanced_profile.quality_profile),
            'ml_readiness_assessment': self._generate_ml_assessment(enhanced_profile),
            'relationship_insights': self._generate_relationship_insights(enhanced_profile.correlation_analysis),
            'anomaly_explanation': self._generate_anomaly_explanation(enhanced_profile.outlier_analysis),
            'usage_recommendations': self._generate_usage_recommendations(enhanced_profile),
            'technical_documentation': self._generate_technical_docs(enhanced_profile)
        }
        
        return metadata
    
    def _build_profiling_context(self, profile: EnhancedTableProfile) -> Dict[str, Any]:
        """Build comprehensive context from profiling data"""
        context = {
            'basic_info': {
                'table_name': profile.table_name,
                'rows': f"{profile.row_count:,}",
                'columns': profile.column_count,
                'business_domain': profile.business_domain,
                'table_type': profile.table_type
            },
            
            'data_quality': {
                'overall_score': profile.quality_profile.overall_quality_score if profile.quality_profile else None,
                'critical_issues': len(profile.quality_profile.critical_alerts) if profile.quality_profile else 0,
                'warning_issues': len(profile.quality_profile.warning_alerts) if profile.quality_profile else 0,
                'recommendations': profile.quality_profile.quality_recommendations if profile.quality_profile else []
            },
            
            'column_composition': {
                'measures': profile.measure_columns,
                'dimensions': profile.dimension_columns,
                'identifiers': profile.identifier_columns,
                'temporal': profile.temporal_columns
            },
            
            'correlation_insights': {
                'strong_relationships': len(profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', [])) if profile.correlation_analysis else 0,
                'redundant_features': len(profile.correlation_analysis.get('feature_redundancy', [])) if profile.correlation_analysis else 0
            },
            
            'outlier_insights': {
                'high_impact_outliers': len(profile.outlier_analysis.get('high_impact_outliers', [])) if profile.outlier_analysis else 0,
                'outlier_recommendations': profile.outlier_analysis.get('outlier_recommendations', []) if profile.outlier_analysis else []
            },
            
            'ml_readiness': {
                'score': profile.ml_readiness_score,
                'factors': profile.ml_readiness_factors
            }
        }
        
        return context
```

## **2. Specialized Prompt Templates for Rich Metadata Generation**

Extend the `PromptTemplateManager` with sophisticated templates:

```python
class RichMetadataPromptTemplates:
    """Advanced prompt templates for comprehensive metadata generation"""
    
    def __init__(self):
        self.templates = {
            'business_summary': {
                'system': '''You are a senior data analyst and business intelligence expert. 
                Generate comprehensive, business-focused summaries that executives and stakeholders can understand.''',
                'user_template': '''Analyze this database table and create a comprehensive business summary.

TABLE ANALYSIS:
{basic_info}

DATA QUALITY INSIGHTS:
- Overall Quality Score: {quality_score}/100
- Critical Issues: {critical_issues}
- Warning Issues: {warning_issues}
- Key Recommendations: {quality_recommendations}

COLUMN COMPOSITION:
- Business Measures: {measures}
- Business Dimensions: {dimensions}
- Identifiers: {identifiers}
- Time-based Columns: {temporal}

RELATIONSHIP PATTERNS:
- Strong Correlations Found: {strong_relationships}
- Redundant Features: {redundant_features}

ANOMALY DETECTION:
- High-Impact Outliers: {high_impact_outliers}
- Outlier Patterns: {outlier_recommendations}

ML READINESS:
- Score: {ml_score}/100
- Key Factors: {ml_factors}

Generate a comprehensive business summary that includes:
1. **Business Purpose**: What this table represents in business terms
2. **Data Quality Assessment**: Current state and improvement opportunities
3. **Key Insights**: Important patterns, relationships, and anomalies discovered
4. **Business Value**: How this data can drive business decisions
5. **Risk Assessment**: Data quality risks and mitigation strategies
6. **Recommended Actions**: Specific steps to improve data utility

Write in executive-friendly language, focusing on business impact rather than technical details.'''
            },
            
            'ml_strategy_recommendation': {
                'system': '''You are a machine learning architect and data scientist. 
                Provide strategic ML recommendations based on comprehensive data profiling.''',
                'user_template': '''Based on comprehensive data profiling, provide ML strategy recommendations.

DATA PROFILE:
{profiling_context}

SPECIFIC ANALYSIS:
- Data Volume: {row_count} records
- Feature Diversity: {feature_diversity}
- Quality Score: {quality_score}/100
- Missing Data: {missing_percentage}%
- Outlier Impact: {outlier_impact}
- Correlation Density: {correlation_density}

Generate strategic ML recommendations covering:
1. **ML Readiness Assessment**: Current state for ML applications
2. **Recommended ML Use Cases**: Specific ML applications suited for this data
3. **Data Preparation Strategy**: Required preprocessing steps
4. **Feature Engineering Opportunities**: How to enhance predictive power
5. **Model Architecture Suggestions**: Suitable ML model types
6. **Risk Mitigation**: Data quality risks for ML models
7. **Performance Expectations**: Realistic accuracy expectations
8. **Implementation Roadmap**: Step-by-step ML deployment strategy

Focus on actionable, practical recommendations.'''
            },
            
            'data_story_narrative': {
                'system': '''You are a data storyteller who transforms complex data insights into compelling narratives.''',
                'user_template': '''Transform this data profiling analysis into a compelling data story.

PROFILING INSIGHTS:
{full_profiling_context}

Create a narrative that includes:
1. **The Data Journey**: How this data tells a business story
2. **Hidden Patterns**: Surprising insights from correlation and outlier analysis
3. **Quality Narrative**: The data quality story and its implications
4. **Predictive Potential**: What this data can predict about the future
5. **Operational Impact**: How data quality affects business operations
6. **Strategic Opportunities**: Untapped potential in the data

Write as an engaging story that makes data insights accessible to non-technical stakeholders.'''
            }
        }
```

## **3. Advanced Metadata Generators**

Create specialized generators for different types of metadata:

```python
class AdvancedMetadataGenerator:
    """Generate sophisticated metadata using LLM and profiling insights"""
    
    def __init__(self, llm_summarizer: EnhancedLLMSemanticSummarizer):
        self.llm = llm_summarizer
        
    def generate_comprehensive_metadata(self, enhanced_profile: EnhancedTableProfile) -> Dict[str, Any]:
        """Generate all types of rich metadata"""
        
        return {
            'executive_summary': self._generate_executive_summary(enhanced_profile),
            'technical_specifications': self._generate_technical_specs(enhanced_profile),
            'ml_strategy_guide': self._generate_ml_strategy(enhanced_profile),
            'data_quality_report': self._generate_quality_report(enhanced_profile),
            'business_intelligence_insights': self._generate_bi_insights(enhanced_profile),
            'predictive_analytics_recommendations': self._generate_predictive_recs(enhanced_profile),
            'data_governance_guidelines': self._generate_governance_guide(enhanced_profile),
            'usage_patterns_analysis': self._generate_usage_patterns(enhanced_profile)
        }
    
    def _generate_executive_summary(self, profile: EnhancedTableProfile) -> Dict[str, Any]:
        """Generate executive-level summary"""
        context = self._build_executive_context(profile)
        
        summary = self.llm.generate_with_template(
            'executive_summary',
            context=context,
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            'summary': summary,
            'key_metrics': self._extract_key_metrics(profile),
            'risk_indicators': self._identify_risk_indicators(profile),
            'opportunity_score': self._calculate_opportunity_score(profile)
        }
    
    def _generate_ml_strategy(self, profile: EnhancedTableProfile) -> Dict[str, Any]:
        """Generate comprehensive ML strategy"""
        
        ml_context = {
            'readiness_score': profile.ml_readiness_score,
            'readiness_factors': profile.ml_readiness_factors,
            'feature_analysis': self._analyze_features_for_ml(profile),
            'quality_constraints': self._identify_ml_quality_constraints(profile),
            'correlation_opportunities': self._identify_correlation_opportunities(profile)
        }
        
        strategy = self.llm.generate_with_template(
            'ml_strategy_recommendation',
            context=ml_context,
            max_tokens=800,
            temperature=0.2
        )
        
        return {
            'strategy_document': strategy,
            'recommended_models': self._recommend_ml_models(profile),
            'preprocessing_pipeline': self._design_preprocessing_pipeline(profile),
            'feature_engineering_suggestions': self._suggest_feature_engineering(profile),
            'validation_strategy': self._design_validation_strategy(profile)
        }
    
    def _generate_data_quality_narrative(self, profile: EnhancedTableProfile) -> Dict[str, Any]:
        """Generate comprehensive data quality narrative"""
        
        if not profile.quality_profile:
            return {'narrative': 'No quality profile available'}
        
        quality_context = {
            'overall_score': profile.quality_profile.overall_quality_score,
            'alert_breakdown': profile.quality_profile.alert_summary,
            'critical_issues': profile.quality_profile.critical_alerts,
            'improvement_recommendations': profile.quality_profile.quality_recommendations,
            'trend_analysis': profile.quality_profile.trend_alerts,
            'distribution_issues': profile.quality_profile.distribution_alerts
        }
        
        narrative = self.llm.generate_with_template(
            'quality_narrative',
            context=quality_context,
            max_tokens=600,
            temperature=0.4
        )
        
        return {
            'quality_story': narrative,
            'improvement_roadmap': self._create_improvement_roadmap(profile.quality_profile),
            'quality_monitoring_suggestions': self._suggest_quality_monitoring(profile.quality_profile),
            'impact_assessment': self._assess_quality_impact(profile.quality_profile)
        }
```

## **4. Context-Aware Metadata Enrichment**

Use the profiling data to create context-aware enhancements:

```python
class ContextualMetadataEnricher:
    """Enrich metadata with contextual intelligence"""
    
    def enrich_with_business_context(self, metadata: Dict, profile: EnhancedTableProfile) -> Dict:
        """Add business context to metadata"""
        
        # Determine business criticality
        business_criticality = self._assess_business_criticality(profile)
        
        # Identify data lineage patterns
        lineage_insights = self._infer_data_lineage(profile)
        
        # Assess regulatory compliance implications
        compliance_assessment = self._assess_compliance_implications(profile)
        
        # Generate usage recommendations
        usage_recommendations = self._generate_contextual_usage_recommendations(profile)
        
        enriched_metadata = {
            **metadata,
            'business_context': {
                'criticality_level': business_criticality,
                'compliance_implications': compliance_assessment,
                'lineage_insights': lineage_insights,
                'usage_recommendations': usage_recommendations,
                'stakeholder_impact': self._assess_stakeholder_impact(profile),
                'integration_opportunities': self._identify_integration_opportunities(profile)
            }
        }
        
        return enriched_metadata
    
    def _assess_business_criticality(self, profile: EnhancedTableProfile) -> str:
        """Assess business criticality based on profiling data"""
        
        criticality_score = 0
        
        # High volume data is typically more critical
        if profile.row_count > 100000:
            criticality_score += 3
        elif profile.row_count > 10000:
            criticality_score += 2
        
        # Tables with financial measures are critical
        financial_indicators = ['price', 'amount', 'cost', 'revenue', 'payment', 'value']
        if any(indicator in str(profile.measure_columns).lower() for indicator in financial_indicators):
            criticality_score += 3
        
        # High quality data is more business-critical
        if profile.quality_profile and profile.quality_profile.overall_quality_score > 90:
            criticality_score += 2
        
        # Customer data is typically critical
        if 'customer' in profile.business_domain.lower() if profile.business_domain else False:
            criticality_score += 3
        
        if criticality_score >= 8:
            return 'CRITICAL'
        elif criticality_score >= 5:
            return 'HIGH'
        elif criticality_score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
```

## **5. Implementation Strategy for Your Existing Code**

Based on your test results, here's how to integrate this:

```python
# Enhance your existing EnhancedTableIntelligenceLayer
class EnhancedTableIntelligenceLayer(TableIntelligenceLayer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add rich metadata generators
        if self.use_llm_summaries and self.llm_summarizer:
            self.metadata_generator = AdvancedMetadataGenerator(self.llm_summarizer)
            self.context_enricher = ContextualMetadataEnricher()
    
    def analyze_table_comprehensive(self, 
                                  table_name: str, 
                                  df: pd.DataFrame,
                                  schema_info: Optional[Dict] = None,
                                  generate_rich_metadata: bool = True) -> EnhancedTableProfile:
        """Enhanced analysis with rich metadata generation"""
        
        # Perform existing comprehensive analysis
        enhanced_profile = super().analyze_table_comprehensive(table_name, df, schema_info)
        
        # Generate rich metadata if requested and LLM is available
        if generate_rich_metadata and self.metadata_generator:
            try:
                self.logger.info("Generating rich metadata using LLM...")
                
                # Generate comprehensive metadata
                rich_metadata = self.metadata_generator.generate_comprehensive_metadata(enhanced_profile)
                
                # Enrich with business context
                enriched_metadata = self.context_enricher.enrich_with_business_context(
                    rich_metadata, enhanced_profile
                )
                
                # Add to profile
                enhanced_profile.rich_metadata = enriched_metadata
                
                self.logger.info("Rich metadata generation completed")
                
            except Exception as e:
                self.logger.warning(f"Rich metadata generation failed: {e}")
                enhanced_profile.rich_metadata = None
        
        return enhanced_profile
```

## **6. Benefits for Your Olist E-commerce Dataset**

Based on your test results, this would provide:

1. **Business Intelligence**: "The orders table shows strong transactional patterns with 99K orders indicating healthy business volume. The high data quality score (93.97) suggests reliable business reporting capabilities."

2. **ML Strategy**: "With payment data showing 80/100 ML readiness, this dataset is excellent for fraud detection, customer lifetime value prediction, and payment behavior analysis."

3. **Data Quality Narratives**: "The order_items table demonstrates perfect data quality, making it ideal for revenue analysis and product performance insights."

4. **Risk Assessment**: "Missing values in order delivery dates could impact customer satisfaction analytics - recommend implementing data validation at collection point."

This approach transforms your technical profiling results into actionable business intelligence that stakeholders can understand and act upon!

Absolutely! The profiling data is a goldmine for building intelligent query contexts. Here's how we can leverage it for sophisticated query building:

## **Query Context Intelligence Using Profiling Data**

The profiling data can provide several key insights for query building:

1. **Smart Join Recommendations** - Using correlation analysis and foreign key detection
2. **Intelligent Filtering Suggestions** - Based on data distributions and outlier patterns
3. **Aggregation Strategy Guidance** - Using measure/dimension classification
4. **Performance Optimization Hints** - Based on cardinality and data volume
5. **Data Quality Warnings** - Alert users about potential data issues
6. **Semantic Query Understanding** - Natural language to SQL translation

## **Implementation: Query Context Builder**

```python
@dataclass
class QueryContext:
    """Rich context for intelligent query building"""
    table_name: str
    
    # Column intelligence
    queryable_measures: List[Dict[str, Any]]
    filter_dimensions: List[Dict[str, Any]]
    groupby_candidates: List[Dict[str, Any]]
    join_candidates: List[Dict[str, Any]]
    
    # Query optimization hints
    performance_hints: List[str]
    data_quality_warnings: List[str]
    
    # Semantic understanding
    business_concepts: List[str]
    common_patterns: List[Dict[str, Any]]
    suggested_queries: List[Dict[str, Any]]
    
    # Advanced insights
    correlation_insights: List[Dict[str, Any]]
    anomaly_patterns: List[Dict[str, Any]]
    temporal_patterns: Optional[Dict[str, Any]]

class IntelligentQueryContextBuilder:
    """Build rich query contexts from profiling data"""
    
    def __init__(self, llm_summarizer: Optional[LLMSemanticSummarizer] = None):
        self.llm_summarizer = llm_summarizer
        self.logger = logging.getLogger(__name__)
    
    def build_query_context(self, 
                           enhanced_profile: EnhancedTableProfile,
                           related_profiles: Optional[List[EnhancedTableProfile]] = None) -> QueryContext:
        """Build comprehensive query context from profiling data"""
        
        self.logger.info(f"Building query context for {enhanced_profile.table_name}")
        
        # Extract queryable measures with intelligence
        queryable_measures = self._extract_queryable_measures(enhanced_profile)
        
        # Extract intelligent filter dimensions
        filter_dimensions = self._extract_filter_dimensions(enhanced_profile)
        
        # Extract groupby candidates
        groupby_candidates = self._extract_groupby_candidates(enhanced_profile)
        
        # Find join opportunities
        join_candidates = self._find_join_candidates(enhanced_profile, related_profiles)
        
        # Generate performance hints
        performance_hints = self._generate_performance_hints(enhanced_profile)
        
        # Generate data quality warnings
        quality_warnings = self._generate_quality_warnings(enhanced_profile)
        
        # Extract business concepts
        business_concepts = self._extract_business_concepts(enhanced_profile)
        
        # Identify common patterns
        common_patterns = self._identify_common_patterns(enhanced_profile)
        
        # Generate suggested queries
        suggested_queries = self._generate_suggested_queries(enhanced_profile)
        
        # Extract correlation insights for query building
        correlation_insights = self._extract_correlation_insights(enhanced_profile)
        
        # Extract anomaly patterns
        anomaly_patterns = self._extract_anomaly_patterns(enhanced_profile)
        
        # Extract temporal patterns
        temporal_patterns = self._extract_temporal_patterns(enhanced_profile)
        
        return QueryContext(
            table_name=enhanced_profile.table_name,
            queryable_measures=queryable_measures,
            filter_dimensions=filter_dimensions,
            groupby_candidates=groupby_candidates,
            join_candidates=join_candidates,
            performance_hints=performance_hints,
            data_quality_warnings=quality_warnings,
            business_concepts=business_concepts,
            common_patterns=common_patterns,
            suggested_queries=suggested_queries,
            correlation_insights=correlation_insights,
            anomaly_patterns=anomaly_patterns,
            temporal_patterns=temporal_patterns
        )
    
    def _extract_queryable_measures(self, profile: EnhancedTableProfile) -> List[Dict[str, Any]]:
        """Extract measures with query intelligence"""
        measures = []
        
        for measure_col in profile.measure_columns:
            measure_info = {
                'column_name': measure_col,
                'data_type': self._get_column_data_type(measure_col, profile),
                'aggregation_functions': self._suggest_aggregation_functions(measure_col, profile),
                'outlier_risk': self._assess_outlier_risk(measure_col, profile),
                'null_handling': self._suggest_null_handling(measure_col, profile),
                'business_meaning': self._infer_business_meaning(measure_col, profile),
                'typical_ranges': self._get_typical_ranges(measure_col, profile),
                'seasonal_patterns': self._check_seasonal_patterns(measure_col, profile)
            }
            measures.append(measure_info)
        
        return measures
    
    def _extract_filter_dimensions(self, profile: EnhancedTableProfile) -> List[Dict[str, Any]]:
        """Extract dimensions with filtering intelligence"""
        dimensions = []
        
        for dim_col in profile.dimension_columns + profile.identifier_columns:
            dim_info = {
                'column_name': dim_col,
                'cardinality': self._get_cardinality_info(dim_col, profile),
                'filter_suggestions': self._generate_filter_suggestions(dim_col, profile),
                'common_values': self._get_common_values(dim_col, profile),
                'filter_performance': self._assess_filter_performance(dim_col, profile),
                'data_quality_issues': self._check_dimension_quality(dim_col, profile),
                'business_hierarchy': self._infer_business_hierarchy(dim_col, profile)
            }
            dimensions.append(dim_info)
        
        return dimensions
    
    def _find_join_candidates(self, 
                            profile: EnhancedTableProfile, 
                            related_profiles: Optional[List[EnhancedTableProfile]]) -> List[Dict[str, Any]]:
        """Find intelligent join opportunities"""
        if not related_profiles:
            return []
        
        join_candidates = []
        
        for related_profile in related_profiles:
            # Find potential join columns
            potential_joins = self._identify_potential_joins(profile, related_profile)
            
            for join_info in potential_joins:
                join_candidate = {
                    'target_table': related_profile.table_name,
                    'join_type': join_info['suggested_join_type'],
                    'join_columns': join_info['join_columns'],
                    'confidence': join_info['confidence'],
                    'performance_impact': join_info['performance_impact'],
                    'data_quality_risk': join_info['data_quality_risk'],
                    'business_relationship': join_info['business_relationship'],
                    'join_cardinality': join_info['join_cardinality']
                }
                join_candidates.append(join_candidate)
        
        # Sort by confidence and performance
        join_candidates.sort(key=lambda x: (x['confidence'], -x['performance_impact']), reverse=True)
        
        return join_candidates
    
    def _generate_performance_hints(self, profile: EnhancedTableProfile) -> List[str]:
        """Generate query performance optimization hints"""
        hints = []
        
        # Volume-based hints
        if profile.row_count > 1000000:
            hints.append("⚡ Large table detected - consider using LIMIT for exploratory queries")
            hints.append("📊 Use aggregate queries instead of detailed scans when possible")
        
        # Cardinality-based hints
        high_cardinality_dims = [col for col in profile.dimension_columns 
                               if self._is_high_cardinality(col, profile)]
        if high_cardinality_dims:
            hints.append(f"🔍 High cardinality columns detected: {', '.join(high_cardinality_dims[:3])} - avoid grouping without filtering")
        
        # Temporal optimization hints
        if profile.temporal_columns:
            hints.append("📅 Temporal columns available - use date range filters for better performance")
            hints.append("⏰ Consider time-based partitioning for large date ranges")
        
        # Correlation-based hints
        if profile.correlation_analysis:
            redundant_features = profile.correlation_analysis.get('feature_redundancy', [])
            if redundant_features:
                redundant_cols = [f"{r['variables']}" for r in redundant_features[:2]]
                hints.append(f"🔄 Redundant columns detected: {', '.join(redundant_cols)} - choose one for analysis")
        
        # Data quality hints
        if profile.quality_profile:
            if profile.quality_profile.overall_quality_score < 80:
                hints.append("⚠️ Data quality issues detected - consider filtering out problematic records")
        
        return hints
    
    def _generate_quality_warnings(self, profile: EnhancedTableProfile) -> List[str]:
        """Generate data quality warnings for query builders"""
        warnings = []
        
        if not profile.quality_profile:
            return warnings
        
        # Critical alerts warnings
        if profile.quality_profile.critical_alerts:
            warnings.append("🚨 CRITICAL: Data contains constant values, infinite values, or unsupported types")
        
        # Missing data warnings
        missing_alerts = [alert for alert in profile.quality_profile.warning_alerts 
                         if 'missing' in str(alert.get('alert_type', '')).lower()]
        if missing_alerts:
            warnings.append("📋 Missing data detected - results may be incomplete, consider NULL handling")
        
        # Skewness warnings
        skew_alerts = [alert for alert in profile.quality_profile.warning_alerts 
                      if 'skew' in str(alert.get('alert_type', '')).lower()]
        if skew_alerts:
            warnings.append("📈 Skewed data distributions detected - aggregations may be misleading")
        
        # High correlation warnings
        correlation_alerts = [alert for alert in profile.quality_profile.warning_alerts 
                            if 'correlation' in str(alert.get('alert_type', '')).lower()]
        if correlation_alerts:
            warnings.append("🔗 High correlations detected - avoid using correlated columns together")
        
        # Outlier warnings
        if profile.outlier_analysis:
            high_impact_outliers = profile.outlier_analysis.get('high_impact_outliers', [])
            if high_impact_outliers:
                outlier_cols = [o['column'] for o in high_impact_outliers[:3]]
                warnings.append(f"📊 Outliers detected in: {', '.join(outlier_cols)} - may skew results")
        
        return warnings
    
    def _generate_suggested_queries(self, profile: EnhancedTableProfile) -> List[Dict[str, Any]]:
        """Generate intelligent query suggestions based on profiling"""
        suggestions = []
        
        # Basic exploration queries
        if profile.measure_columns and profile.dimension_columns:
            suggestions.append({
                'type': 'summary_analysis',
                'title': 'Summary Analysis',
                'description': f'Get overview of key metrics by main dimensions',
                'sql_template': f'''SELECT 
    {', '.join(profile.dimension_columns[:2])},
    {', '.join([f'AVG({col}) as avg_{col}' for col in profile.measure_columns[:3]])}
FROM {profile.table_name}
GROUP BY {', '.join(profile.dimension_columns[:2])}
ORDER BY {profile.measure_columns[0] if profile.measure_columns else profile.dimension_columns[0]}''',
                'business_value': 'Understand key performance metrics across business dimensions'
            })
        
        # Time-based analysis if temporal columns exist
        if profile.temporal_columns and profile.measure_columns:
            suggestions.append({
                'type': 'temporal_trend',
                'title': 'Temporal Trend Analysis',
                'description': 'Analyze trends over time',
                'sql_template': f'''SELECT 
    DATE_TRUNC('month', {profile.temporal_columns[0]}) as time_period,
    {', '.join([f'SUM({col}) as total_{col}' for col in profile.measure_columns[:2]])}
FROM {profile.table_name}
WHERE {profile.temporal_columns[0]} >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', {profile.temporal_columns[0]})
ORDER BY time_period''',
                'business_value': 'Track business performance trends over time'
            })
        
        # Data quality investigation queries
        if profile.quality_profile and profile.quality_profile.overall_quality_score < 90:
            suggestions.append({
                'type': 'data_quality_check',
                'title': 'Data Quality Investigation',
                'description': 'Investigate data quality issues',
                'sql_template': f'''SELECT 
    COUNT(*) as total_records,
    {', '.join([f'COUNT({col}) as non_null_{col}' for col in (profile.measure_columns + profile.dimension_columns)[:4]])}
FROM {profile.table_name}''',
                'business_value': 'Understand data completeness and quality issues'
            })
        
        # Outlier investigation queries
        if profile.outlier_analysis and profile.outlier_analysis.get('high_impact_outliers'):
            outlier_col = profile.outlier_analysis['high_impact_outliers'][0]['column']
            suggestions.append({
                'type': 'outlier_analysis',
                'title': 'Outlier Investigation',
                'description': f'Investigate outliers in {outlier_col}',
                'sql_template': f'''SELECT 
    {outlier_col},
    {', '.join(profile.dimension_columns[:2]) if profile.dimension_columns else '*'}
FROM {profile.table_name}
WHERE {outlier_col} > (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {outlier_col})
    FROM {profile.table_name}
)
ORDER BY {outlier_col} DESC
LIMIT 20''',
                'business_value': 'Identify and investigate unusual data patterns'
            })
        
        # Business domain specific queries
        if profile.business_domain:
            domain_query = self._generate_domain_specific_query(profile)
            if domain_query:
                suggestions.append(domain_query)
        
        return suggestions
    
    def _generate_domain_specific_query(self, profile: EnhancedTableProfile) -> Optional[Dict[str, Any]]:
        """Generate domain-specific query suggestions"""
        
        if 'customer' in profile.business_domain.lower():
            return {
                'type': 'customer_analysis',
                'title': 'Customer Segmentation Analysis',
                'description': 'Analyze customer distribution and characteristics',
                'sql_template': f'''SELECT 
    {profile.dimension_columns[0] if profile.dimension_columns else 'customer_segment'},
    COUNT(*) as customer_count,
    {f'AVG({profile.measure_columns[0]}) as avg_value' if profile.measure_columns else 'COUNT(*) as total'}
FROM {profile.table_name}
GROUP BY {profile.dimension_columns[0] if profile.dimension_columns else 'customer_segment'}
ORDER BY customer_count DESC''',
                'business_value': 'Understand customer distribution and behavior patterns'
            }
        
        elif 'order' in profile.business_domain.lower():
            return {
                'type': 'order_analysis',
                'title': 'Order Performance Analysis',
                'description': 'Analyze order patterns and performance',
                'sql_template': f'''SELECT 
    {profile.dimension_columns[0] if profile.dimension_columns else 'order_status'},
    COUNT(*) as order_count,
    {f'SUM({profile.measure_columns[0]}) as total_value' if profile.measure_columns else 'COUNT(*) as total'}
FROM {profile.table_name}
GROUP BY {profile.dimension_columns[0] if profile.dimension_columns else 'order_status'}
ORDER BY order_count DESC''',
                'business_value': 'Monitor order volume and performance metrics'
            }
        
        elif 'product' in profile.business_domain.lower():
            return {
                'type': 'product_analysis',
                'title': 'Product Performance Analysis',
                'description': 'Analyze product sales and performance',
                'sql_template': f'''SELECT 
    {profile.dimension_columns[0] if profile.dimension_columns else 'product_category'},
    COUNT(*) as product_count,
    {f'AVG({profile.measure_columns[0]}) as avg_price' if profile.measure_columns else 'COUNT(*) as total'}
FROM {profile.table_name}
GROUP BY {profile.dimension_columns[0] if profile.dimension_columns else 'product_category'}
ORDER BY product_count DESC''',
                'business_value': 'Understand product portfolio performance'
            }
        
        return None
    
    def _extract_correlation_insights(self, profile: EnhancedTableProfile) -> List[Dict[str, Any]]:
        """Extract correlation insights for query optimization"""
        insights = []
        
        if not profile.correlation_analysis:
            return insights
        
        # Strong correlations - suggest using together or choosing one
        strong_correlations = profile.correlation_analysis.get('linear_relationships', {}).get('strong_linear', [])
        for corr in strong_correlations:
            insights.append({
                'type': 'strong_correlation',
                'variables': corr['variables'],
                'correlation': corr['correlation'],
                'suggestion': 'These variables are highly correlated - consider using one for analysis or both for validation',
                'query_impact': 'Can use either variable for similar insights'
            })
        
        # Feature redundancy - warn about using together
        redundant_features = profile.correlation_analysis.get('feature_redundancy', [])
        for redundant in redundant_features:
            insights.append({
                'type': 'redundancy_warning',
                'variables': redundant['variables'],
                'correlation': redundant['correlation_value'],
                'suggestion': 'These features are redundant - avoid using both in the same analysis',
                'query_impact': 'Using both may not provide additional insights'
            })
        
        return insights
```

## **Natural Language Query Understanding Enhancement**

```python
class NLQueryContextEnhancer:
    """Enhance natural language query understanding using profiling context"""
    
    def __init__(self, llm_summarizer: LLMSemanticSummarizer):
        self.llm = llm_summarizer
    
    def enhance_nl_query_understanding(self, 
                                     query_text: str, 
                                     query_context: QueryContext) -> Dict[str, Any]:
        """Enhance natural language query understanding with profiling context"""
        
        # Build context for LLM
        context = self._build_nl_context(query_context)
        
        # Generate enhanced understanding
        enhanced_understanding = self.llm.generate_with_template(
            'nl_query_enhancement',
            query_text=query_text,
            table_context=context,
            max_tokens=600,
            temperature=0.2
        )
        
        return {
            'original_query': query_text,
            'enhanced_understanding': enhanced_understanding,
            'suggested_columns': self._suggest_relevant_columns(query_text, query_context),
            'potential_joins': self._suggest_joins_for_query(query_text, query_context),
            'performance_warnings': self._generate_performance_warnings(query_text, query_context),
            'data_quality_considerations': self._generate_quality_considerations(query_text, query_context)
        }
    
    def _build_nl_context(self, query_context: QueryContext) -> Dict[str, Any]:
        """Build context for natural language understanding"""
        return {
            'table_name': query_context.table_name,
            'available_measures': [m['column_name'] for m in query_context.queryable_measures],
            'available_dimensions': [d['column_name'] for d in query_context.filter_dimensions],
            'business_concepts': query_context.business_concepts,
            'common_patterns': [p['description'] for p in query_context.common_patterns],
            'join_opportunities': [j['target_table'] for j in query_context.join_candidates],
            'performance_considerations': query_context.performance_hints,
            'data_quality_warnings': query_context.data_quality_warnings
        }
```

## **Query Execution Context Provider**

```python
class QueryExecutionContextProvider:
    """Provide execution context for query optimization"""
    
    def __init__(self, query_context_builder: IntelligentQueryContextBuilder):
        self.context_builder = query_context_builder
    
    def get_execution_context(self, 
                            sql_query: str, 
                            query_context: QueryContext) -> Dict[str, Any]:
        """Get execution context for query optimization"""
        
        # Parse SQL to understand query structure
        query_structure = self._parse_query_structure(sql_query)
        
        # Generate optimization suggestions
        optimizations = self._generate_optimizations(query_structure, query_context)
        
        # Generate data quality warnings
        quality_warnings = self._generate_execution_warnings(query_structure, query_context)
        
        # Estimate query complexity
        complexity_assessment = self._assess_query_complexity(query_structure, query_context)
        
        return {
            'query_structure': query_structure,
            'optimization_suggestions': optimizations,
            'data_quality_warnings': quality_warnings,
            'complexity_assessment': complexity_assessment,
            'expected_result_characteristics': self._predict_result_characteristics(query_structure, query_context)
        }
```

## **Integration with Your Existing System**

Based on your Olist test results, here's how to integrate:

```python
# Enhanced usage in your existing system
def test_query_context_building():
    """Test query context building with your Olist data"""
    
    # Initialize enhanced intelligence layer with query context
    intelligence = EnhancedTableIntelligenceLayer(
        enable_profiling=True,
        enable_advanced_quality=True,
        enable_correlation_analysis=True,
        use_llm_summaries=True  # Enable for rich context
    )
    
    # Build query context builder
    query_context_builder = IntelligentQueryContextBuilder(intelligence.llm_summarizer)
    
    # Analyze your tables and build contexts
    profiles = {}
    query_contexts = {}
    
    for table_name in ['orders', 'order_items', 'order_payments', 'customers']:
        df = load_table_data(table_name)  # Your data loading function
        
        # Get enhanced profile
        profile = intelligence.analyze_table_comprehensive(table_name, df)
        profiles[table_name] = profile
        
        # Build query context
        related_profiles = [p for name, p in profiles.items() if name != table_name]
        query_context = query_context_builder.build_query_context(profile, related_profiles)
        query_contexts[table_name] = query_context
        
        print(f"\n=== Query Context for {table_name.upper()} ===")
        print(f"Queryable Measures: {len(query_context.queryable_measures)}")
        print(f"Filter Dimensions: {len(query_context.filter_dimensions)}")
        print(f"Join Candidates: {len(query_context.join_candidates)}")
        print(f"Suggested Queries: {len(query_context.suggested_queries)}")
        
        # Show some suggestions
        for suggestion in query_context.suggested_queries[:2]:
            print(f"\n📊 {suggestion['title']}: {suggestion['description']}")
```

This approach transforms your profiling data into actionable query intelligence that can power sophisticated query builders, natural language interfaces, and query optimization systems!