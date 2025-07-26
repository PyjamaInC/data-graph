import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MLRelationshipDetector:
    def __init__(self):
        self.feature_extractors = {
            'statistical': self._extract_statistical_features,
            'semantic': self._extract_semantic_features,
            'structural': self._extract_structural_features
        }
        
    def detect_relationships(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                           col1: str, col2: str, table1: str, table2: str) -> Dict[str, Any]:
        """Comprehensive relationship detection between two columns"""
        
        series1, series2 = df1[col1].dropna(), df2[col2].dropna()
        
        if len(series1) == 0 or len(series2) == 0:
            return {'relationship_type': 'NO_DATA', 'confidence': 0.0}
        
        # Extract features
        features = {}
        for feature_type, extractor in self.feature_extractors.items():
            try:
                features.update(extractor(series1, series2, col1, col2, table1, table2))
            except Exception as e:
                print(f"Error extracting {feature_type} features: {e}")
                continue
        
        # Determine relationship type and confidence
        relationship_type, confidence = self._classify_relationship(features, series1, series2)
        
        return {
            'relationship_type': relationship_type,
            'confidence': confidence,
            'features': features,
            'evidence': self._generate_evidence(features, relationship_type)
        }
    
    def _extract_statistical_features(self, series1: pd.Series, series2: pd.Series, 
                                    col1: str, col2: str, table1: str, table2: str) -> Dict[str, float]:
        """Extract statistical features for relationship detection"""
        features = {}
        
        # Data type compatibility
        features['same_dtype'] = float(series1.dtype == series2.dtype)
        features['both_numeric'] = float(pd.api.types.is_numeric_dtype(series1) and 
                                       pd.api.types.is_numeric_dtype(series2))
        features['both_categorical'] = float(pd.api.types.is_categorical_dtype(series1) or 
                                           series1.dtype == 'object') and \
                                     float(pd.api.types.is_categorical_dtype(series2) or 
                                           series2.dtype == 'object')
        
        # Value overlap analysis
        set1 = set(series1.unique())
        set2 = set(series2.unique())
        
        if set1 and set2:
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            
            features['value_overlap_jaccard'] = len(intersection) / len(union)
            features['value_overlap_ratio'] = len(intersection) / min(len(set1), len(set2))
            features['unique_value_ratio'] = len(set1) / len(set2) if len(set2) > 0 else 0
        else:
            features['value_overlap_jaccard'] = 0.0
            features['value_overlap_ratio'] = 0.0
            features['unique_value_ratio'] = 0.0
        
        # Statistical correlations (for numeric data)
        if features['both_numeric'] > 0:
            try:
                # Align series for correlation calculation
                common_length = min(len(series1), len(series2))
                s1_sample = series1.sample(n=common_length, random_state=42).reset_index(drop=True)
                s2_sample = series2.sample(n=common_length, random_state=42).reset_index(drop=True)
                
                pearson_corr, pearson_p = pearsonr(s1_sample, s2_sample)
                spearman_corr, spearman_p = spearmanr(s1_sample, s2_sample)
                
                features['pearson_correlation'] = abs(pearson_corr) if not np.isnan(pearson_corr) else 0.0
                features['spearman_correlation'] = abs(spearman_corr) if not np.isnan(spearman_corr) else 0.0
                features['pearson_p_value'] = pearson_p if not np.isnan(pearson_p) else 1.0
                features['spearman_p_value'] = spearman_p if not np.isnan(spearman_p) else 1.0
                
            except Exception as e:
                features['pearson_correlation'] = 0.0
                features['spearman_correlation'] = 0.0
                features['pearson_p_value'] = 1.0
                features['spearman_p_value'] = 1.0
        
        # Mutual information
        try:
            if features['both_numeric'] > 0:
                # For numeric-numeric relationships
                common_length = min(len(series1), len(series2))
                s1_sample = series1.sample(n=common_length, random_state=42).reset_index(drop=True)
                s2_sample = series2.sample(n=common_length, random_state=42).reset_index(drop=True)
                
                mi_score = mutual_info_regression(s1_sample.values.reshape(-1, 1), s2_sample)[0]
                features['mutual_information'] = mi_score
                
            else:
                # For categorical relationships
                le1, le2 = LabelEncoder(), LabelEncoder()
                
                # Sample for performance
                sample_size = min(1000, len(series1), len(series2))
                s1_sample = series1.sample(n=sample_size, random_state=42)
                s2_sample = series2.sample(n=sample_size, random_state=42)
                
                s1_encoded = le1.fit_transform(s1_sample.astype(str))
                s2_encoded = le2.fit_transform(s2_sample.astype(str))
                
                mi_score = mutual_info_classif(s1_encoded.reshape(-1, 1), s2_encoded)[0]
                features['mutual_information'] = mi_score
                
        except Exception as e:
            features['mutual_information'] = 0.0
        
        # Distribution similarity (for numeric data)
        if features['both_numeric'] > 0:
            try:
                from scipy.stats import ks_2samp
                
                # Sample for performance
                sample_size = min(1000, len(series1), len(series2))
                s1_sample = series1.sample(n=sample_size, random_state=42)
                s2_sample = series2.sample(n=sample_size, random_state=42)
                
                ks_statistic, ks_p_value = ks_2samp(s1_sample, s2_sample)
                features['distribution_similarity'] = 1.0 - ks_statistic  # Higher = more similar
                features['ks_p_value'] = ks_p_value
                
            except Exception as e:
                features['distribution_similarity'] = 0.0
                features['ks_p_value'] = 1.0
        
        return features
    
    def _extract_semantic_features(self, series1: pd.Series, series2: pd.Series, 
                                 col1: str, col2: str, table1: str, table2: str) -> Dict[str, float]:
        """Extract semantic features based on names and content"""
        features = {}
        
        # Column name similarity
        features['name_exact_match'] = float(col1.lower() == col2.lower())
        features['name_contains'] = float(col1.lower() in col2.lower() or col2.lower() in col1.lower())
        
        # Common name patterns
        col1_lower, col2_lower = col1.lower(), col2.lower()
        
        # ID patterns
        id_patterns = ['id', 'key', 'code', 'number', 'num']
        features['both_have_id_pattern'] = float(any(pattern in col1_lower for pattern in id_patterns) and 
                                                any(pattern in col2_lower for pattern in id_patterns))
        
        # Foreign key patterns
        features['fk_pattern_match'] = float(table1.lower() in col2_lower or table2.lower() in col1_lower)
        
        # Name similarity (simple string similarity)
        features['name_similarity'] = self._calculate_string_similarity(col1, col2)
        
        # Table name relationship
        features['same_table'] = float(table1 == table2)
        
        return features
    
    def _extract_structural_features(self, series1: pd.Series, series2: pd.Series, 
                                   col1: str, col2: str, table1: str, table2: str) -> Dict[str, float]:
        """Extract structural features"""
        features = {}
        
        # Cardinality analysis
        unique1, unique2 = series1.nunique(), series2.nunique()
        total1, total2 = len(series1), len(series2)
        
        features['cardinality_ratio'] = unique1 / unique2 if unique2 > 0 else 0
        features['size_ratio'] = total1 / total2 if total2 > 0 else 0
        
        # Uniqueness patterns
        features['series1_unique_ratio'] = unique1 / total1 if total1 > 0 else 0
        features['series2_unique_ratio'] = unique2 / total2 if total2 > 0 else 0
        
        # Potential primary key indicators
        features['series1_is_unique'] = float(unique1 == total1)
        features['series2_is_unique'] = float(unique2 == total2)
        
        # One-to-many patterns
        features['one_to_many_pattern'] = float(
            (features['series1_is_unique'] > 0.9 and features['series2_unique_ratio'] < 0.8) or
            (features['series2_is_unique'] > 0.9 and features['series1_unique_ratio'] < 0.8)
        )
        
        return features
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity"""
        # Simple character-based similarity
        str1_lower = str1.lower().replace('_', '').replace('-', '')
        str2_lower = str2.lower().replace('_', '').replace('-', '')
        
        if str1_lower == str2_lower:
            return 1.0
        
        # Jaccard similarity on character level
        set1 = set(str1_lower)
        set2 = set(str2_lower)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _classify_relationship(self, features: Dict[str, float], 
                             series1: pd.Series, series2: pd.Series) -> Tuple[str, float]:
        """Classify relationship type based on extracted features"""
        
        # Rule-based classification with confidence scoring
        
        # Foreign Key Detection
        fk_score = (
            features.get('value_overlap_ratio', 0) * 0.4 +
            features.get('fk_pattern_match', 0) * 0.3 +
            features.get('one_to_many_pattern', 0) * 0.2 +
            features.get('both_have_id_pattern', 0) * 0.1
        )
        
        # Correlation Detection (for numeric columns)
        corr_score = 0
        if features.get('both_numeric', 0) > 0:
            corr_score = max(
                features.get('pearson_correlation', 0),
                features.get('spearman_correlation', 0)
            )
        
        # Semantic Similarity
        semantic_score = (
            features.get('name_exact_match', 0) * 0.5 +
            features.get('name_similarity', 0) * 0.3 +
            features.get('name_contains', 0) * 0.2
        )
        
        # Mutual Information Score
        mi_score = features.get('mutual_information', 0)
        
        # Determine relationship type and confidence
        if fk_score > 0.6:
            return 'FOREIGN_KEY', fk_score
        elif corr_score > 0.7:
            relationship = 'POSITIVELY_CORRELATED' if features.get('pearson_correlation', 0) >= 0 else 'NEGATIVELY_CORRELATED'
            return relationship, corr_score
        elif semantic_score > 0.8:
            return 'SAME_DOMAIN', semantic_score
        elif mi_score > 0.3:
            return 'INFORMATION_DEPENDENCY', mi_score
        elif features.get('value_overlap_ratio', 0) > 0.5:
            return 'SIMILAR_VALUES', features.get('value_overlap_ratio', 0)
        else:
            return 'WEAK_RELATIONSHIP', max(fk_score, corr_score, semantic_score, mi_score)
    
    def _generate_evidence(self, features: Dict[str, float], relationship_type: str) -> Dict[str, Any]:
        """Generate evidence dictionary for the detected relationship"""
        evidence = {
            'detection_method': 'ml_feature_analysis',
            'key_features': {}
        }
        
        # Include top features that support the relationship
        if relationship_type == 'FOREIGN_KEY':
            evidence['key_features'] = {
                'value_overlap_ratio': features.get('value_overlap_ratio', 0),
                'fk_pattern_match': features.get('fk_pattern_match', 0),
                'one_to_many_pattern': features.get('one_to_many_pattern', 0)
            }
        elif 'CORRELATED' in relationship_type:
            evidence['key_features'] = {
                'pearson_correlation': features.get('pearson_correlation', 0),
                'spearman_correlation': features.get('spearman_correlation', 0),
                'mutual_information': features.get('mutual_information', 0)
            }
        elif relationship_type == 'SAME_DOMAIN':
            evidence['key_features'] = {
                'name_similarity': features.get('name_similarity', 0),
                'name_exact_match': features.get('name_exact_match', 0),
                'semantic_similarity': features.get('name_contains', 0)
            }
        
        return evidence