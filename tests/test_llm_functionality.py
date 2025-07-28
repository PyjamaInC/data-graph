#!/usr/bin/env python3
"""
Comprehensive Tests for LLM-Enhanced Table Intelligence

Tests all the new LLM functionality including:
- LLMSemanticSummarizer
- EnhancedStatisticalProfiler
- PromptTemplateManager
- LLMConfigManager
- Integration with TableIntelligenceLayer and SemanticTableGraphBuilder
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.table_intelligence import (
    TableIntelligenceLayer, LLMConfig, LLMSemanticSummarizer, 
    EnhancedStatisticalProfiler, PromptTemplateManager
)
from src.knowledge_graph.llm_config import LLMConfigManager, LLMProviderConfig


class TestLLMConfig(unittest.TestCase):
    """Test LLM configuration classes"""
    
    def test_llm_provider_config_creation(self):
        """Test LLMProviderConfig creation and conversion"""
        config = LLMProviderConfig(
            provider="ollama",
            model="llama3.2:latest",
            temperature=0.1
        )
        
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "llama3.2:latest")
        self.assertEqual(config.temperature, 0.1)
        
        # Test to_dict method
        config_dict = config.to_dict()
        self.assertIn("provider", config_dict)
        self.assertNotIn("api_key", config_dict)  # Should exclude None values
    
    def test_llm_config_manager_defaults(self):
        """Test LLMConfigManager loads default configurations"""
        manager = LLMConfigManager()
        
        # Should have default providers
        self.assertIn("ollama", manager.configs)
        self.assertIn("huggingface", manager.configs)
        self.assertIn("llamacpp", manager.configs)
        
        # Default active provider should be ollama
        self.assertEqual(manager.active_provider, "ollama")
        
        # Should be able to get active config
        active_config = manager.get_active_config()
        self.assertIsNotNone(active_config)
        self.assertEqual(active_config.provider, "ollama")


class TestPromptTemplateManager(unittest.TestCase):
    """Test prompt template management"""
    
    def setUp(self):
        self.template_manager = PromptTemplateManager()
    
    def test_template_retrieval(self):
        """Test getting templates"""
        system_prompt, user_prompt = self.template_manager.get_prompt(
            'table_summary',
            table_name="test_table",
            row_count=100,
            column_count=5,
            column_analysis="test columns",
            business_context="test context"
        )
        
        self.assertIn("expert data analyst", system_prompt.lower())
        self.assertIn("test_table", user_prompt)
        self.assertIn("100", user_prompt)
    
    def test_custom_template_addition(self):
        """Test adding custom templates"""
        self.template_manager.add_custom_template(
            "test_template",
            "Test system prompt",
            "Test user template with {param}"
        )
        
        system_prompt, user_prompt = self.template_manager.get_prompt(
            'test_template',
            param="test_value"
        )
        
        self.assertEqual(system_prompt, "Test system prompt")
        self.assertIn("test_value", user_prompt)
    
    def test_unknown_template_error(self):
        """Test error handling for unknown templates"""
        with self.assertRaises(ValueError):
            self.template_manager.get_prompt('unknown_template')


class TestEnhancedStatisticalProfiler(unittest.TestCase):
    """Test enhanced statistical profiling"""
    
    def setUp(self):
        self.profiler = EnhancedStatisticalProfiler()
    
    def test_numeric_column_statistics(self):
        """Test statistics for numeric columns"""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # Include outlier
        stats = self.profiler.extract_column_statistics("test_col", data)
        
        # Check basic stats
        self.assertEqual(stats['total_count'], 6)
        self.assertEqual(stats['null_count'], 0)
        self.assertEqual(stats['unique_count'], 6)
        
        # Check numeric stats
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('std', stats)
        self.assertIn('outlier_count', stats)
        
        # Should detect outlier (100)
        self.assertGreater(stats['outlier_count'], 0)
    
    def test_categorical_column_statistics(self):
        """Test statistics for categorical columns"""
        data = pd.Series(['A', 'B', 'A', 'C', 'A', 'B'])
        stats = self.profiler.extract_column_statistics("test_col", data)
        
        # Check categorical stats
        self.assertIn('cardinality', stats)
        self.assertIn('top_values', stats)
        self.assertIn('mode', stats)
        
        self.assertEqual(stats['cardinality'], 3)  # A, B, C
        self.assertEqual(stats['mode'], 'A')  # Most frequent
    
    def test_temporal_column_statistics(self):
        """Test statistics for temporal columns"""
        data = pd.date_range('2023-01-01', periods=5)
        stats = self.profiler.extract_column_statistics("test_col", data)
        
        # Check temporal stats
        self.assertIn('min_date', stats)
        self.assertIn('max_date', stats)
        self.assertIn('date_range_days', stats)
        
        self.assertEqual(stats['date_range_days'], 4)  # 5 days = 4 day range


class TestLLMSemanticSummarizer(unittest.TestCase):
    """Test LLM semantic summarization"""
    
    def setUp(self):
        self.config = LLMConfig(
            provider="ollama",
            model="test_model",
            temperature=0.1
        )
    
    @patch('src.knowledge_graph.table_intelligence.OLLAMA_AVAILABLE', False)
    def test_fallback_when_ollama_unavailable(self):
        """Test fallback behavior when Ollama is not available"""
        summarizer = LLMSemanticSummarizer(self.config)
        self.assertFalse(summarizer.ollama_available)
        
        # Should use fallback summary
        result = summarizer.generate_table_summary(
            "test_table",
            {"row_count": 100, "column_count": 5},
            []
        )
        
        self.assertIn("test_table", result)
        self.assertIn("100 rows", result)
    
    def test_prompt_creation(self):
        """Test prompt creation for table summary"""
        summarizer = LLMSemanticSummarizer(self.config)
        
        # Mock column insights
        mock_insight = Mock()
        mock_insight.column_name = "test_col"
        mock_insight.data_type.value = "INTEGER"
        mock_insight.semantic_role.value = "MEASURE"
        mock_insight.uniqueness_ratio = 1.0
        mock_insight.completeness_ratio = 1.0
        mock_insight.key_patterns = ["high_cardinality"]
        mock_insight.statistical_summary = {"mean": 50.0}
        
        system_prompt, user_prompt = summarizer._create_table_summary_prompt(
            "test_table",
            {"row_count": 100, "column_count": 1, "business_domain": "test"},
            [mock_insight]
        )
        
        self.assertIn("data analyst", system_prompt.lower())
        self.assertIn("test_table", user_prompt)
        self.assertIn("test_col", user_prompt)
        self.assertIn("MEASURE", user_prompt)


class TestTableIntelligenceIntegration(unittest.TestCase):
    """Test integration of LLM features with TableIntelligenceLayer"""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'total_spent': [100.0, 200.0, 150.0],
            'segment': ['Premium', 'Standard', 'Premium']
        })
    
    def test_basic_analysis_without_llm(self):
        """Test basic analysis without LLM enhancement"""
        intelligence = TableIntelligenceLayer(use_llm_summaries=False)
        profile = intelligence.analyze_table("customers", self.sample_data)
        
        self.assertEqual(profile.table_name, "customers")
        self.assertEqual(profile.row_count, 3)
        self.assertEqual(profile.column_count, 4)
        self.assertIsNotNone(profile.semantic_summary)
        self.assertIsInstance(profile.key_concepts, list)
    
    def test_llm_configuration_setup(self):
        """Test LLM configuration setup"""
        llm_config = LLMConfig(provider="test", model="test_model")
        intelligence = TableIntelligenceLayer(
            use_llm_summaries=True,
            llm_config=llm_config
        )
        
        self.assertTrue(intelligence.use_llm_summaries)
        self.assertIsNotNone(intelligence.llm_summarizer)
        self.assertIsNotNone(intelligence.statistical_profiler)
    
    def test_column_insights_with_enhanced_stats(self):
        """Test that enhanced statistics are collected when LLM is enabled"""
        llm_config = LLMConfig(provider="test", model="test_model")
        intelligence = TableIntelligenceLayer(
            use_llm_summaries=True,
            llm_config=llm_config
        )
        
        profile = intelligence.analyze_table("customers", self.sample_data)
        
        # Check that statistical summaries are collected
        for insight in intelligence._analyze_columns(self.sample_data):
            if insight.statistical_summary:
                self.assertIsInstance(insight.statistical_summary, dict)
                self.assertIn('total_count', insight.statistical_summary)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration"""
    
    def test_config_manager_integration(self):
        """Test LLMConfigManager integration with TableIntelligenceLayer"""
        config_manager = LLMConfigManager()
        llm_params = config_manager.get_llm_config_for_table_intelligence()
        
        # Should return valid parameters
        self.assertIn('provider', llm_params)
        self.assertIn('model', llm_params)
        self.assertIn('temperature', llm_params)
        
        # Should be able to create LLMConfig
        llm_config = LLMConfig(**llm_params)
        self.assertIsInstance(llm_config, LLMConfig)
    
    def test_provider_switching(self):
        """Test switching between providers"""
        config_manager = LLMConfigManager()
        
        # Switch to huggingface
        config_manager.set_active_provider("huggingface")
        self.assertEqual(config_manager.active_provider, "huggingface")
        
        # Get config for new provider
        config = config_manager.get_active_config()
        self.assertEqual(config.provider, "huggingface")


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow"""
    
    def setUp(self):
        # Create more realistic sample data
        np.random.seed(42)
        self.customers = pd.DataFrame({
            'customer_id': range(1, 11),
            'customer_name': [f'Customer_{i}' for i in range(1, 11)],
            'registration_date': pd.date_range('2023-01-01', periods=10),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 10),
            'lifetime_value': np.random.uniform(100, 1000, 10),
            'total_orders': np.random.randint(1, 20, 10)
        })
        
        self.orders = pd.DataFrame({
            'order_id': range(1, 21),
            'customer_id': np.random.choice(range(1, 11), 20),
            'order_date': pd.date_range('2023-01-01', periods=20, freq='2D'),
            'total_amount': np.random.uniform(10, 500, 20),
            'order_status': np.random.choice(['Completed', 'Pending'], 20)
        })
    
    def test_complete_workflow_without_llm(self):
        """Test complete workflow without LLM"""
        intelligence = TableIntelligenceLayer(use_llm_summaries=False)
        
        # Analyze both tables
        customer_profile = intelligence.analyze_table("customers", self.customers)
        order_profile = intelligence.analyze_table("orders", self.orders)
        
        # Verify profiles
        self.assertEqual(customer_profile.table_name, "customers")
        self.assertEqual(order_profile.table_name, "orders")
        
        # Should classify business domains
        self.assertIsNotNone(customer_profile.business_domain)
        self.assertIsNotNone(order_profile.business_domain)
        
        # Should identify table types
        self.assertIn(customer_profile.table_type, ['fact', 'dimension', 'bridge', 'temporal'])
        self.assertIn(order_profile.table_type, ['fact', 'dimension', 'bridge', 'temporal'])
    
    @patch('src.knowledge_graph.table_intelligence.OLLAMA_AVAILABLE', False)
    def test_graceful_degradation(self):
        """Test that system gracefully degrades when LLM is not available"""
        llm_config = LLMConfig(provider="ollama", model="test_model")
        intelligence = TableIntelligenceLayer(
            use_llm_summaries=True,
            llm_config=llm_config
        )
        
        # Should still work, just use fallback summaries
        profile = intelligence.analyze_table("customers", self.customers)
        
        self.assertIsNotNone(profile.semantic_summary)
        self.assertEqual(profile.table_name, "customers")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)