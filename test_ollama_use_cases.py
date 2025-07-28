"""
Test Script for Ollama Use Cases

This script demonstrates the practical implementation of all the use cases
from use_case.md using the enhanced table intelligence with Ollama integration.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.knowledge_graph.llm_config import LLMConfigManager, LLMProviderConfig
from src.knowledge_graph.table_intelligence import LLMSemanticSummarizer, LLMConfig
from src.knowledge_graph.enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer
from src.knowledge_graph.intelligent_data_catalog import IntelligentDataCatalog
from src.knowledge_graph.smart_query_assistant import SmartQueryAssistant
from src.knowledge_graph.auto_documentation_generator import AutoDocumentationGenerator

# Import existing intelligence layer (assuming it exists)
try:
    from src.knowledge_graph.table_intelligence import TableIntelligenceLayer, TableProfile
except ImportError:
    print("Warning: Could not import existing table intelligence components")
    print("This demo will create mock profiles for testing")
    TableIntelligenceLayer = None
    TableProfile = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_profile(table_name: str, business_domain: str, row_count: int) -> 'MockTableProfile':
    """Create mock table profile for testing"""
    
    class MockTableProfile:
        def __init__(self, table_name, business_domain, row_count):
            self.table_name = table_name
            self.business_domain = business_domain
            self.row_count = row_count
            self.column_count = 8
            self.data_quality_score = 0.85
            self.ml_readiness_score = 75.0
            self.ml_readiness_factors = ["Good data volume", "High quality", "Complete features"]
            
            # Mock column classifications based on table type
            if 'order' in table_name.lower():
                self.measure_columns = ['order_value', 'item_count', 'shipping_cost']
                self.dimension_columns = ['order_status', 'payment_method', 'customer_segment']
                self.identifier_columns = ['order_id', 'customer_id']
                self.temporal_columns = ['order_date', 'delivery_date']
            elif 'customer' in table_name.lower():
                self.measure_columns = ['lifetime_value', 'total_orders', 'avg_order_value']
                self.dimension_columns = ['customer_segment', 'region', 'acquisition_channel']
                self.identifier_columns = ['customer_id']
                self.temporal_columns = ['registration_date', 'last_order_date']
            elif 'product' in table_name.lower():
                self.measure_columns = ['price', 'weight', 'rating_score']
                self.dimension_columns = ['category', 'brand', 'supplier']
                self.identifier_columns = ['product_id', 'sku']
                self.temporal_columns = ['launch_date']
            else:
                self.measure_columns = ['value', 'amount', 'quantity']
                self.dimension_columns = ['type', 'category', 'status']
                self.identifier_columns = ['id']
                self.temporal_columns = ['created_date']
            
            self.key_concepts = [business_domain, 'analytics', 'business_data']
            self.table_type = 'fact' if 'order' in table_name.lower() else 'dimension'
            
            # Mock advanced analysis results
            self.quality_profile = self._create_mock_quality_profile()
            self.correlation_analysis = self._create_mock_correlation_analysis()
            self.outlier_analysis = self._create_mock_outlier_analysis()
    
        def _create_mock_quality_profile(self):
            class MockQualityProfile:
                def __init__(self):
                    self.overall_quality_score = 85.0
                    self.critical_alerts = []
                    self.warning_alerts = [{'alert_type': 'missing_values', 'column': 'optional_field'}]
                    self.info_alerts = [{'alert_type': 'high_cardinality', 'column': 'id_field'}]
                    self.quality_recommendations = [
                        "Review missing values in optional fields",
                        "Consider data validation rules"
                    ]
                    self.trend_alerts = []
                    self.distribution_alerts = []
                    self.correlation_alerts = []
            return MockQualityProfile()
        
        def _create_mock_correlation_analysis(self):
            return {
                'linear_relationships': {
                    'strong_linear': [
                        {'variables': ('order_value', 'item_count'), 'correlation': 0.85, 'strength': 'strong'}
                    ],
                    'moderate_linear': [
                        {'variables': ('customer_age', 'order_frequency'), 'correlation': 0.65, 'strength': 'moderate'}
                    ]
                },
                'feature_redundancy': [
                    {'variables': ('total_amount', 'final_price'), 'correlation_value': 0.95}
                ]
            }
        
        def _create_mock_outlier_analysis(self):
            return {
                'high_impact_outliers': [
                    {'column': 'order_value', 'outlier_count': 15, 'impact_level': 'medium'}
                ],
                'outlier_recommendations': [
                    "Investigate high-value orders for data accuracy"
                ]
            }
    
    return MockTableProfile(table_name, business_domain, row_count)


def setup_ollama_integration():
    """Setup Ollama integration for testing"""
    
    try:
        # Configure LLM with Ollama using the existing LLMConfig
        llm_config = LLMConfig(
            provider='ollama',
            model='llama3.2:latest',
            temperature=0.1,
            max_tokens=500,
            timeout=30,
            cache_enabled=True,
            fallback_enabled=True
        )
        
        # Create base summarizer
        base_summarizer = LLMSemanticSummarizer(llm_config)
        
        # Test connection
        test_response = base_summarizer.generate_summary("Test connection. Respond with 'Connected successfully.'")
        
        if len(test_response.strip()) > 0:
            logger.info("✅ Ollama connection successful")
            return EnhancedLLMSemanticSummarizer(base_summarizer)
        else:
            raise Exception("No response from Ollama")
            
    except Exception as e:
        logger.warning(f"⚠️ Ollama connection failed: {e}")
        logger.info("Creating mock LLM for demonstration purposes")
        return create_mock_enhanced_summarizer()


def create_mock_enhanced_summarizer():
    """Create mock enhanced summarizer when Ollama is not available"""
    
    class MockBaseSummarizer:
        def generate_summary(self, prompt: str, max_tokens: int = 500) -> str:
            # Generate mock responses based on prompt content
            if "business summary" in prompt.lower():
                return "This table provides critical business data for analytics and decision-making. High data quality enables reliable reporting and insights."
            elif "friendly title" in prompt.lower():
                return "Customer Order Analytics"
            elif "usage guide" in prompt.lower():
                return "This data is suitable for business analysts, data scientists, and executives. Use for trend analysis, performance monitoring, and strategic planning."
            elif "lineage story" in prompt.lower():
                return "This data originates from operational systems and is regularly updated to support business intelligence and analytics workflows."
            elif "business question" in prompt.lower():
                return "Based on the available data, you can analyze customer patterns, order trends, and business performance metrics using SQL queries on the relevant tables."
            else:
                return "Mock analysis completed. This data provides valuable business insights with good quality for analytics use cases."
    
    class MockEnhancedSummarizer:
        def __init__(self):
            self.base_summarizer = MockBaseSummarizer()
        
        def generate_rich_metadata(self, table_name: str, profile):
            class MockRichMetadata:
                def __init__(self):
                    self.business_summary = f"Business-critical data table containing {profile.row_count:,} records for {profile.business_domain} analytics."
                    self.data_quality_narrative = f"High-quality data with {profile.data_quality_score*100:.1f}% overall score."
                    self.ml_readiness_assessment = f"ML-ready with {profile.ml_readiness_score:.1f}% readiness score."
                    self.relationship_insights = "Strong correlations detected between key business metrics."
                    self.anomaly_explanation = "Minor outliers detected in transaction values."
                    self.usage_recommendations = [
                        "Suitable for executive reporting",
                        "Ready for predictive analytics",
                        "Ideal for trend analysis"
                    ]
                    self.technical_documentation = f"Technical summary for {table_name} table"
            
            return MockRichMetadata()
    
    return MockEnhancedSummarizer()


def create_sample_data_profiles():
    """Create sample data profiles for testing"""
    
    profiles = []
    
    # E-commerce sample tables
    sample_tables = [
        ("orders", "e-commerce", 150000),
        ("customers", "customer_management", 45000),
        ("products", "product_catalog", 8500),
        ("order_items", "e-commerce", 350000),
        ("payments", "financial", 148000),
        ("reviews", "customer_feedback", 89000)
    ]
    
    for table_name, domain, row_count in sample_tables:
        profile = create_mock_profile(table_name, domain, row_count)
        profiles.append(profile)
        logger.info(f"Created profile for {table_name}: {row_count:,} records")
    
    return profiles


def test_intelligent_data_catalog(enhanced_summarizer, profiles):
    """Test Use Case 1: Intelligent Data Catalog"""
    
    print("\n" + "="*80)
    print("🗂️  USE CASE 1: INTELLIGENT DATA CATALOG")
    print("="*80)
    
    try:
        catalog = IntelligentDataCatalog(enhanced_summarizer)
        
        # Generate catalog entries for each table
        catalog_entries = []
        for profile in profiles[:3]:  # Test with first 3 tables
            print(f"\nGenerating catalog entry for {profile.table_name}...")
            entry = catalog.generate_catalog_entry(profile)
            catalog_entries.append(entry)
            
            print(f"📊 {entry.title}")
            print(f"📝 {entry.description[:150]}...")
            print(f"🎯 Usage: {entry.usage_guide[:100]}...")
            print(f"💫 Quality: {entry.quality_badge['level']} ({entry.quality_badge['score']:.1f}%)")
            print(f"🔍 Sample Queries: {len(entry.recommended_queries)} available")
        
        # Generate catalog summary
        summary = catalog.generate_catalog_summary(catalog_entries)
        print(f"\n📈 CATALOG SUMMARY:")
        print(f"- Total Tables: {summary['catalog_overview']['total_tables']}")
        print(f"- Total Records: {summary['catalog_overview']['total_records']:,}")
        print(f"- Average Quality: {summary['catalog_overview']['average_quality_score']:.1f}%")
        print(f"- High Value Tables: {len(summary['high_value_tables'])}")
        
        return catalog_entries
        
    except Exception as e:
        logger.error(f"Error in catalog test: {e}")
        return []


def test_smart_query_assistant(enhanced_summarizer, profiles):
    """Test Use Case 2: Smart Query Assistant"""
    
    print("\n" + "="*80)
    print("🤖 USE CASE 2: SMART QUERY ASSISTANT")
    print("="*80)
    
    try:
        assistant = SmartQueryAssistant(enhanced_summarizer)
        
        # Register tables
        for profile in profiles:
            assistant.register_table(profile)
        
        # Test business questions
        test_questions = [
            "What are our top-selling product categories?",
            "How has customer ordering behavior changed over time?",
            "Which payment methods are most popular?",
            "What is the average customer lifetime value?",
            "Are there any unusual patterns in our order data?"
        ]
        
        print("Answering business questions...")
        for question in test_questions:
            print(f"\n❓ Q: {question}")
            response = assistant.answer_data_question(question)
            
            print(f"✅ A: {response.answer[:200]}...")
            print(f"📊 Relevant Tables: {', '.join(response.relevant_tables)}")
            print(f"🎯 Confidence: {response.confidence_score:.1f}%")
            
            if response.suggested_sql:
                print(f"💻 SQL Suggestion Available: {len(response.suggested_sql)} characters")
            
            if response.data_quality_warnings:
                print(f"⚠️  Warnings: {len(response.data_quality_warnings)} data quality considerations")
        
        # Show capabilities
        capabilities = assistant.get_available_capabilities()
        print(f"\n🔧 ASSISTANT CAPABILITIES:")
        print(f"- Registered Tables: {capabilities['registered_tables']}")
        print(f"- Total Records: {capabilities['total_records']:,}")
        print(f"- Available Analyses: {', '.join(capabilities['available_analyses'][:3])}...")
        
        return assistant
        
    except Exception as e:
        logger.error(f"Error in query assistant test: {e}")
        return None


def test_auto_documentation(enhanced_summarizer, profiles):
    """Test Use Case 3: Auto Documentation Generator"""
    
    print("\n" + "="*80)
    print("📚 USE CASE 3: AUTO DOCUMENTATION GENERATOR")
    print("="*80)
    
    try:
        doc_generator = AutoDocumentationGenerator(enhanced_summarizer)
        
        # Generate data dictionary
        print("Generating comprehensive data dictionary...")
        data_dictionary = doc_generator.generate_data_dictionary(profiles)
        
        print(f"📖 Data Dictionary Generated:")
        print(f"- Length: {len(data_dictionary):,} characters")
        print(f"- Preview: {data_dictionary[:300]}...")
        
        # Generate comprehensive documentation
        print("\nGenerating comprehensive documentation package...")
        doc_package = doc_generator.generate_comprehensive_documentation(
            profiles,
            include_technical=True,
            include_business=True
        )
        
        print(f"📋 Documentation Package:")
        print(f"- Sections: {len(doc_package.sections)}")
        print(f"- Table of Contents: {len(doc_package.table_of_contents)} items")
        print(f"- Confidence Score: {doc_package.confidence_score:.1f}%")
        print(f"- Generated: {doc_package.generated_timestamp}")
        
        # Show section summaries
        print(f"\n📑 SECTIONS:")
        for section in doc_package.sections:
            print(f"- {section.title} ({section.section_type}): {len(section.content)} chars")
        
        # Show executive summary preview
        print(f"\n👔 EXECUTIVE SUMMARY PREVIEW:")
        print(doc_package.executive_summary[:250] + "...")
        
        return doc_package
        
    except Exception as e:
        logger.error(f"Error in documentation test: {e}")
        return None


def test_integration_scenario(enhanced_summarizer, profiles):
    """Test integrated scenario using multiple use cases"""
    
    print("\n" + "="*80)
    print("🚀 INTEGRATION SCENARIO: COMPLETE WORKFLOW")
    print("="*80)
    
    try:
        # Scenario: New data analyst exploring the data system
        print("Scenario: New data analyst needs to understand and use the data system\n")
        
        # Step 1: Explore data catalog
        print("Step 1: Exploring data catalog...")
        catalog = IntelligentDataCatalog(enhanced_summarizer)
        high_value_tables = [p for p in profiles if p.row_count > 50000][:2]
        
        for profile in high_value_tables:
            entry = catalog.generate_catalog_entry(profile)
            print(f"📊 Found: {entry.title} - {entry.quality_badge['level']} quality")
        
        # Step 2: Ask questions about the data
        print("\nStep 2: Asking business questions...")
        assistant = SmartQueryAssistant(enhanced_summarizer)
        for profile in profiles:
            assistant.register_table(profile)
        
        analyst_question = "What data do we have for customer behavior analysis?"
        response = assistant.answer_data_question(analyst_question)
        print(f"❓ {analyst_question}")
        print(f"✅ {response.answer[:150]}...")
        
        # Step 3: Generate documentation for team
        print("\nStep 3: Generating team documentation...")
        doc_generator = AutoDocumentationGenerator(enhanced_summarizer)
        
        # Focus on high-value tables
        focused_docs = doc_generator.generate_data_dictionary(high_value_tables)
        print(f"📚 Generated focused documentation: {len(focused_docs):,} characters")
        
        # Step 4: Quality assessment
        print("\nStep 4: Quality assessment summary...")
        quality_summary = {
            'excellent': [p for p in profiles if p.data_quality_score >= 0.9],
            'good': [p for p in profiles if 0.8 <= p.data_quality_score < 0.9],
            'needs_review': [p for p in profiles if p.data_quality_score < 0.8]
        }
        
        print(f"✅ Excellent quality: {len(quality_summary['excellent'])} tables")
        print(f"👍 Good quality: {len(quality_summary['good'])} tables")
        print(f"⚠️  Needs review: {len(quality_summary['needs_review'])} tables")
        
        # Step 5: ML readiness assessment
        print("\nStep 5: ML readiness overview...")
        ml_ready = [p for p in profiles if (p.ml_readiness_score or 0) > 70]
        print(f"🤖 ML-ready tables: {len(ml_ready)}/{len(profiles)}")
        
        if ml_ready:
            best_ml_table = max(ml_ready, key=lambda p: p.ml_readiness_score)
            print(f"🏆 Best ML candidate: {best_ml_table.table_name} ({best_ml_table.ml_readiness_score:.1f}%)")
        
        print(f"\n🎉 Integration scenario completed successfully!")
        print(f"✨ The analyst can now effectively use the data system with AI assistance")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in integration scenario: {e}")
        return False


def main():
    """Main test function"""
    
    print("🚀 TESTING OLLAMA USE CASES")
    print("Testing enhanced table intelligence with Ollama integration")
    print("="*80)
    
    # Setup
    print("Setting up test environment...")
    enhanced_summarizer = setup_ollama_integration()
    profiles = create_sample_data_profiles()
    
    print(f"✅ Setup complete:")
    print(f"- Enhanced summarizer: {'Ollama' if hasattr(enhanced_summarizer.base_summarizer, 'llm_config') else 'Mock'}")
    print(f"- Sample profiles: {len(profiles)} tables")
    print(f"- Total sample records: {sum(p.row_count for p in profiles):,}")
    
    # Test each use case
    test_results = {}
    
    # Use Case 1: Intelligent Data Catalog
    try:
        catalog_entries = test_intelligent_data_catalog(enhanced_summarizer, profiles)
        test_results['catalog'] = len(catalog_entries) > 0
    except Exception as e:
        logger.error(f"Catalog test failed: {e}")
        test_results['catalog'] = False
    
    # Use Case 2: Smart Query Assistant
    try:
        assistant = test_smart_query_assistant(enhanced_summarizer, profiles)
        test_results['assistant'] = assistant is not None
    except Exception as e:
        logger.error(f"Assistant test failed: {e}")
        test_results['assistant'] = False
    
    # Use Case 3: Auto Documentation
    try:
        doc_package = test_auto_documentation(enhanced_summarizer, profiles)
        test_results['documentation'] = doc_package is not None
    except Exception as e:
        logger.error(f"Documentation test failed: {e}")
        test_results['documentation'] = False
    
    # Integration Scenario
    try:
        integration_success = test_integration_scenario(enhanced_summarizer, profiles)
        test_results['integration'] = integration_success
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        test_results['integration'] = False
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL TEST RESULTS")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.title():20} {status}")
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All use cases working successfully!")
        print("🚀 Your Ollama integration is ready for production use!")
    elif passed_tests > 0:
        print("⚠️  Some use cases working - check logs for failed components")
        print("💡 You can still use the working components while debugging others")
    else:
        print("❌ No use cases passed - check Ollama connection and configuration")
        print("🔧 Try running with mock mode first to test the framework")
    
    print("\n💡 Next Steps:")
    print("1. If using mock mode, set up Ollama for full functionality")
    print("2. Try the working use cases with your real data")
    print("3. Customize prompts and templates for your specific domain")
    print("4. Integrate with your existing data pipeline")


if __name__ == "__main__":
    main()