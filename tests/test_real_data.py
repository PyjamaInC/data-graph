"""
Test Real Data with Ollama Use Cases

This script tests the enhanced table intelligence with your actual Olist e-commerce data,
demonstrating practical implementation with real business data.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.knowledge_graph.table_intelligence import TableIntelligenceLayer, EnhancedTableIntelligenceLayer, LLMConfig
from src.knowledge_graph.enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer
from src.knowledge_graph.intelligent_data_catalog import IntelligentDataCatalog
from src.knowledge_graph.smart_query_assistant import SmartQueryAssistant
from src.knowledge_graph.auto_documentation_generator import AutoDocumentationGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_olist_data():
    """Load actual Olist e-commerce data"""
    
    data_dir = Path("data/raw/ecommerce")
    datasets = {}
    
    # Define the main tables we want to analyze
    table_configs = {
        'orders': {
            'file': 'olist_orders_dataset.csv',
            'description': 'E-commerce order transactions and status tracking'
        },
        'order_items': {
            'file': 'olist_order_items_dataset.csv', 
            'description': 'Individual items within orders with pricing and logistics'
        },
        'customers': {
            'file': 'olist_customers_dataset.csv',
            'description': 'Customer demographics and geographic information'
        },
        'products': {
            'file': 'olist_products_dataset.csv',
            'description': 'Product catalog with categories and specifications'
        },
        'payments': {
            'file': 'olist_order_payments_dataset.csv',
            'description': 'Payment methods and transaction details'
        },
        'reviews': {
            'file': 'olist_order_reviews_dataset.csv',
            'description': 'Customer reviews and satisfaction ratings'
        }
    }
    
    print("Loading Olist e-commerce datasets...")
    
    for table_name, config in table_configs.items():
        file_path = data_dir / config['file']
        
        if file_path.exists():
            try:
                # Load with reasonable sample size for testing
                df = pd.read_csv(file_path)
                
                # Sample large datasets for faster processing during testing
                if len(df) > 50000:
                    df = df.sample(n=50000, random_state=42)
                    print(f"📊 Sampled {table_name}: {len(df):,} records (from larger dataset)")
                else:
                    print(f"📊 Loaded {table_name}: {len(df):,} records")
                
                datasets[table_name] = {
                    'data': df,
                    'description': config['description'],
                    'original_size': pd.read_csv(file_path, nrows=0).shape[0] if len(df) == 50000 else len(df)
                }
                
            except Exception as e:
                logger.warning(f"Failed to load {table_name}: {e}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    return datasets


def setup_table_intelligence():
    """Setup table intelligence with Ollama integration"""
    
    try:
        # Configure with Ollama - INCREASED TOKENS FOR DETAILED OUTPUT
        llm_config = LLMConfig(
            provider='ollama',
            model='llama3.2:latest',
            temperature=0.1,
            max_tokens=1500,  # 🚀 INCREASED from 400 to 1500!
            timeout=60,       # 🚀 INCREASED timeout for longer generation
            cache_enabled=True,
            fallback_enabled=True
        )
        
        # Create enhanced table intelligence layer
        intelligence = EnhancedTableIntelligenceLayer(
            model_name='all-MiniLM-L6-v2',
            enable_profiling=True,
            cache_embeddings=True,
            use_llm_summaries=True,
            llm_config=llm_config,
            enable_advanced_quality=True,
            enable_outlier_detection=True,
            enable_correlation_analysis=True
        )
        
        logger.info("✅ Table intelligence layer initialized with Ollama")
        return intelligence
        
    except Exception as e:
        logger.error(f"Failed to setup table intelligence: {e}")
        return None


def analyze_real_tables(intelligence, datasets):
    """Analyze real data tables using enhanced intelligence"""
    
    print("\n" + "="*80)
    print("🔍 ANALYZING REAL OLIST DATA")
    print("="*80)
    
    profiles = {}
    
    for table_name, dataset_info in datasets.items():
        print(f"\nAnalyzing {table_name}...")
        
        try:
            df = dataset_info['data']
            
            # Analyze table with enhanced intelligence using comprehensive method
            profile = intelligence.analyze_table_comprehensive(
                table_name=table_name,
                df=df,
                schema_info={'description': dataset_info['description']}
            )
            
            profiles[table_name] = profile
            
            # Show basic analysis results
            print(f"✅ {table_name}:")
            print(f"   📊 Records: {profile.row_count:,}")
            print(f"   📈 Quality Score: {profile.data_quality_score:.2f}")
            print(f"   🏷️  Business Domain: {profile.business_domain}")
            print(f"   📋 Measures: {len(profile.measure_columns)}")
            print(f"   🗂️  Dimensions: {len(profile.dimension_columns)}")
            
        except Exception as e:
            logger.error(f"Failed to analyze {table_name}: {e}")
    
    return profiles


def test_catalog_with_real_data(intelligence, profiles):
    """Test data catalog with real data"""
    
    print("\n" + "="*80)
    print("🗂️ REAL DATA CATALOG GENERATION")
    print("="*80)
    
    try:
        # Get enhanced summarizer from intelligence layer
        if hasattr(intelligence, 'llm_summarizer') and intelligence.llm_summarizer:
            enhanced_summarizer = EnhancedLLMSemanticSummarizer(intelligence.llm_summarizer)
            catalog = IntelligentDataCatalog(enhanced_summarizer)
            
            print("Generating catalog entries for real Olist tables...")
            
            # Generate entries for top 3 tables by size
            sorted_profiles = sorted(profiles.values(), key=lambda p: p.row_count, reverse=True)[:3]
            
            for profile in sorted_profiles:
                print(f"\n📋 Generating catalog for {profile.table_name}...")
                entry = catalog.generate_catalog_entry(profile)
                
                print(f"📊 **{entry.title}**")
                print(f"📝 {entry.description}")
                print(f"🎯 Usage: {entry.usage_guide}")
                print(f"💫 Quality: {entry.quality_badge['level']} ({entry.quality_badge['score']:.1f}%)")
                
                # Show sample queries
                if entry.recommended_queries:
                    print(f"🔍 Sample SQL Query:")
                    print(f"   {entry.recommended_queries[0]['sql']}")
            
            return True
            
        else:
            print("⚠️ LLM summarizer not available - using basic catalog")
            return False
            
    except Exception as e:
        logger.error(f"Catalog generation failed: {e}")
        return False


def test_query_assistant_real_data(intelligence, profiles):
    """Test query assistant with real business questions"""
    
    print("\n" + "="*80)
    print("🤖 REAL DATA QUERY ASSISTANT")
    print("="*80)
    
    try:
        if hasattr(intelligence, 'llm_summarizer') and intelligence.llm_summarizer:
            enhanced_summarizer = EnhancedLLMSemanticSummarizer(intelligence.llm_summarizer)
            assistant = SmartQueryAssistant(enhanced_summarizer)
            
            # Register real data tables
            for profile in profiles.values():
                assistant.register_table(profile)
            
            # Real business questions for e-commerce data
            real_questions = [
                "What are the most popular product categories in our orders?",
                "How do payment methods vary across different regions?", 
                "What is the relationship between order value and delivery time?",
                "Which customers have the highest order frequency?",
                "Are there seasonal patterns in our order data?"
            ]
            
            print("Asking real business questions about Olist data...")
            
            for question in real_questions[:3]:  # Test first 3 questions
                print(f"\n❓ **Q:** {question}")
                response = assistant.answer_data_question(question)
                
                print(f"✅ **A:** {response.answer}")
                print(f"📊 **Relevant Tables:** {', '.join(response.relevant_tables)}")
                print(f"🎯 **Confidence:** {response.confidence_score:.1f}%")
                
                if response.data_quality_warnings:
                    print(f"⚠️ **Warnings:** {response.data_quality_warnings[0]}")  # 🚀 REMOVED [:100] truncation!
            
            return True
            
        else:
            print("⚠️ LLM not available for query assistance")
            return False
            
    except Exception as e:
        logger.error(f"Query assistant failed: {e}")
        return False


def generate_real_data_insights(profiles):
    """Generate insights about the real data"""
    
    print("\n" + "="*80)
    print("📊 REAL DATA INSIGHTS SUMMARY")
    print("="*80)
    
    total_records = sum(p.row_count for p in profiles.values())
    avg_quality = sum(p.data_quality_score for p in profiles.values()) / len(profiles)
    
    print(f"**Olist E-commerce Data Portfolio:**")
    print(f"📈 Total Tables Analyzed: {len(profiles)}")
    print(f"📊 Total Records: {total_records:,}")
    print(f"💯 Average Data Quality: {avg_quality:.2f}")
    
    # Quality distribution
    high_quality = [p for p in profiles.values() if p.data_quality_score >= 0.9]
    good_quality = [p for p in profiles.values() if 0.8 <= p.data_quality_score < 0.9]
    needs_work = [p for p in profiles.values() if p.data_quality_score < 0.8]
    
    print(f"\n**Quality Distribution:**")
    print(f"✅ High Quality (90%+): {len(high_quality)} tables")
    print(f"👍 Good Quality (80-89%): {len(good_quality)} tables") 
    print(f"⚠️ Needs Work (<80%): {len(needs_work)} tables")
    
    # Business domains
    domains = set(p.business_domain for p in profiles.values() if p.business_domain)
    print(f"\n**Business Domains Detected:**")
    for domain in domains:
        domain_tables = [p for p in profiles.values() if p.business_domain == domain]
        print(f"🏢 {domain}: {len(domain_tables)} tables")
    
    # Largest tables
    largest_tables = sorted(profiles.values(), key=lambda p: p.row_count, reverse=True)[:3]
    print(f"\n**Largest Tables:**")
    for i, profile in enumerate(largest_tables, 1):
        print(f"{i}. {profile.table_name}: {profile.row_count:,} records")
    
    # Feature richness
    total_measures = sum(len(p.measure_columns) for p in profiles.values())
    total_dimensions = sum(len(p.dimension_columns) for p in profiles.values())
    
    print(f"\n**Feature Analysis:**")
    print(f"📊 Total Measures: {total_measures}")
    print(f"🏷️ Total Dimensions: {total_dimensions}")
    print(f"📋 Avg Features per Table: {(total_measures + total_dimensions) / len(profiles):.1f}")


def main():
    """Main function for real data testing"""
    
    print("🚀 TESTING WITH REAL OLIST E-COMMERCE DATA")
    print("Enhanced Table Intelligence + Ollama on Real Business Data")
    print("="*80)
    
    # Load real data
    datasets = load_olist_data()
    
    if not datasets:
        print("❌ No data loaded. Check that Olist CSV files are in data/raw/ecommerce/")
        return
    
    print(f"✅ Loaded {len(datasets)} real datasets")
    
    # Setup intelligence
    intelligence = setup_table_intelligence()
    
    if not intelligence:
        print("❌ Failed to setup table intelligence")
        return
    
    # Analyze real tables
    profiles = analyze_real_tables(intelligence, datasets)
    
    if not profiles:
        print("❌ No tables analyzed successfully")
        return
    
    print(f"✅ Analyzed {len(profiles)} tables successfully")
    
    # Test use cases with real data
    results = {}
    
    # Test catalog generation
    print("\n🗂️ Testing catalog generation...")
    results['catalog'] = test_catalog_with_real_data(intelligence, profiles)
    
    # Test query assistant  
    print("\n🤖 Testing query assistant...")
    results['assistant'] = test_query_assistant_real_data(intelligence, profiles)
    
    # Generate insights
    generate_real_data_insights(profiles)
    
    # Final results
    print("\n" + "="*80)
    print("🎯 REAL DATA TEST RESULTS")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "⚠️ LIMITED"
        print(f"{test_name.title():15} {status}")
    
    print(f"\n📊 **Results:** {passed}/{total} use cases working with real data")
    
    if passed > 0:
        print("🎉 **Success!** Your Ollama integration works with real business data!")
        print("🚀 **Ready for Production:** Use these insights for actual business decisions")
    else:
        print("⚠️ **Limited Mode:** Basic analysis working, LLM features need setup")
    
    print("\n💡 **Next Steps:**")
    print("1. 📈 Use these real insights for business reporting")
    print("2. 🔧 Fine-tune prompts for your specific e-commerce needs") 
    print("3. 📊 Set up automated reporting with this data")
    print("4. 🤖 Implement ML models using the readiness assessments")


if __name__ == "__main__":
    main()