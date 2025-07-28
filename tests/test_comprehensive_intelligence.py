#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Data Exploration Agent

This test uses ALL real e-commerce data and creates complex queries that force the agent
to navigate across multiple tables, leveraging semantic relationships and catalog insights.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from knowledge_graph.table_intelligence import EnhancedTableIntelligenceLayer, LLMSemanticSummarizer, LLMConfig
from knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
from knowledge_graph.enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer
from knowledge_graph.intelligent_data_catalog import IntelligentDataCatalog
from agents.react_agents.enhanced_data_exploration_agent import EnhancedDataExplorationReActAgent

def load_full_ecommerce_data():
    """Load all e-commerce datasets"""
    
    data_path = Path(__file__).parent / "data" / "raw" / "ecommerce"
    
    datasets = {}
    
    # Load all CSV files
    csv_files = {
        'customers': 'olist_customers_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv', 
        'order_items': 'olist_order_items_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'product_categories': 'product_category_name_translation.csv'
    }
    
    print("📂 Loading comprehensive e-commerce dataset...")
    
    for table_name, filename in csv_files.items():
        file_path = data_path / filename
        try:
            df = pd.read_csv(file_path)
            datasets[table_name] = df
            print(f"  ✅ {table_name}: {df.shape}")
        except Exception as e:
            print(f"  ❌ Failed to load {table_name}: {e}")
    
    print(f"\n📊 Total dataset size: {sum(df.shape[0] for df in datasets.values())} rows across {len(datasets)} tables")
    
    return datasets

def test_comprehensive_intelligence():
    """Test comprehensive intelligence with cross-table analysis"""
    
    print("🚀 COMPREHENSIVE INTELLIGENCE TEST")
    print("=" * 80)
    
    # Load full dataset
    tables = load_full_ecommerce_data()
    
    if not tables:
        print("❌ No data loaded. Exiting test.")
        return
    
    # Initialize components
    print("\n🧠 Initializing Full Intelligence Stack...")
    enhanced_intelligence = EnhancedTableIntelligenceLayer()
    semantic_graph_builder = EnhancedKnowledgeGraphBuilder()
    
    # Initialize LLM components with Ollama (as configured in llm_config.py)
    try:
        # Use Ollama configuration for knowledge graph components
        llm_config = LLMConfig(
            provider="ollama",
            model="llama3.2:latest",
            max_tokens=700,
            temperature=0.1
        )
        
        base_summarizer = LLMSemanticSummarizer(llm_config)
        enhanced_summarizer = EnhancedLLMSemanticSummarizer(base_summarizer)
        intelligent_catalog = IntelligentDataCatalog(enhanced_summarizer)
        print("✅ Full LLM-powered intelligence stack initialized with Ollama")
    except Exception as e:
        print(f"⚠️ Warning: LLM components failed: {e}")
        enhanced_summarizer = None
        intelligent_catalog = None
    
    # Initialize comprehensive agent
    agent = EnhancedDataExplorationReActAgent(
        enhanced_intelligence=enhanced_intelligence,
        semantic_graph_builder=semantic_graph_builder,
        enhanced_summarizer=enhanced_summarizer,
        intelligent_catalog=intelligent_catalog,
        llm_model="gpt-4o-mini-2024-07-18"
    )
    
    # Start with simpler scenarios to test core functionality
    complex_scenarios = [
        {
            "name": "Cross-Table Revenue Analysis",
            "question": "What is the relationship between customer geographic location, order payment methods, and total revenue? Which regions and payment methods drive the highest revenue?",
            "complexity": "High - requires joining customers, orders, payments, and geolocation"
        }
    ]
    
    results = {}
    
    # Run complex scenarios
    for i, scenario in enumerate(complex_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"🔍 COMPLEX TEST {i}: {scenario['name']}")
        print(f"❓ Question: {scenario['question']}")
        print(f"🧩 Complexity: {scenario['complexity']}")
        print(f"{'='*80}")
        
        try:
            # Run comprehensive analysis
            result = agent.explore_for_insights(scenario['question'], tables)
            
            # Store results
            results[scenario['name']] = result
            
            # Display comprehensive results
            print(f"\n📊 COMPREHENSIVE EXPLORATION SUMMARY:")
            summary = result['exploration_summary']
            print(f"  Iterations: {summary['iterations_used']}")
            print(f"  Confidence: {summary['confidence_level']:.1%}")
            print(f"  Strategy: {summary['strategy_used']}")
            print(f"  Operations: {summary['operations_executed']}")
            print(f"  Total Findings: {summary['total_findings']}")
            
            print(f"\n💡 INTELLIGENCE-DRIVEN INSIGHTS:")
            insights = result['insights']
            print(f"  Direct Answer: {insights['direct_answer']}")
            
            print(f"\n🔑 KEY INSIGHTS:")
            for j, insight in enumerate(insights['key_insights'][:5], 1):
                print(f"  {j}. {insight}")
            
            print(f"\n📈 SUPPORTING EVIDENCE:")
            for j, evidence in enumerate(insights.get('supporting_evidence', [])[:3], 1):
                print(f"  {j}. {evidence}")
            
            print(f"\n💼 BUSINESS IMPLICATIONS:")
            for j, implication in enumerate(insights.get('business_implications', [])[:3], 1):
                print(f"  {j}. {implication}")
            
            print(f"\n🎯 INTELLIGENCE CONTEXT:")
            intelligence_context = result['intelligence_context']
            print(f"  Tables Profiled: {intelligence_context['profiles_generated']}")
            print(f"  Relationships Used: {len(intelligence_context.get('strategy_applied', {}).get('key_operations', []))}")
            
            # Show catalog insights if available
            if 'catalog_insights' in intelligence_context and intelligence_context['catalog_insights']:
                print(f"\n📚 CATALOG INTELLIGENCE:")
                for table_name, catalog_info in intelligence_context['catalog_insights'].items():
                    print(f"  {table_name}: {catalog_info['title']}")
                    quality = catalog_info['quality_badge']
                    print(f"    Quality: {quality['icon']} {quality['level']} ({quality['score']:.1f}%)")
                    if catalog_info.get('recommended_queries'):
                        print(f"    Recommended: {catalog_info['recommended_queries'][0].get('title', 'Analysis')}")
            
            print(f"\n✅ Confidence Score: {insights.get('confidence_score', 0):.1%}")
            
        except Exception as e:
            print(f"❌ ERROR in {scenario['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[scenario['name']] = {"error": str(e)}
    
    # Comprehensive assessment
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE INTELLIGENCE ASSESSMENT")
    print(f"{'='*80}")
    
    successful_tests = len([r for r in results.values() if 'error' not in r])
    total_tests = len(complex_scenarios)
    
    print(f"✅ Successful Complex Tests: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        # Assess intelligence capabilities
        cross_table_analysis = 0
        relationship_usage = 0
        catalog_integration = 0
        high_confidence = 0
        
        for name, result in results.items():
            if 'error' not in result:
                # Check cross-table analysis
                intelligence_context = result.get('intelligence_context', {})
                if intelligence_context.get('profiles_generated', 0) >= 3:
                    cross_table_analysis += 1
                
                # Check relationship usage
                if 'strategy_applied' in intelligence_context:
                    relationship_usage += 1
                
                # Check catalog integration
                if 'catalog_insights' in intelligence_context and intelligence_context['catalog_insights']:
                    catalog_integration += 1
                
                # Check confidence
                insights = result.get('insights', {})
                if insights.get('confidence_score', 0) >= 0.7:
                    high_confidence += 1
        
        print(f"🔗 Cross-Table Analysis: {cross_table_analysis}/{successful_tests}")
        print(f"🕸️ Relationship Intelligence: {relationship_usage}/{successful_tests}")
        print(f"📚 Catalog Integration: {catalog_integration}/{successful_tests}")
        print(f"💪 High Confidence Results: {high_confidence}/{successful_tests}")
        
        # Overall intelligence score
        intelligence_score = (cross_table_analysis + relationship_usage + catalog_integration + high_confidence) / (successful_tests * 4)
        
        if intelligence_score >= 0.8:
            grade = "🟢 OUTSTANDING - Full intelligence stack working effectively"
        elif intelligence_score >= 0.6:
            grade = "🟡 GOOD - Most intelligence features functioning well"
        else:
            grade = "🔴 NEEDS IMPROVEMENT - Intelligence integration incomplete"
        
        print(f"\n🎯 Overall Intelligence Assessment: {grade}")
        print(f"📊 Intelligence Score: {intelligence_score:.1%}")
        
        # Summary of what was tested
        print(f"\n📋 COMPREHENSIVE TEST SUMMARY:")
        print(f"  📊 Total Data Points: {sum(df.shape[0] for df in tables.values())}")
        print(f"  🗃️ Tables Analyzed: {len(tables)}")
        print(f"  🔍 Complex Scenarios: {len(complex_scenarios)}")
        print(f"  🧠 Intelligence Features: Semantic relationships, LLM catalog, cross-table analysis")
        print(f"  🤖 AI Model: GPT-4o-mini with enhanced context")
    
    return results

if __name__ == "__main__":
    # Set environment for testing
    os.environ.setdefault('OPENAI_API_KEY', 'your-key-here')
    
    # Suppress warnings and progress bars
    os.environ['YDATA_PROFILING_DISABLE_PROGRESS_BAR'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        results = test_comprehensive_intelligence()
        print("\n🎉 Comprehensive intelligence testing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()