#!/usr/bin/env python3
"""
Test for the revised Enhanced Data Exploration ReAct Agent

This test uses the revised agent with proper question-specific analysis.
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

def create_sample_data():
    """Create realistic sample e-commerce data for testing"""
    np.random.seed(42)
    
    # Create sample orders data
    n_orders = 1000
    orders = pd.DataFrame({
        'order_id': [f'order_{i:05d}' for i in range(n_orders)],
        'customer_id': [f'customer_{i:04d}' for i in np.random.randint(0, 500, n_orders)],
        'order_status': np.random.choice(['delivered', 'shipped', 'processing', 'canceled'], n_orders, p=[0.7, 0.15, 0.1, 0.05]),
        'order_purchase_timestamp': pd.date_range('2022-01-01', periods=n_orders, freq='2H'),
        'order_delivered_customer_date': pd.date_range('2022-01-01', periods=n_orders, freq='2H') + pd.Timedelta(days=7),
        'order_estimated_delivery_date': pd.date_range('2022-01-01', periods=n_orders, freq='2H') + pd.Timedelta(days=10)
    })
    
    # Create sample order items
    n_items = 2500
    order_items = pd.DataFrame({
        'order_id': np.random.choice(orders['order_id'], n_items),
        'product_id': [f'product_{i:04d}' for i in np.random.randint(0, 200, n_items)],
        'seller_id': [f'seller_{i:03d}' for i in np.random.randint(0, 50, n_items)],
        'shipping_limit_date': pd.date_range('2022-01-01', periods=n_items, freq='3H'),
        'price': np.random.lognormal(3, 1, n_items).round(2),
        'freight_value': np.random.exponential(10, n_items).round(2)
    })
    
    # Create sample customers
    n_customers = 500
    customers = pd.DataFrame({
        'customer_id': [f'customer_{i:04d}' for i in range(n_customers)],
        'customer_unique_id': [f'unique_{i:04d}' for i in range(n_customers)],
        'customer_zip_code_prefix': np.random.randint(10000, 99999, n_customers),
        'customer_city': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília', 'Salvador'], n_customers),
        'customer_state': np.random.choice(['SP', 'RJ', 'MG', 'DF', 'BA'], n_customers)
    })
    
    # Create sample products 
    n_products = 200
    products = pd.DataFrame({
        'product_id': [f'product_{i:04d}' for i in range(n_products)],
        'product_category_name': np.random.choice([
            'health_beauty', 'computers_accessories', 'auto', 'furniture_decor', 
            'watches_gifts', 'sports_leisure', 'baby', 'fashion_male_clothing'
        ], n_products),
        'product_name_length': np.random.randint(20, 100, n_products),
        'product_description_length': np.random.randint(100, 2000, n_products),
        'product_photos_qty': np.random.randint(1, 10, n_products),
        'product_weight_g': np.random.lognormal(5, 1, n_products).round(0),
        'product_length_cm': np.random.lognormal(3, 0.5, n_products).round(1),
        'product_height_cm': np.random.lognormal(2.5, 0.5, n_products).round(1),
        'product_width_cm': np.random.lognormal(3, 0.5, n_products).round(1)
    })
    
    # Add some missing values and outliers for quality testing
    orders.loc[np.random.choice(orders.index, 50, replace=False), 'order_delivered_customer_date'] = pd.NaT
    order_items.loc[np.random.choice(order_items.index, 30, replace=False), 'freight_value'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(order_items.index, 20, replace=False)
    order_items.loc[outlier_indices, 'price'] = order_items.loc[outlier_indices, 'price'] * 10
    
    tables = {
        'orders': orders,
        'order_items': order_items,
        'customers': customers,
        'products': products
    }
    
    print("📊 Created sample e-commerce data:")
    for name, df in tables.items():
        print(f"  {name}: {df.shape}")
    
    return tables

def test_revised_enhanced_agent():
    """Test the revised enhanced agent with specific questions"""
    
    print("🚀 REVISED ENHANCED AGENT TEST")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading E-commerce Data...")
    tables = create_sample_data()
    
    # Initialize components
    print("\n🧠 Initializing Intelligence Components...")
    enhanced_intelligence = EnhancedTableIntelligenceLayer()
    semantic_graph_builder = EnhancedKnowledgeGraphBuilder()
    
    # Initialize LLM components with Ollama (as configured in llm_config.py)
    try:
        # Use Ollama configuration for knowledge graph components
        llm_config = LLMConfig(
            provider="ollama",
            model="llama3.2:latest",
            max_tokens=1000,
            temperature=0.1
        )
        
        base_summarizer = LLMSemanticSummarizer(llm_config)
        enhanced_summarizer = EnhancedLLMSemanticSummarizer(base_summarizer)
        intelligent_catalog = IntelligentDataCatalog(enhanced_summarizer)
        print("✅ LLM components initialized with Ollama")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize LLM components: {e}")
        enhanced_summarizer = None
        intelligent_catalog = None
    
    # Initialize revised agent
    agent = EnhancedDataExplorationReActAgent(
        enhanced_intelligence=enhanced_intelligence,
        semantic_graph_builder=semantic_graph_builder,
        enhanced_summarizer=enhanced_summarizer,
        intelligent_catalog=intelligent_catalog,
        llm_model="gpt-4o-mini-2024-07-18"
    )
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Quality Assessment Analysis",
            "question": "What are the main data quality issues in this e-commerce dataset?"
        },
        {
            "name": "Correlation Investigation", 
            "question": "What relationships exist between order values, shipping costs, and delivery times?"
        },
        {
            "name": "Outlier Detection",
            "question": "Are there any unusual patterns or outliers in the pricing data?"
        }
    ]
    
    results = {}
    
    # Run each test scenario
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🔍 TEST {i}: {scenario['name']}")
        print(f"❓ Question: {scenario['question']}")
        print(f"{'='*60}")
        
        try:
            # Run the revised agent
            result = agent.explore_for_insights(scenario['question'], tables)
            
            # Store results
            results[scenario['name']] = result
            
            # Display key results
            print(f"\n📊 EXPLORATION SUMMARY:")
            summary = result['exploration_summary']
            print(f"  Iterations: {summary['iterations_used']}")
            print(f"  Confidence: {summary['confidence_level']:.1%}")
            print(f"  Strategy: {summary['strategy_used']}")
            print(f"  Intelligence-Driven: {summary['intelligence_driven']}")
            
            print(f"\n💡 INSIGHTS:")
            insights = result['insights']
            if 'direct_answer' in insights:
                print(f"  Direct Answer: {insights['direct_answer']}")
            if 'key_insights' in insights:
                for j, insight in enumerate(insights['key_insights'][:3], 1):
                    print(f"  {j}. {insight}")
            if 'confidence_score' in insights:
                print(f"  Confidence: {insights['confidence_score']:.1%}")
            
            print(f"\n🎯 QUESTION-SPECIFIC ANALYSIS:")
            intelligence_context = result['intelligence_context']
            if 'question_analysis' in intelligence_context:
                question_analysis = intelligence_context['question_analysis']
                print(f"  Detected Intent: {question_analysis['primary_intent']['intent']}")
                print(f"  Strategy: {question_analysis['analysis_strategy']['approach']}")
                print(f"  Priority Tables: {question_analysis['analysis_strategy']['priority_tables']}")
            elif 'limitation_detected' in intelligence_context:
                print(f"  Limitation Detected: {intelligence_context.get('limitations', 'Unknown')}")
                print(f"  Analysis Status: Limited due to data constraints")
            else:
                print(f"  Analysis Context: Available but structure varies")
            
            # Display catalog insights if available
            if 'catalog_insights' in intelligence_context and intelligence_context['catalog_insights']:
                print(f"\n📚 CATALOG INSIGHTS:")
                for table_name, catalog_info in intelligence_context['catalog_insights'].items():
                    print(f"  {table_name}: {catalog_info['title']}")
                    quality = catalog_info['quality_badge']
                    print(f"    Quality: {quality['icon']} {quality['level']} ({quality['score']:.1f}%)")
            
        except Exception as e:
            print(f"❌ ERROR in {scenario['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[scenario['name']] = {"error": str(e)}
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("🎯 REVISED AGENT ASSESSMENT")
    print(f"{'='*80}")
    
    successful_tests = len([r for r in results.values() if 'error' not in r])
    total_tests = len(test_scenarios)
    
    print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        # Check if we're getting question-specific responses
        specific_responses = 0
        multi_iteration = 0
        
        for name, result in results.items():
            if 'error' not in result:
                # Check if response is question-specific (not generic)
                insights = result.get('insights', {})
                direct_answer = insights.get('direct_answer', '')
                if 'quality' in name.lower() and any(word in direct_answer.lower() for word in ['missing', 'null', 'complete', 'quality']):
                    specific_responses += 1
                elif 'correlation' in name.lower() and any(word in direct_answer.lower() for word in ['relationship', 'correlation', 'price', 'shipping']):
                    specific_responses += 1
                elif 'outlier' in name.lower() and any(word in direct_answer.lower() for word in ['outlier', 'unusual', 'pricing', 'anomal']):
                    specific_responses += 1
                
                # Check iterations
                iterations = result.get('exploration_summary', {}).get('iterations_used', 0)
                if iterations > 1:
                    multi_iteration += 1
        
        print(f"🎯 Question-Specific Responses: {specific_responses}/{successful_tests}")
        print(f"🔄 Multi-Iteration Exploration: {multi_iteration}/{successful_tests}")
        
        # Overall performance
        if specific_responses >= successful_tests * 0.8 and multi_iteration >= successful_tests * 0.5:
            grade = "🟢 EXCELLENT - Agent provides specific answers with proper exploration"
        elif specific_responses >= successful_tests * 0.6:
            grade = "🟡 GOOD - Some question-specific responses"
        else:
            grade = "🔴 NEEDS IMPROVEMENT - Still generic responses"
        
        print(f"🎯 Overall Assessment: {grade}")
    
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
        results = test_revised_enhanced_agent()
        print("\n🎉 Testing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()