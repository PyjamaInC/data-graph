#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Intelligence-Driven ReAct Agent

This test demonstrates the full capabilities of the enhanced agent:
- Comprehensive knowledge graph intelligence
- Quality-aware operations
- Correlation-driven analysis
- Outlier detection and investigation
- Business context understanding
- Intelligence-driven operation generation

Test data: Real Olist e-commerce datasets
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from knowledge_graph.table_intelligence import EnhancedTableIntelligenceLayer
from knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
from knowledge_graph.enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer
from agents.react_agents.comprehensive_enhanced_agent import ComprehensiveEnhancedAgent

def load_sample_ecommerce_data():
    """Load sample Olist e-commerce data for testing"""
    data_dir = Path("data/raw/ecommerce")
    
    # Check if data exists
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Creating sample data for testing...")
        return create_sample_data()
    
    tables = {}
    
    # Load key tables for comprehensive testing
    table_files = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv', 
        'customers': 'olist_customers_dataset.csv',
        'products': 'olist_products_dataset.csv'
    }
    
    for table_name, filename in table_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Limit to manageable size for testing
                tables[table_name] = df.head(2000)
                print(f"✅ Loaded {table_name}: {df.shape}")
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
        else:
            print(f"⚠️ File not found: {filename}")
    
    # If no real data found, create sample data
    if not tables:
        print("No real data found, creating sample data...")
        return create_sample_data()
    
    return tables

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

def test_comprehensive_enhanced_agent():
    """Test the comprehensive enhanced agent with various question types"""
    
    print("🚀 COMPREHENSIVE ENHANCED AGENT TEST")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading E-commerce Data...")
    tables = load_sample_ecommerce_data()
    
    if not tables:
        print("❌ No data available for testing")
        return
    
    # Initialize components
    print("\n🧠 Initializing Intelligence Components...")
    enhanced_intelligence = EnhancedTableIntelligenceLayer()
    semantic_graph_builder = EnhancedKnowledgeGraphBuilder()
    enhanced_summarizer = None  # Will initialize if needed
    
    # Initialize comprehensive agent
    agent = ComprehensiveEnhancedAgent(
        enhanced_intelligence=enhanced_intelligence,
        semantic_graph_builder=semantic_graph_builder,
        enhanced_summarizer=enhanced_summarizer,
        llm_model="gpt-4o-mini-2024-07-18"
    )
    
    # Test scenarios that showcase intelligence capabilities
    test_scenarios = [
        {
            "name": "Quality Assessment Analysis",
            "question": "What are the main data quality issues in this e-commerce dataset?",
            "expected_intelligence": ["quality_profile", "critical_alerts", "missing_value_analysis"]
        },
        {
            "name": "Correlation Investigation", 
            "question": "What relationships exist between order values, shipping costs, and delivery times?",
            "expected_intelligence": ["correlation_analysis", "linear_relationships", "feature_redundancy"]
        },
        {
            "name": "Outlier Detection",
            "question": "Are there any unusual patterns or outliers in the pricing and order data?",
            "expected_intelligence": ["outlier_analysis", "high_impact_outliers", "anomaly_detection"]
        },
        {
            "name": "Temporal Analysis",
            "question": "What seasonal trends can we observe in the order data over time?",
            "expected_intelligence": ["temporal_analysis", "seasonal_patterns", "time_series"]
        },
        {
            "name": "Segmentation Analysis",
            "question": "How do customer segments differ in terms of purchasing behavior?",
            "expected_intelligence": ["segmentation", "customer_behavior", "purchasing_patterns"]
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
            # Run the agent
            result = agent.explore_for_insights(scenario['question'], tables)
            
            # Store results
            results[scenario['name']] = result
            
            # Display key results
            print(f"\n📊 EXPLORATION SUMMARY:")
            summary = result['exploration_summary']
            print(f"  Iterations: {summary['iterations_used']}")
            print(f"  Confidence: {summary['confidence_level']:.1%}")
            print(f"  Findings: {summary['total_findings']}")
            print(f"  Intelligence-Driven: {summary['intelligence_driven']}")
            print(f"  Operations: {summary['operations_executed']}")
            
            print(f"\n🧠 INTELLIGENCE CONTEXT:")
            intel_context = result['intelligence_context']
            print(f"  Profiles Generated: {intel_context['profiles_generated']}")
            print(f"  Analysis Plans: {dict(list(intel_context['analysis_plans'].items())[:3])}")
            
            print(f"\n💡 KEY INSIGHTS:")
            insights = result['insights']
            if isinstance(insights, dict):
                if 'direct_answer' in insights:
                    print(f"  Answer: {insights['direct_answer'][:150]}...")
                if 'key_insights' in insights:
                    for insight in insights['key_insights'][:3]:
                        print(f"  • {insight}")
                if 'confidence_score' in insights:
                    print(f"  Confidence: {insights['confidence_score']:.1%}")
            
            print(f"\n📋 RECOMMENDATIONS:")
            recommendations = result.get('recommendations', [])
            for rec in recommendations[:3]:
                print(f"  • {rec}")
            
            print(f"\n📈 DATA QUALITY SUMMARY:")
            quality = result.get('data_quality_summary', {})
            if quality:
                print(f"  Overall Score: {quality.get('overall_score', 0):.1%}")
                print(f"  Critical Issues: {quality.get('critical_issues', 0)}")
            
            # Validate intelligence usage
            print(f"\n✅ INTELLIGENCE VALIDATION:")
            validation_score = validate_intelligence_usage(result, scenario['expected_intelligence'])
            print(f"  Intelligence Usage Score: {validation_score:.1%}")
            
        except Exception as e:
            print(f"❌ ERROR in {scenario['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[scenario['name']] = {"error": str(e)}
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE ASSESSMENT")
    print(f"{'='*80}")
    
    successful_tests = len([r for r in results.values() if 'error' not in r])
    total_tests = len(test_scenarios)
    
    print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        # Calculate average confidence
        confidences = []
        intelligence_scores = []
        
        for name, result in results.items():
            if 'error' not in result:
                summary = result.get('exploration_summary', {})
                conf = summary.get('confidence_level', 0)
                confidences.append(conf)
                
                # Check intelligence usage
                intel_used = summary.get('intelligence_driven', False)
                ops_count = summary.get('operations_executed', 0)
                intelligence_scores.append(1.0 if intel_used and ops_count > 0 else 0.5)
        
        if confidences:
            avg_confidence = np.mean(confidences)
            avg_intelligence = np.mean(intelligence_scores)
            
            print(f"📊 Average Confidence: {avg_confidence:.1%}")
            print(f"🧠 Intelligence Usage: {avg_intelligence:.1%}")
            
            # Overall grade
            overall_score = (avg_confidence + avg_intelligence) / 2
            if overall_score >= 0.8:
                grade = "🟢 EXCELLENT"
            elif overall_score >= 0.6:
                grade = "🟡 GOOD" 
            else:
                grade = "🔴 NEEDS IMPROVEMENT"
            
            print(f"🎯 Overall Performance: {overall_score:.1%} ({grade})")
    
    return results

def validate_intelligence_usage(result, expected_features):
    """Validate that the agent properly used intelligence features"""
    score = 0.0
    total_checks = len(expected_features)
    
    # Check if intelligence context exists
    intel_context = result.get('intelligence_context', {})
    if intel_context.get('profiles_generated', 0) > 0:
        score += 0.2
    
    # Check if operations were intelligence-driven
    summary = result.get('exploration_summary', {})
    if summary.get('intelligence_driven', False):
        score += 0.3
    
    # Check for specific intelligence features
    insights = result.get('insights', {})
    intel_meta = insights.get('intelligence_metadata', {}) if isinstance(insights, dict) else {}
    
    if intel_meta.get('analysis_approach') == 'comprehensive_intelligence_driven':
        score += 0.3
    
    # Check recommendations based on intelligence
    recommendations = result.get('recommendations', [])
    intel_recs = [r for r in recommendations if any(keyword in r.lower() 
                  for keyword in ['quality', 'correlation', 'outlier', 'intelligence'])]
    if intel_recs:
        score += 0.2
    
    return min(1.0, score)

if __name__ == "__main__":
    # Set environment for testing
    os.environ.setdefault('OPENAI_API_KEY', 'your-key-here')
    
    # Suppress ydata-profiling progress bars
    os.environ['YDATA_PROFILING_DISABLE_PROGRESS_BAR'] = '1'
    
    # Suppress huggingface tokenizer warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Suppress warnings and progress bars
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        results = test_comprehensive_enhanced_agent()
        print("\n🎉 Testing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()