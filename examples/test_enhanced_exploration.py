"""
Test Enhanced Data Exploration with Full Intelligence

This script demonstrates the enhanced agent that properly uses all the
knowledge graph infrastructure for intelligent data exploration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

# Import enhanced components
from src.agents.react_agents.enhanced_data_exploration_agent import EnhancedDataExplorationReActAgent
from src.knowledge_graph.table_intelligence import EnhancedTableIntelligenceLayer
from src.knowledge_graph.semantic_table_graph import SemanticTableGraphBuilder
from src.knowledge_graph.enhanced_llm_summarizer import EnhancedLLMSemanticSummarizer
from src.knowledge_graph.llm_config import LLMConfigManager


def create_enhanced_agent():
    """Create agent with full intelligence infrastructure"""
    
    print("🔧 Initializing intelligence components...")
    
    # Initialize LLM configuration
    try:
        llm_config_manager = LLMConfigManager()
        llm_config = llm_config_manager.get_active_config()
    except Exception as e:
        print(f"⚠️ Using default LLM config: {e}")
        llm_config = None
    
    # Create enhanced table intelligence layer
    enhanced_intelligence = EnhancedTableIntelligenceLayer(
        model_name='all-MiniLM-L6-v2',
        enable_profiling=True,
        enable_advanced_quality=True,
        enable_outlier_detection=True,
        enable_correlation_analysis=True,
        enable_temporal_analysis=True,
        enable_ml_classification=True,
        use_llm_summaries=False,  # Set to False to avoid LLM dependency in testing
        llm_config=llm_config
    )
    
    # Create knowledge graph and semantic graph builder
    kg = nx.MultiDiGraph()
    semantic_graph_builder = SemanticTableGraphBuilder(kg, enhanced_intelligence)
    
    # Create enhanced LLM summarizer (optional)
    enhanced_summarizer = None
    
    # Initialize agent with full intelligence
    agent = EnhancedDataExplorationReActAgent(
        enhanced_intelligence=enhanced_intelligence,
        semantic_graph_builder=semantic_graph_builder,
        enhanced_summarizer=enhanced_summarizer,
        llm_model="gpt-4o-mini-2024-07-18"
    )
    
    print("✅ Intelligence components initialized successfully")
    
    return agent, enhanced_intelligence, semantic_graph_builder


def create_realistic_sample_data():
    """Create more realistic sample data with proper temporal coverage"""
    
    # Create data with different temporal spans
    np.random.seed(42)
    
    # Orders data - 45 days (insufficient for seasonal)
    n_orders = 2000
    start_date = datetime.now() - timedelta(days=45)
    
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.randint(1, 300, n_orders),
        'order_date': pd.date_range(start=start_date, periods=n_orders, freq='30min'),
        'order_value': np.random.exponential(75, n_orders) * (1 + np.random.normal(0, 0.15, n_orders)),
        'status': np.random.choice(['completed', 'pending', 'cancelled', 'refunded'], 
                                  n_orders, p=[0.75, 0.15, 0.08, 0.02]),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer'], 
                                         n_orders, p=[0.4, 0.3, 0.2, 0.1]),
        'shipping_cost': np.where(
            np.random.exponential(75, n_orders) > 50,  # Free shipping over $50
            0,
            np.random.uniform(5, 15, n_orders)
        ),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 
                                  n_orders, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 
                                       n_orders, p=[0.5, 0.4, 0.1])
    })
    
    # Add some data quality issues
    orders.loc[np.random.choice(orders.index, 50, replace=False), 'shipping_cost'] = np.nan
    orders.loc[np.random.choice(orders.index, 30, replace=False), 'device_type'] = np.nan
    
    # Customers data with join date spanning 2 years
    n_customers = 300
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'customer_name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
        'join_date': pd.date_range(end=datetime.now(), periods=n_customers, freq='2D'),
        'customer_type': np.random.choice(['regular', 'premium', 'vip', 'new'], 
                                        n_customers, p=[0.6, 0.25, 0.1, 0.05]),
        'age': np.random.randint(18, 75, n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 
                                 'Phoenix', 'Philadelphia', 'San Antonio'], n_customers),
        'lifetime_value': np.random.exponential(500, n_customers),
        'email_verified': np.random.choice([True, False], n_customers, p=[0.85, 0.15]),
        'preferred_contact': np.random.choice(['email', 'sms', 'phone'], 
                                            n_customers, p=[0.7, 0.25, 0.05])
    })
    
    # Products data
    n_products = 100
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys', 'Health']
    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product_{i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'subcategory': [f'Sub_{i%10}' for i in range(1, n_products + 1)],
        'price': np.random.uniform(10, 500, n_products),
        'cost': np.random.uniform(5, 250, n_products),
        'stock_quantity': np.random.randint(0, 1000, n_products),
        'supplier_id': np.random.randint(1, 20, n_products),
        'rating': np.round(np.random.uniform(3.0, 5.0, n_products), 1),
        'review_count': np.random.randint(0, 500, n_products)
    })
    
    # Calculate profit margin
    products['profit_margin'] = (products['price'] - products['cost']) / products['price'] * 100
    
    # Order items with realistic relationships
    order_items = []
    for order_id in orders['order_id'].sample(int(0.8 * len(orders))):  # 80% of orders have items
        n_items = np.random.randint(1, 5)  # 1-4 items per order
        for _ in range(n_items):
            order_items.append({
                'order_id': order_id,
                'product_id': np.random.randint(1, n_products + 1),
                'quantity': np.random.randint(1, 3),
                'unit_price': products.sample(1)['price'].values[0] * np.random.uniform(0.9, 1.1),
                'discount_applied': np.random.uniform(0, 0.2)
            })
    
    order_items_df = pd.DataFrame(order_items)
    
    return {
        'orders': orders,
        'customers': customers,
        'products': products,
        'order_items': order_items_df
    }


def test_intelligent_exploration():
    """Test the enhanced agent with intelligence"""
    
    print("🚀 Enhanced Data Exploration with Full Intelligence")
    print("=" * 80)
    
    # Create realistic sample data
    print("\n📊 Creating realistic sample data...")
    tables = create_realistic_sample_data()
    
    print(f"\nData created:")
    for name, df in tables.items():
        print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"    Date range: {df.select_dtypes(include=['datetime64']).min().min() if not df.select_dtypes(include=['datetime64']).empty else 'No dates'}")
    
    # Initialize enhanced agent
    print("\n🧠 Initializing Enhanced Agent with Intelligence...")
    agent, enhanced_intelligence, semantic_graph_builder = create_enhanced_agent()
    
    # Test questions that will trigger different behaviors
    test_questions = [
        {
            "question": "What are the seasonal patterns in order values?",
            "expected": "Should detect insufficient temporal data (only 45 days)"
        },
        {
            "question": "Which customer segments generate the most revenue?",
            "expected": "Should use customer profiling and segmentation"
        },
        {
            "question": "What factors correlate with high order values?",
            "expected": "Should use correlation analysis from intelligence"
        },
        {
            "question": "Are there any unusual patterns or anomalies in the order data?",
            "expected": "Should use outlier detection capabilities"
        },
        {
            "question": "How does shipping cost vary by region and order value?",
            "expected": "Should identify relationships and patterns"
        }
    ]
    
    # Test first two questions
    for i, test_case in enumerate(test_questions[:2], 1):
        print(f"\n\n{'='*80}")
        print(f"📝 Test Question {i}: {test_case['question']}")
        print(f"📌 Expected: {test_case['expected']}")
        print("="*80)
        
        try:
            # Run intelligent exploration
            result = agent.explore_for_insights(test_case['question'], tables)
            
            # Display enhanced results
            print_enhanced_results(result)
            
        except Exception as e:
            print(f"\n❌ Error during exploration: {e}")
            import traceback
            traceback.print_exc()


def print_enhanced_results(result):
    """Print results with intelligence context"""
    
    print(f"\n📊 Exploration Summary:")
    summary = result['exploration_summary']
    print(f"  - Strategy Used: {summary.get('strategy_used', 'unknown')}")
    print(f"  - Iterations: {summary['iterations_used']}")
    print(f"  - Confidence: {summary['confidence_level']:.2%}")
    print(f"  - Total Findings: {summary['total_findings']}")
    
    # Intelligence usage summary
    if 'intelligence_summary' in result:
        print(f"\n🧠 Intelligence Usage:")
        intel = result['intelligence_summary']
        print(f"  - Profiles Generated: {intel.get('profiles_generated', 0)}")
        print(f"  - Relationships Detected: {intel.get('relationships_detected', 0)}")
        print(f"  - Quality Issues Found: {intel.get('quality_issues_found', 0)}")
        print(f"  - Exploration Strategy: {intel.get('exploration_strategy', 'N/A')}")
    
    print(f"\n💡 Direct Answer:")
    print(f"  {result['insights']['direct_answer']}")
    
    print(f"\n🔍 Key Insights:")
    for insight in result['insights']['key_insights'][:3]:
        print(f"  • {insight}")
    
    print(f"\n📈 Supporting Evidence:")
    for evidence in result['insights']['supporting_evidence'][:3]:
        print(f"  • {evidence}")
    
    if result['insights'].get('data_quality_notes'):
        print(f"\n⚠️ Data Quality Notes:")
        for note in result['insights']['data_quality_notes'][:3]:
            print(f"  • {note}")
    
    print(f"\n🎯 Confidence Score: {result['insights'].get('confidence_score', 0):.1%}")
    
    print(f"\n🔮 Intelligent Recommendations:")
    for rec in result['recommendations'][:3]:
        print(f"  • {rec}")


def test_specific_intelligence_features():
    """Test specific intelligence features"""
    
    print("\n\n🧪 Testing Specific Intelligence Features")
    print("=" * 80)
    
    # Create minimal test data
    test_data = {
        'sales': pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'revenue': np.random.uniform(1000, 5000, 30),
            'units': np.random.randint(10, 100, 30),
            'region': np.random.choice(['North', 'South'], 30),
            'product_category': np.random.choice(['A', 'B', 'C'], 30)
        })
    }
    
    # Add some missing values
    test_data['sales'].loc[5:10, 'revenue'] = np.nan
    
    print("\n📊 Test Data Created:")
    print(f"  - Shape: {test_data['sales'].shape}")
    print(f"  - Date Range: {test_data['sales']['date'].min()} to {test_data['sales']['date'].max()}")
    print(f"  - Missing Values: {test_data['sales'].isnull().sum().sum()}")
    
    # Create agent
    agent, _, _ = create_enhanced_agent()
    
    # Test temporal limitation detection
    question = "What are the yearly seasonal trends in revenue?"
    print(f"\n🔍 Testing: {question}")
    
    try:
        result = agent.explore_for_insights(question, test_data)
        print(f"\n✅ Result: {result['insights']['direct_answer']}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Run the main test
    test_intelligent_exploration()
    
    # Optionally test specific features
    # test_specific_intelligence_features()