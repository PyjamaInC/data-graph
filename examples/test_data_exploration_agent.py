"""
Example usage of the Data Exploration ReAct Agent

This script demonstrates how to use the agent for discovering insights in data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from src.agents.react_agents.data_exploration_agent import DataExplorationReActAgent

# Create sample data for testing
def create_sample_data():
    """Create sample e-commerce data for testing"""
    
    # Orders data
    np.random.seed(42)
    n_orders = 1000
    
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.randint(1, 200, n_orders),
        'order_date': pd.date_range('2023-01-01', periods=n_orders, freq='h'),
        'order_value': np.random.exponential(50, n_orders) * (1 + np.random.normal(0, 0.2, n_orders)),
        'status': np.random.choice(['completed', 'pending', 'cancelled'], n_orders, p=[0.8, 0.15, 0.05]),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'cash'], n_orders),
        'shipping_cost': np.random.uniform(5, 25, n_orders),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_orders)
    })
    
    # Customers data
    n_customers = 200
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'customer_name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
        'join_date': pd.date_range('2022-01-01', periods=n_customers, freq='D'),
        'customer_type': np.random.choice(['regular', 'premium', 'vip'], n_customers, p=[0.7, 0.25, 0.05]),
        'age': np.random.randint(18, 70, n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers)
    })
    
    # Products data
    n_products = 50
    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product_{i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_products),
        'price': np.random.uniform(10, 500, n_products),
        'cost': np.random.uniform(5, 250, n_products),
        'stock_quantity': np.random.randint(0, 1000, n_products)
    })
    
    # Order items (linking orders to products)
    n_items = 2500
    order_items = pd.DataFrame({
        'order_id': np.random.randint(1, n_orders + 1, n_items),
        'product_id': np.random.randint(1, n_products + 1, n_items),
        'quantity': np.random.randint(1, 5, n_items),
        'unit_price': np.random.uniform(10, 500, n_items)
    })
    
    return {
        'orders': orders,
        'customers': customers,
        'products': products,
        'order_items': order_items
    }


def main():
    """Run data exploration examples"""
    
    print("🚀 Data Exploration ReAct Agent Demo")
    print("=" * 80)
    
    # Create sample data
    tables = create_sample_data()
    print(f"\n📊 Created sample data with {len(tables)} tables:")
    for name, df in tables.items():
        print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize the agent
    agent = DataExplorationReActAgent(llm_model="gpt-4o-mini-2024-07-18")
    
    # Test questions
    test_questions = [
        "What are the seasonal patterns in order values?",
        "Which customer segments generate the most revenue?",
        "Are there any correlations between payment methods and order values?",
        "What products have the highest profit margins?",
        "How does shipping cost vary by region?",
        "What factors influence order cancellation rates?"
    ]
    
    # Run exploration for each question
    for i, question in enumerate(test_questions[:2], 1):  # Limit to 2 for demo
        print(f"\n\n{'='*80}")
        print(f"📝 Question {i}: {question}")
        print("="*80)
        
        try:
            # Run the exploration
            result = agent.explore_for_insights(question, tables)
            
            # Display results
            print(f"\n📊 Exploration Summary:")
            print(f"  - Iterations: {result['exploration_summary']['iterations_used']}")
            print(f"  - Confidence: {result['exploration_summary']['confidence_level']:.2%}")
            print(f"  - Total Findings: {result['exploration_summary']['total_findings']}")
            
            print(f"\n💡 Direct Answer:")
            print(f"  {result['insights']['direct_answer']}")
            
            print(f"\n🔍 Key Insights:")
            for insight in result['insights']['key_insights']:
                print(f"  • {insight}")
            
            print(f"\n📈 Supporting Evidence:")
            for evidence in result['insights']['supporting_evidence'][:3]:
                print(f"  • {evidence}")
            
            print(f"\n💭 Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"  • {rec}")
                
        except Exception as e:
            print(f"\n❌ Error during exploration: {e}")
            import traceback
            traceback.print_exc()


def test_specific_operations():
    """Test specific pandas operations"""
    
    print("\n\n🧪 Testing Specific Operations")
    print("=" * 80)
    
    # Create simple test data
    test_data = {
        'sales': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'revenue': np.random.uniform(1000, 5000, 100),
            'units': np.random.randint(10, 100, 100),
            'region': np.random.choice(['North', 'South'], 100)
        })
    }
    
    agent = DataExplorationReActAgent()
    
    # Test the toolkit directly
    toolkit = agent.toolkit
    
    test_operations = [
        "tables['sales'].shape",
        "tables['sales']['revenue'].mean()",
        "tables['sales'].groupby('region')['revenue'].sum()",
        "tables['sales'].set_index('date').resample('M')['revenue'].sum()",
        "tables['sales'][['revenue', 'units']].corr()"
    ]
    
    for op in test_operations:
        print(f"\n🔧 Testing: {op}")
        result = toolkit.execute_pandas_operation(op, test_data)
        if result['success']:
            print(f"✅ Result: {agent._summarize_result(result)}")
        else:
            print(f"❌ Error: {result['error']['message']}")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Optionally test specific operations
    # test_specific_operations()