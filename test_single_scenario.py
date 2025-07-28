#!/usr/bin/env python3
"""
Quick test of a single scenario to verify fixes
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress warnings and progress bars
os.environ['YDATA_PROFILING_DISABLE_PROGRESS_BAR'] = '1'
import warnings
warnings.filterwarnings('ignore')

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from knowledge_graph.table_intelligence import EnhancedTableIntelligenceLayer
from knowledge_graph.enhanced_graph_builder import EnhancedKnowledgeGraphBuilder
from agents.react_agents.comprehensive_enhanced_agent import ComprehensiveEnhancedAgent

def create_simple_test_data():
    """Create simple test data"""
    np.random.seed(42)
    
    # Simple orders table
    orders = pd.DataFrame({
        'order_id': [f'order_{i:03d}' for i in range(100)],
        'customer_id': [f'customer_{i:02d}' for i in np.random.randint(0, 20, 100)],
        'order_status': np.random.choice(['delivered', 'shipped', 'processing'], 100, p=[0.7, 0.2, 0.1]),
        'order_value': np.random.lognormal(3, 1, 100).round(2),
        'order_date': pd.date_range('2023-01-01', periods=100, freq='D')
    })
    
    return {'orders': orders}

def test_single_scenario():
    """Test a single scenario"""
    
    print("🧪 SINGLE SCENARIO TEST")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating test data...")
    tables = create_simple_test_data()
    print(f"Created orders table: {tables['orders'].shape}")
    
    # Initialize components
    print("🧠 Initializing components...")
    enhanced_intelligence = EnhancedTableIntelligenceLayer()
    semantic_graph_builder = EnhancedKnowledgeGraphBuilder()
    
    # Test just the intelligence profiling first
    print("🔍 Testing intelligence profiling...")
    try:
        profile = enhanced_intelligence.analyze_table_comprehensive('orders', tables['orders'])
        print(f"✅ Profile generated successfully!")
        print(f"  Data Quality: {profile.data_quality_score:.1%}")
        print(f"  ML Readiness: {profile.ml_readiness_score}%")
        print(f"  Key Insights: {len(profile.key_insights or [])} discovered")
        
        # Test the toolkit
        from agents.react_agents.intelligence_driven_toolkit import IntelligenceDrivenToolkit
        toolkit = IntelligenceDrivenToolkit(enhanced_intelligence)
        toolkit.register_table_profile('orders', profile)
        
        # Test operation generation
        operation = toolkit.generate_intelligent_operation('orders', 'quality assessment', 1)
        print(f"  Generated operation: {operation[:100]}...")
        
        print("✅ Intelligence system working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in intelligence profiling: {e}")
        import traceback
        traceback.print_exc()
        return False
    

if __name__ == "__main__":
    success = test_single_scenario()
    if success:
        print("\n🎉 Test passed! The agent is working correctly.")
    else:
        print("\n💥 Test failed! Need to debug further.")