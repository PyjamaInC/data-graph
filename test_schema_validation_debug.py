#!/usr/bin/env python3
"""
Debug Schema Validation Agent Issues
Extract and run notebook testing code to identify schema problems
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def load_ecommerce_data():
    """Load the ecommerce dataset"""
    data_dir = Path("data/raw/ecommerce")
    datasets = {}
    
    files_to_load = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv', 
        'order_items': 'olist_order_items_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv'
    }
    
    print("🔍 Checking available files:")
    for table_name, filename in files_to_load.items():
        filepath = data_dir / filename
        print(f"   {table_name}: {filepath} - {'✅ exists' if filepath.exists() else '❌ missing'}")
    
    for table_name, filename in files_to_load.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                # Load sample for faster processing
                df = pd.read_csv(filepath, nrows=1000)
                datasets[table_name] = df
                print(f"✅ Loaded {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"   Columns: {list(df.columns)}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⚠️ File not found: {filepath}")
    
    return datasets

def test_schema_discovery():
    """Test schema discovery process"""
    print("=" * 60)
    print("🔧 Testing Schema Discovery")
    print("=" * 60)
    
    # Load data
    datasets = load_ecommerce_data()
    
    if not datasets:
        print("❌ No datasets available for testing")
        return None, None
    
    try:
        from schema.schema_manager import SchemaManager
        
        print(f"\n🔧 Initializing Schema Manager...")
        schema_manager = SchemaManager()
        schema_manager.auto_discovery.use_profiling = False  # Faster for testing
        
        # Discover schema and assign it back to schema_manager
        print(f"🔍 Discovering schema from {len(datasets)} datasets...")
        discovered_schema = schema_manager.discover_schema_from_data(datasets, "ecommerce")
        schema_manager.schema = discovered_schema
        
        print(f"✅ Schema discovered: {len(discovered_schema.tables)} tables")
        
        # Show detailed schema summary
        print(f"\n📋 Detailed Schema Information:")
        for table_name, table_schema in discovered_schema.tables.items():
            print(f"   📋 {table_name}: {len(table_schema.columns)} columns")
            for col_name, col_schema in table_schema.columns.items():
                role = col_schema.semantic_role.value if hasattr(col_schema.semantic_role, 'value') else str(col_schema.semantic_role)
                print(f"      - {col_name}: {role}")
        
        return schema_manager, datasets
        
    except Exception as e:
        print(f"❌ Schema discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return None, datasets

def test_schema_validation_agent(schema_manager):
    """Test the schema validation agent with detailed debugging"""
    print("=" * 60)
    print("🗂️ Testing Schema Validation Agent")
    print("=" * 60)
    
    if not schema_manager:
        print("❌ No schema manager available")
        return
    
    try:
        from agents.react_agents.schema_validator import SchemaValidationAgent
        from agents.react_agents.state_manager import StateManager
        
        schema_agent = SchemaValidationAgent(schema_manager)
        state_manager = StateManager()
        
        # Test queries that were failing
        test_cases = [
            {
                'query': 'Show customer orders with total prices by location',
                'intent': {
                    'action_type': 'geographical_analysis',
                    'target_concepts': ['customer', 'orders', 'location', 'total', 'prices'],
                    'analysis_scope': 'multi_table',
                    'confidence': 0.9
                }
            },
            {
                'query': 'What are the top selling products by revenue?',
                'intent': {
                    'action_type': 'aggregation',
                    'target_concepts': ['top', 'selling', 'products', 'revenue'],
                    'analysis_scope': 'multi_table', 
                    'confidence': 0.9
                }
            },
            {
                'query': 'Analyze sales trends over time by month',
                'intent': {
                    'action_type': 'trend_analysis',
                    'target_concepts': ['sales', 'trends', 'time', 'month'],
                    'analysis_scope': 'temporal',
                    'confidence': 0.9
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['query']}")
            print(f"   Intent concepts: {test_case['intent']['target_concepts']}")
            
            # Initialize state with intent
            state = state_manager.initialize_state(test_case['query'], "E-commerce")
            state['intent_profile'] = test_case['intent']
            
            # Debug: Show what we're looking for
            print(f"   🔍 Looking for concepts in schema...")
            
            # Manual concept matching debug
            for concept in test_case['intent']['target_concepts']:
                print(f"     Searching for '{concept}':")
                matches = schema_agent._find_concept_matches(concept, schema_agent._determine_required_roles(test_case['intent']['action_type']))
                if matches:
                    for match in matches[:2]:  # Show top 2
                        print(f"       ✅ {match['table']}.{match['column']} ({match['confidence']:.2f}) - {match['match_type']}")
                else:
                    print(f"       ❌ No matches found")
            
            try:
                result = schema_agent.execute(state)
                
                if 'validated_mapping' in result:
                    mapping = result['validated_mapping']
                    print(f"   📋 Tables: {mapping.get('relevant_tables', [])}")
                    print(f"   🔗 Joins needed: {mapping.get('joins_needed', False)}")
                    print(f"   ⭐ Confidence: {mapping.get('mapping_confidence', 0):.2f}")
                    
                    # Show concept mappings
                    concept_mappings = mapping.get('concept_mappings', {})
                    for concept, matches in concept_mappings.items():
                        if matches:
                            best_match = matches[0]
                            print(f"   📍 {concept} → {best_match['table']}.{best_match['column']} ({best_match['confidence']:.2f})")
                        else:
                            print(f"   ❌ {concept} → No mapping found")
                
                if 'performance_metrics' in result:
                    metrics = result['performance_metrics']
                    print(f"   📊 Tokens: {metrics.get('total_tokens', 0)}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ Failed to import ReAct components: {e}")
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run comprehensive debugging"""
    print("🧪 Schema Validation Debug Script")
    print("=" * 60)
    
    # Test schema discovery first
    schema_manager, datasets = test_schema_discovery()
    
    if schema_manager:
        # Test schema validation agent
        test_schema_validation_agent(schema_manager)
    else:
        print("❌ Cannot test schema validation without working schema manager")
    
    print("\n" + "=" * 60)
    print("🎯 Debug Summary:")
    print("   1. Check which tables are actually discovered")
    print("   2. Verify concept matching logic")
    print("   3. Identify why 'products' concept fails")
    print("   4. Improve semantic matching without hardcoding")

if __name__ == "__main__":
    main()