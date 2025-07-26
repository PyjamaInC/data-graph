"""
Test basic schema discovery functionality
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append('src')

from schema.schema_manager import SchemaManager

def test_basic_schema():
    """Test basic schema discovery without profiling"""
    
    print("🚀 Testing Basic Schema Discovery")
    print("=" * 60)
    
    # Load sample data
    data_dir = Path("data/raw/ecommerce")
    datasets = {}
    
    # Load just a few files for quick testing
    test_files = ['olist_customers_dataset.csv', 'olist_orders_dataset.csv', 'olist_order_items_dataset.csv']
    
    for filename in test_files:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, nrows=1000)
            table_name = filename.replace('olist_', '').replace('_dataset.csv', '')
            datasets[table_name] = df
            print(f"✓ Loaded {table_name}: {df.shape}")
    
    # Test without profiling
    schema_manager = SchemaManager()
    schema_manager.auto_discovery.use_profiling = False  # Disable profiling
    
    print("\n📊 Discovering schema...")
    schema = schema_manager.discover_schema_from_data(datasets, "test_ecommerce")
    
    print(f"\n✅ Schema discovered: {schema.name}")
    print(f"Tables: {len(schema.tables)}")
    
    # Display schema details
    for table_name, table_schema in schema.tables.items():
        print(f"\n📋 Table: {table_name}")
        print(f"  Columns: {len(table_schema.columns)}")
        print(f"  Foreign Keys: {table_schema.foreign_keys}")
        
        # Show some columns
        for i, (col_name, col_schema) in enumerate(table_schema.columns.items()):
            if i < 5:  # Show first 5 columns
                print(f"  • {col_name}:")
                print(f"    - Type: {col_schema.data_type.value}")
                print(f"    - Role: {col_schema.semantic_role.value}")
                print(f"    - Nullable: {col_schema.nullable}")
    
    # Save schema
    output_path = Path("schemas/test_schema.yaml")
    saved_path = schema_manager.save_schema_config(schema, output_path, format='yaml')
    print(f"\n💾 Schema saved to: {saved_path}")
    
    return schema

if __name__ == "__main__":
    schema = test_basic_schema()