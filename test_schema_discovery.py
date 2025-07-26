"""
Test script for schema discovery with Brazilian e-commerce data
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append('src')

from schema.schema_manager import SchemaManager

def test_schema_discovery():
    """Test auto-discovery with the Brazilian e-commerce dataset"""
    
    # Load the Brazilian e-commerce data
    data_dir = Path("data/raw/ecommerce")
    
    datasets = {}
    csv_files = list(data_dir.glob("*.csv"))
    
    print(f"Loading {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=1000)  # Limit rows for faster testing
            table_name = csv_file.stem.replace('olist_', '').replace('_dataset', '')
            datasets[table_name] = df
            print(f"✓ Loaded {table_name}: {df.shape}")
        except Exception as e:
            print(f"✗ Failed to load {csv_file}: {e}")
    
    if not datasets:
        print("No datasets loaded!")
        return
    
    # Initialize schema manager
    schema_manager = SchemaManager()
    
    # Auto-discover schema
    print("\n🔍 Auto-discovering schema...")
    schema = schema_manager.discover_schema_from_data(datasets, "brazilian_ecommerce")
    
    # Display results
    print(f"\n📊 Schema Discovery Results for '{schema.name}':")
    print(f"Dataset: {schema.description}")
    print(f"Tables: {len(schema.tables)}")
    
    for table_name, table_schema in schema.tables.items():
        print(f"\n📋 Table: {table_name}")
        print(f"  Business Domain: {table_schema.business_domain}")
        print(f"  Columns: {len(table_schema.columns)}")
        print(f"  Foreign Keys: {table_schema.foreign_keys}")
        
        # Show some interesting columns
        measures = []
        dimensions = []
        identifiers = []
        
        for col_name, col_schema in table_schema.columns.items():
            if col_schema.semantic_role.value == 'measure':
                measures.append(f"{col_name} ({col_schema.data_type.value})")
            elif col_schema.semantic_role.value == 'dimension':
                dimensions.append(f"{col_name} ({col_schema.cardinality} unique)")
            elif col_schema.semantic_role.value == 'identifier':
                identifiers.append(col_name)
        
        if measures:
            print(f"  📈 Measures: {', '.join(measures[:3])}")
        if dimensions:
            print(f"  📊 Dimensions: {', '.join(dimensions[:3])}")
        if identifiers:
            print(f"  🔑 Identifiers: {', '.join(identifiers[:3])}")
    
    # Save schema as YAML config
    print(f"\n💾 Saving schema configuration...")
    config_path = schema_manager.save_schema_config(schema, format='yaml')
    print(f"Schema saved to: {config_path}")
    
    # Demonstrate query planning based on schema
    print(f"\n🤖 Example Query Planning:")
    print("For query: 'sales analysis by different factors'")
    
    # Find relevant columns
    sales_columns = []
    groupby_columns = []
    
    for table_name, table_schema in schema.tables.items():
        for col_name, col_schema in table_schema.columns.items():
            # Look for sales-related measures
            if (col_schema.semantic_role.value == 'measure' and 
                any(keyword in col_name.lower() for keyword in ['price', 'value', 'amount', 'total'])):
                sales_columns.append(f"{table_name}.{col_name}")
            
            # Look for good groupby dimensions
            if (col_schema.semantic_role.value == 'dimension' and 
                col_schema.cardinality and col_schema.cardinality < 50):
                groupby_columns.append(f"{table_name}.{col_name}")
    
    print(f"  📈 Sales Measures: {sales_columns[:5]}")
    print(f"  📊 Groupby Options: {groupby_columns[:5]}")
    
    return schema

if __name__ == "__main__":
    schema = test_schema_discovery()