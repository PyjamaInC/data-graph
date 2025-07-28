import asyncio
import sys
sys.path.append('.')

from src.data.connectors.csv_connector import CSVConnector

async def test_data_loading():
    """Test loading the Brazilian e-commerce dataset"""
    connector = CSVConnector()
    
    # Load e-commerce data
    print("Loading e-commerce dataset...")
    tables = await connector.load_data({
        'data_path': 'data/raw/ecommerce'
    })
    
    # Analyze schema
    print("\nAnalyzing schema...")
    metadata = await connector.analyze_schema()
    
    print("\nLoaded Tables:")
    print("-" * 50)
    for name, meta in metadata.items():
        print(f"{name:30} | {meta.row_count:7} rows | {meta.column_count:3} columns")
    
    # Show sample data from each table
    print("\nSample Data:")
    print("=" * 50)
    for table_name, df in tables.items():
        print(f"\n{table_name}:")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"Data types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print("-" * 50)
    
    return connector, tables

if __name__ == "__main__":
    connector, tables = asyncio.run(test_data_loading())