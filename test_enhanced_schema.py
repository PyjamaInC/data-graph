"""
Test enhanced schema discovery with ydata-profiling integration
"""

import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime

sys.path.append('src')

from schema.schema_manager import SchemaManager
from knowledge_graph.enhanced_relationship_detector import ProfilingBasedRelationshipDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_enhanced_schema_discovery():
    """Test enhanced schema discovery with Brazilian e-commerce data"""
    
    print("🚀 Testing Enhanced Schema Discovery with ydata-profiling")
    print("=" * 60)
    
    # Load the Brazilian e-commerce data
    data_dir = Path("data/raw/ecommerce")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return None
    
    datasets = {}
    csv_files = list(data_dir.glob("*.csv"))
    
    print(f"\n📁 Loading {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        try:
            # Load full dataset (or limit for testing)
            df = pd.read_csv(csv_file, nrows=5000)  # Limit for faster testing
            table_name = csv_file.stem.replace('olist_', '').replace('_dataset', '')
            datasets[table_name] = df
            print(f"  ✓ Loaded {table_name}: {df.shape}")
        except Exception as e:
            print(f"  ✗ Failed to load {csv_file}: {e}")
    
    if not datasets:
        print("❌ No datasets loaded!")
        return None
    
    # Test 1: Schema Discovery with ydata-profiling
    print("\n" + "="*60)
    print("📊 TEST 1: Schema Discovery with ydata-profiling")
    print("="*60)
    
    schema_manager = SchemaManager()
    
    start_time = datetime.now()
    schema = schema_manager.discover_schema_from_data(datasets, "brazilian_ecommerce")
    discovery_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️  Discovery completed in {discovery_time:.2f} seconds")
    print(f"\n📋 Schema Summary:")
    print(f"  • Dataset: {schema.name}")
    print(f"  • Description: {schema.description}")
    print(f"  • Tables: {len(schema.tables)}")
    print(f"  • Created: {schema.created_at}")
    
    # Display detailed insights for each table
    for table_name, table_schema in schema.tables.items():
        print(f"\n📊 Table: {table_name}")
        print(f"  • Columns: {len(table_schema.columns)}")
        print(f"  • Business Domain: {table_schema.business_domain}")
        print(f"  • Foreign Keys: {len(table_schema.foreign_keys)}")
        
        # Group columns by semantic role
        role_groups = {}
        for col_name, col_schema in table_schema.columns.items():
            role = col_schema.semantic_role.value
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append((col_name, col_schema))
        
        # Display columns by role
        for role, columns in role_groups.items():
            if columns:
                print(f"\n  🏷️  {role.upper()} columns ({len(columns)}):")
                for col_name, col_schema in columns[:3]:  # Show first 3
                    print(f"    • {col_name}:")
                    print(f"      - Type: {col_schema.data_type.value}")
                    print(f"      - Null %: {col_schema.null_ratio*100:.1f}%")
                    print(f"      - Unique %: {col_schema.unique_ratio*100:.1f}%")
                    if col_schema.business_domain:
                        print(f"      - Domain: {col_schema.business_domain}")
                    if col_schema.aggregation_methods:
                        print(f"      - Aggregations: {', '.join(col_schema.aggregation_methods)}")
    
    # Test 2: Relationship Detection with Profiling
    print("\n" + "="*60)
    print("🔗 TEST 2: Relationship Detection with ydata-profiling")
    print("="*60)
    
    relationship_detector = ProfilingBasedRelationshipDetector()
    
    start_time = datetime.now()
    relationships = relationship_detector.detect_dataset_relationships(datasets, sample_size=2000)
    detection_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️  Relationship detection completed in {detection_time:.2f} seconds")
    
    # Display discovered relationships
    print(f"\n🔍 Discovered Relationships:")
    
    # Correlations
    for corr_type, correlations in relationships.get('correlations', {}).items():
        if correlations:
            print(f"\n  📈 {corr_type.upper()} Correlations ({len(correlations)}):")
            for corr in correlations[:5]:  # Show top 5
                print(f"    • {corr['source']} ↔ {corr['target']}")
                print(f"      Correlation: {corr['correlation']:.3f} ({corr['strength']} {corr['direction']})")
    
    # Foreign Keys
    foreign_keys = relationships.get('foreign_keys', [])
    if foreign_keys:
        print(f"\n  🔑 Foreign Key Relationships ({len(foreign_keys)}):")
        for fk in foreign_keys:
            print(f"    • {fk['source_table']}.{fk['source_column']} → "
                  f"{fk['target_table']}.{fk['target_column']} "
                  f"(confidence: {fk['confidence']:.2f})")
    
    # Missing Patterns
    missing = relationships.get('missing_patterns', {})
    high_missing = missing.get('high_missing_columns', [])
    if high_missing:
        print(f"\n  ⚠️  High Missing Value Columns:")
        for col_info in high_missing[:5]:
            print(f"    • {col_info['column']}: {col_info['missing_percentage']:.1f}% missing")
    
    # Get recommendations
    recommendations = relationship_detector.generate_analysis_recommendations(relationships)
    if recommendations:
        print(f"\n💡 Analysis Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Test 3: Save and Load Schema Configuration
    print("\n" + "="*60)
    print("💾 TEST 3: Schema Configuration Management")
    print("="*60)
    
    # Save as YAML
    yaml_path = schema_manager.save_schema_config(schema, format='yaml')
    print(f"  ✓ Schema saved to: {yaml_path}")
    
    # Save as JSON
    json_path = schema_manager.save_schema_config(
        schema, 
        output_path=Path("schemas/brazilian_ecommerce_schema.json"),
        format='json'
    )
    print(f"  ✓ Schema saved to: {json_path}")
    
    # Test loading from config
    loaded_schema = schema_manager.load_schema_from_config(yaml_path)
    print(f"  ✓ Schema loaded successfully: {loaded_schema.name}")
    
    # Test 4: Query Planning Example
    print("\n" + "="*60)
    print("🤖 TEST 4: Query Planning Based on Schema")
    print("="*60)
    
    # Example queries to test
    test_queries = [
        "Show me sales analysis by different factors",
        "Customer segmentation analysis",
        "Product performance over time",
        "Geographic distribution of orders"
    ]
    
    print("\nExample column recommendations for queries:")
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        # Find relevant columns based on schema
        relevant_measures = []
        relevant_dimensions = []
        relevant_temporal = []
        
        for table_name, table_schema in schema.tables.items():
            for col_name, col_schema in table_schema.columns.items():
                full_name = f"{table_name}.{col_name}"
                
                # Match based on query keywords and semantic roles
                if 'sales' in query.lower() and col_schema.semantic_role.value == 'measure':
                    if any(kw in col_name.lower() for kw in ['price', 'value', 'amount', 'total']):
                        relevant_measures.append(full_name)
                
                elif 'customer' in query.lower() and 'customer' in table_name:
                    if col_schema.semantic_role.value == 'dimension':
                        relevant_dimensions.append(full_name)
                
                elif 'time' in query.lower() and col_schema.semantic_role.value == 'temporal':
                    relevant_temporal.append(full_name)
                
                elif 'geographic' in query.lower() and col_schema.semantic_role.value == 'geographical':
                    relevant_dimensions.append(full_name)
        
        if relevant_measures:
            print(f"  📊 Measures: {', '.join(relevant_measures[:3])}")
        if relevant_dimensions:
            print(f"  📈 Dimensions: {', '.join(relevant_dimensions[:3])}")
        if relevant_temporal:
            print(f"  📅 Time columns: {', '.join(relevant_temporal[:3])}")
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)
    
    return schema, relationships

if __name__ == "__main__":
    schema, relationships = test_enhanced_schema_discovery()