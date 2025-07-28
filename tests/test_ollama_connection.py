#!/usr/bin/env python3
"""Test script to verify Ollama connectivity and model availability - Updated 2024"""

import sys
import json
try:
    import ollama
    print("✓ Ollama library imported successfully")
except ImportError:
    print("✗ Ollama library not found. Please install with: pip install ollama")
    sys.exit(1)

def test_ollama_connection():
    """Test basic connectivity to Ollama"""
    try:
        # List available models
        models = ollama.list()
        print(f"\n✓ Connected to Ollama successfully")
        print(f"Available models: {len(models.get('models', []))}")
        
        if models.get('models'):
            print("\nModels installed:")
            for i, model in enumerate(models['models']):
                try:
                    # Handle different possible model data structures
                    model_name = model.get('name', model.get('model', f'model_{i}'))
                    model_size = model.get('size', 0)
                    
                    # Convert size to GB if it's a number
                    if isinstance(model_size, (int, float)) and model_size > 0:
                        size_gb = model_size / 1e9
                        print(f"  - {model_name} (size: {size_gb:.2f} GB)")
                    else:
                        print(f"  - {model_name} (size: unknown)")
                        
                    # Debug: print the raw model data if there are issues
                    if not model.get('name') and not model.get('model'):
                        print(f"    Debug - Raw model data: {model}")
                        
                except Exception as model_error:
                    print(f"  - Model {i}: Error reading model info - {model_error}")
                    print(f"    Raw data: {model}")
        else:
            print("\n⚠ No models found. You may need to pull a model first:")
            print("  ollama pull llama3.2:3b")
            print("  ollama pull phi3:mini")
            print("  ollama pull mistral:7b")
            return False
        
        # Test with the first available model
        if models.get('models'):
            try:
                # Try to get the first model name safely
                first_model = models['models'][0]
                model_name = first_model.get('name', first_model.get('model', None))
                
                if not model_name:
                    print(f"\n⚠ Could not determine model name from: {first_model}")
                    # Try to extract any string that looks like a model name
                    for key, value in first_model.items():
                        if isinstance(value, str) and ('llama' in value.lower() or 'phi' in value.lower() or 'mistral' in value.lower()):
                            model_name = value
                            print(f"  Using detected model name: {model_name}")
                            break
                
                if not model_name:
                    print("✗ Could not determine any model name to test")
                    return False
                    
                print(f"\nTesting model: {model_name}")
                
                # Test the chat method (recommended approach)
                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Say hello in one word.'
                        }
                    ],
                    options={
                        'temperature': 0.1,
                        'num_predict': 10
                    }
                )
                
                content = response.get('message', {}).get('content', '').strip()
                print(f"Chat Response: {content}")
                print(f"✓ Chat method test successful")
                
                # Test token counting if available
                try:
                    prompt_tokens = response.get('prompt_eval_count', 0)
                    output_tokens = response.get('eval_count', 0)
                    if prompt_tokens > 0 or output_tokens > 0:
                        print(f"Token usage: {prompt_tokens + output_tokens} tokens (prompt: {prompt_tokens}, output: {output_tokens})")
                except Exception:
                    pass  # Token info not always available
                
                return True
                
            except Exception as test_error:
                print(f"✗ Model test failed: {test_error}")
                print(f"  This might be a model compatibility issue")
                return False
            
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            print("\n✗ Cannot connect to Ollama. Please ensure Ollama is running:")
            print("  1. Install Ollama from https://ollama.ai")
            print("  2. Start Ollama service")
            print("  3. Verify it's running on http://localhost:11434")
        elif "not found" in error_msg:
            print(f"\n✗ Model not found: {e}")
            print("  Try pulling a model first: ollama pull llama3.2:3b")
        else:
            print(f"\n✗ Error testing Ollama: {str(e)}")
            print(f"  Full error details: {e}")
        return False

def debug_ollama_models():
    """Debug function to see the raw model data structure"""
    try:
        models = ollama.list()
        print("\n=== DEBUG: Raw Ollama Response ===")
        print(json.dumps(models, indent=2, default=str))
        print("=== END DEBUG ===")
        return models
    except Exception as e:
        print(f"Debug failed: {e}")
        return None

def test_manual_model():
    """Test a manually specified model"""
    test_models = ["llama3.2:3b", "llama3.2", "phi3:mini", "phi3", "mistral:7b", "mistral"]
    
    print("\nTrying to test common models manually:")
    for model_name in test_models:
        try:
            print(f"  Testing {model_name}...")
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Hi'}],
                options={'num_predict': 5}
            )
            print(f"  ✓ {model_name} works!")
            return model_name
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"  - {model_name}: Not installed")
            else:
                print(f"  - {model_name}: Error - {e}")
    
    return None

if __name__ == "__main__":
    print("Testing Ollama connectivity and setup...")
    print("=" * 50)
    
    # First, let's debug what we're getting from Ollama
    debug_models = debug_ollama_models()
    
    # Main connection test
    success = test_ollama_connection()
    
    if not success:
        print("\nTrying manual model testing...")
        working_model = test_manual_model()
        if working_model:
            print(f"\n✓ Found working model: {working_model}")
            success = True
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Ollama is working!")
        print("\nTo install the recommended model for your enhancement:")
        print("  ollama pull llama3.2:3b")
        print("\nYour setup is ready for LLM-powered table intelligence!")
    else:
        print("\n" + "=" * 50)
        print("❌ Ollama setup needs attention")
        print("\nQuick fix - try installing a model:")
        print("  ollama pull llama3.2:3b")
        print("  ollama pull phi3:mini")
        print("\nThen run this test again.")
        
        if debug_models:
            print(f"\nFound models data but couldn't parse it properly.")
            print("This might be a version compatibility issue.")