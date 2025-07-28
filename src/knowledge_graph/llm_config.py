"""
LLM Configuration Module

Provides configuration management for LLM integration in the knowledge graph system.
Supports multiple providers (Ollama, Hugging Face, LlamaCpp) with easy switching.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider"""
    provider: str
    model: str
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 300
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class LLMConfigManager:
    """
    Manages LLM configuration with support for:
    - Multiple providers
    - Environment variable overrides
    - Configuration file persistence
    - Default fallback values
    """
    
    DEFAULT_CONFIGS = {
        "ollama": LLMProviderConfig(
            provider="ollama",
            model="llama3.2:latest",
            api_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=300,
            timeout=30
        ),
        "huggingface": LLMProviderConfig(
            provider="huggingface",
            model="microsoft/DialoGPT-medium",
            temperature=0.1,
            max_tokens=300,
            timeout=30
        ),
        "llamacpp": LLMProviderConfig(
            provider="llamacpp",
            model="llama-2-7b-chat.Q4_K_M.gguf",
            temperature=0.1,
            max_tokens=300,
            timeout=30
        )
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self.configs: Dict[str, LLMProviderConfig] = {}
        self.active_provider = "ollama"  # Default provider
        
        # Load configurations
        self._load_default_configs()
        self._load_file_configs()
        self._load_env_configs()
    
    def _load_default_configs(self):
        """Load default configurations"""
        self.configs = {
            name: LLMProviderConfig(**config.to_dict())
            for name, config in self.DEFAULT_CONFIGS.items()
        }
    
    def _load_file_configs(self):
        """Load configurations from file if exists"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_configs = json.load(f)
                
                # Override with file configs
                for provider, config in file_configs.get('providers', {}).items():
                    if provider in self.configs:
                        # Update existing config
                        for key, value in config.items():
                            if hasattr(self.configs[provider], key):
                                setattr(self.configs[provider], key, value)
                    else:
                        # Add new provider config
                        self.configs[provider] = LLMProviderConfig(**config)
                
                # Set active provider from file
                self.active_provider = file_configs.get('active_provider', self.active_provider)
                
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
    
    def _load_env_configs(self):
        """Load configurations from environment variables"""
        # Environment variables override file configs
        # Format: LLM_<PROVIDER>_<SETTING>
        # Example: LLM_OLLAMA_MODEL, LLM_OLLAMA_TEMPERATURE
        
        for provider in self.configs:
            prefix = f"LLM_{provider.upper()}_"
            
            # Check for provider-specific settings
            env_vars = {
                'model': os.getenv(f"{prefix}MODEL"),
                'api_url': os.getenv(f"{prefix}API_URL"),
                'api_key': os.getenv(f"{prefix}API_KEY"),
                'temperature': os.getenv(f"{prefix}TEMPERATURE"),
                'max_tokens': os.getenv(f"{prefix}MAX_TOKENS"),
                'timeout': os.getenv(f"{prefix}TIMEOUT")
            }
            
            # Apply environment overrides
            for key, value in env_vars.items():
                if value is not None:
                    if key in ['temperature']:
                        value = float(value)
                    elif key in ['max_tokens', 'timeout']:
                        value = int(value)
                    setattr(self.configs[provider], key, value)
        
        # Check for active provider override
        active_provider_env = os.getenv("LLM_ACTIVE_PROVIDER")
        if active_provider_env and active_provider_env in self.configs:
            self.active_provider = active_provider_env
    
    def get_active_config(self) -> LLMProviderConfig:
        """Get configuration for the active provider"""
        return self.configs.get(self.active_provider, self.configs["ollama"])
    
    def get_provider_config(self, provider: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider"""
        return self.configs.get(provider)
    
    def set_active_provider(self, provider: str):
        """Set the active provider"""
        if provider not in self.configs:
            raise ValueError(f"Unknown provider: {provider}")
        self.active_provider = provider
    
    def save_to_file(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
        
        config_data = {
            "active_provider": self.active_provider,
            "providers": {
                name: config.to_dict()
                for name, config in self.configs.items()
            }
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_llm_config_for_table_intelligence(self) -> Dict[str, Any]:
        """
        Get configuration in format expected by TableIntelligenceLayer
        
        Returns:
            Dictionary with LLMConfig parameters
        """
        active_config = self.get_active_config()
        
        return {
            'provider': active_config.provider,
            'model': active_config.model,
            'temperature': active_config.temperature,
            'max_tokens': active_config.max_tokens,
            'timeout': active_config.timeout,
            'cache_enabled': True,
            'fallback_enabled': True
        }
    
    def test_provider_connection(self, provider: Optional[str] = None) -> bool:
        """
        Test connection to a provider
        
        Args:
            provider: Provider to test (uses active provider if None)
            
        Returns:
            True if connection successful
        """
        test_provider = provider or self.active_provider
        config = self.get_provider_config(test_provider)
        
        if not config:
            return False
        
        if config.provider == "ollama":
            try:
                import ollama
                models = ollama.list()
                return len(models.get('models', [])) > 0
            except:
                return False
        
        # Add tests for other providers as needed
        return False


# Example configuration file format (llm_config.json):
"""
{
  "active_provider": "ollama",
  "providers": {
    "ollama": {
      "provider": "ollama",
      "model": "llama3.2:latest",
      "api_url": "http://localhost:11434",
      "temperature": 0.1,
      "max_tokens": 300,
      "timeout": 30
    },
    "huggingface": {
      "provider": "huggingface",
      "model": "microsoft/phi-2",
      "temperature": 0.2,
      "max_tokens": 500,
      "timeout": 60
    }
  }
}
"""