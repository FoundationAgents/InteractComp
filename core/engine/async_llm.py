# LLM Engine, Logger Engine, And Search Engine

# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang & didi
# @Desc    : 

from openai import AsyncOpenAI

import yaml
from pathlib import Path
from typing import Dict, Optional, Any

from core.engine.logs import logger, LogLevel

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")
        self.top_p = config.get("top_p", 1)

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            config_paths = [
                Path("config/global_config.yaml"),
                Path("config/global_config2.yaml"),
                Path("./config/global_config.yaml"),
                Path("config/infra_config.yaml")
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),  # Map api_key to key
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1)  # Add top_p parameter
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        # openai: https://platform.openai.com/docs/pricing
        # anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01}, 
        "o3": {"input": 0.002, "output": 0.008},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
        "gpt-5": {"input": 0.00125, "output": 0.01},
        "gpt-5-mini": {"input":0.00025, "output": 0.002},
        "moonshotai/kimi-k2": {"input": 0.000296, "output": 0.001185}, 
        "deepseek/deepseek-chat-v3.1": {"input":0.0002 , "output":0.0008},
        "z-ai/glm-4.5": {"input": 0.00033, "output": 0.00132},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
        "claude-4-sonnet": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
        "deepseek-chat": {"input":0.002 , "output":0.008},
        "deepseek-reasoner": {"input":0.004 , "output":0.016},
        "qwen3-235b-a22b": {"input":0.00286 , "output":0.01144},
        "x-ai/grok-4": {"input":0.00315 , "output":0.01575},
    }


    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # Return default pricing if no match found
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

class AsyncLLM:
    def __init__(self, config, system_msg:str = None, max_completion_tokens:int = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
            max_tokens: Optional maximum number of tokens to generate
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        self.max_completion_tokens = max_completion_tokens
    
    def _build_model_params(self, tokens_to_use):
        """Build model-specific parameters based on model capabilities"""
        params = {
            "temperature": self.config.temperature,
        }
        
        # Add token limit parameter based on model support
        if tokens_to_use is not None:
            if "o3" in self.config.model or "gpt-5" in self.config.model:
                # o3 and gpt-5 only support max_completion_tokens
                params["max_completion_tokens"] = tokens_to_use
            else:
                # Other models use max_tokens
                params["max_tokens"] = tokens_to_use
        
        # Add top_p only for models that support it
        if "o3" not in self.config.model and "gpt-5" not in self.config.model:
            params["top_p"] = self.config.top_p
        
        return params
        
    async def __call__(self, prompt, max_tokens=None):
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})

        logger.log_to_file(LogLevel.INFO, f"LLM Prompt: \n{message}")

        # Prefer to use the max_tokens argument passed to the function; if it is None, use the instance variable.
        tokens_to_use = max_tokens if max_tokens is not None else self.max_completion_tokens

        # Build model-specific parameters
        model_params = self._build_model_params(tokens_to_use)

        response = await self.aclient.chat.completions.create(
            model=self.config.model,
            messages=message,
            **model_params
        )

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # Track token usage and calculate cost
        self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )
        
        ret = response.choices[0].message.content
        logger.log_to_file(LogLevel.INFO, f"LLM Response: {ret}")
        
        return ret
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()    
    

def create_llm_instance(llm_config) -> AsyncLLM:
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
    

# 