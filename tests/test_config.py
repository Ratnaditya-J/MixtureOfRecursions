"""Tests for configuration utilities."""

import pytest
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_project_paths_exist(self):
        """Test that project paths are properly defined."""
        assert Config.PROJECT_ROOT.exists()
        assert Config.DATA_DIR.exists()
        assert Config.RESULTS_DIR.exists()
        assert Config.MODELS_DIR.exists()
    
    def test_default_values(self):
        """Test that default configuration values are set."""
        assert Config.DEFAULT_MODEL is not None
        assert Config.MAX_TOKENS > 0
        assert 0 <= Config.TEMPERATURE <= 2
    
    def test_get_api_key(self):
        """Test API key retrieval method."""
        # Should not raise an error even if keys are None
        openai_key = Config.get_api_key("openai")
        anthropic_key = Config.get_api_key("anthropic")
        hf_key = Config.get_api_key("huggingface")
        
        # Should return None for unknown providers
        unknown_key = Config.get_api_key("unknown_provider")
        assert unknown_key is None
