# Contributing to Mixture-of-Recursions (MoR)

🎉 Thank you for your interest in contributing to the MoR project! This document provides guidelines for contributing to our research implementation.

## 🚀 Quick Start

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**: `python -m venv venv && source venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Install development tools**: `pip install -e .[dev]`
5. **Set up pre-commit hooks**: `pre-commit install`

## 🔬 Types of Contributions

### Research Contributions
- **New routing strategies** for adaptive computation
- **Efficiency optimizations** and hardware adaptations
- **Novel architectures** combining MoR with other techniques
- **Empirical studies** and benchmarking results

### Code Contributions
- **Bug fixes** and stability improvements
- **Performance optimizations** and memory efficiency
- **New features** and model configurations
- **Documentation** improvements and tutorials

### Community Contributions
- **Issue reports** with detailed reproduction steps
- **Feature requests** with research motivation
- **Tutorials** and educational content
- **Benchmarks** and evaluation scripts

## 📋 Development Workflow

### 1. Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/mixture-of-recursions.git
cd mixture-of-recursions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,training]

# Set up pre-commit hooks
pre-commit install
```

### 2. Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Run tests
pytest tests/ -v

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/

# Commit your changes
git add .
git commit -m "feat: add your feature description"
```

### 3. Testing Your Changes

```bash
# Run the full test suite
pytest tests/ -v --cov=src

# Test the demos work
python simple_mor_demo.py
python advanced_mor_demo.py

# Test different model configurations
python run_mor_experiment.py train --model_size small --max_epochs 1
```

### 4. Submitting Your Contribution

1. **Push to your fork**: `git push origin feature/your-feature-name`
2. **Create a Pull Request** on GitHub
3. **Fill out the PR template** with detailed information
4. **Wait for review** and address any feedback

## 🧪 Research Guidelines

### Experimental Standards
- **Reproducibility**: Include random seeds and exact configurations
- **Baselines**: Compare against relevant standard models
- **Metrics**: Report both efficiency and quality metrics
- **Statistical significance**: Use multiple runs and proper statistical tests

### Code Quality for Research
- **Modular design**: Keep experiments and core model separate
- **Configuration management**: Use config files for hyperparameters
- **Logging**: Include comprehensive logging and visualization
- **Documentation**: Document your experimental setup thoroughly

## 📊 Benchmarking Guidelines

### Performance Benchmarks
- **Hardware specifications**: Document GPU/CPU used
- **Memory usage**: Report peak memory consumption
- **Throughput**: Tokens per second for different model sizes
- **Latency**: Per-token processing time

### Quality Benchmarks
- **Standard datasets**: WikiText, GLUE, SuperGLUE
- **Perplexity**: Language modeling performance
- **Downstream tasks**: Task-specific evaluation
- **Efficiency trade-offs**: Quality vs. computation curves

## 🐛 Bug Reports

When reporting bugs, please include:

```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.10.8]
- PyTorch: [e.g., 2.1.2]
- CUDA: [e.g., 11.8]

**Model Configuration:**
- Size: [small/medium/large]
- Max recursion depth: [e.g., 3]
- Advanced features: [list enabled features]

**Reproduction Steps:**
1. Command or code used
2. Input data or configuration
3. Expected vs. actual behavior

**Error Output:**
[Full traceback]
```

## 💡 Feature Requests

For feature requests, please provide:

- **Research motivation**: Why is this feature important?
- **Use cases**: How would researchers use this?
- **Implementation ideas**: Suggestions for how to implement
- **Related work**: Links to relevant papers or implementations

## 📝 Code Style

We use automated formatting and linting:

- **Black**: Code formatting (`black src/ tests/`)
- **isort**: Import sorting (`isort src/ tests/`)
- **flake8**: Linting (`flake8 src/ tests/`)
- **Type hints**: Use type annotations where possible

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `MoRModel`, `LearnedThresholdRouter`)
- **Functions/variables**: `snake_case` (e.g., `compute_recursion_depths`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)
- **Files**: `snake_case.py` (e.g., `mor_model.py`)

## 🧪 Testing

### Test Categories
- **Unit tests**: Individual components (`test_mor_model.py`)
- **Integration tests**: Full model pipeline (`test_integration.py`)
- **Performance tests**: Efficiency and memory usage
- **Regression tests**: Prevent performance degradation

### Writing Tests
```python
def test_your_feature():
    """Test description with expected behavior."""
    # Arrange
    config = MoRConfig(hidden_size=128)
    model = MoRModel(config)
    
    # Act
    result = model.your_method()
    
    # Assert
    assert result.shape == expected_shape
    assert not torch.isnan(result).any()
```

## 📚 Documentation

### Docstring Style
```python
def compute_recursion_depths(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute recursion depths for each token using the router.
    
    Args:
        hidden_states: Input token representations [batch_size, seq_len, hidden_size]
        
    Returns:
        Recursion depths for each token [batch_size, seq_len]
        
    Example:
        >>> depths = model.compute_recursion_depths(hidden_states)
        >>> print(depths.shape)  # torch.Size([2, 10])
    """
```

### README Updates
When adding new features, update:
- Feature list in README
- Usage examples
- Configuration options
- Performance benchmarks

## 🏆 Recognition

Contributors will be recognized in:
- **README acknowledgments** for significant contributions
- **Paper co-authorship** for major research contributions
- **Release notes** for each version
- **Community highlights** in project updates

## 🤝 Code of Conduct

- **Be respectful** and inclusive in all interactions
- **Provide constructive feedback** in reviews
- **Help newcomers** get started with the project
- **Focus on research impact** and scientific rigor
- **Credit others' work** appropriately

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For research questions and ideas
- **Email**: [your-email] for sensitive matters
- **Discord/Slack**: [link] for real-time chat

## 🎯 Roadmap

Current priorities:
1. **Hardware optimization** for different GPU architectures
2. **Training stability** improvements and best practices
3. **Multimodal extensions** for vision-language models
4. **Production deployment** tools and optimizations

---

Thank you for contributing to the future of efficient transformers! 🚀
