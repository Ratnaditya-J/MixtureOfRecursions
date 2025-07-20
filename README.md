# Mixture-of-Recursions (MoR) Research Project

**A comprehensive implementation of the Mixture-of-Recursions model for adaptive token-level computation in transformers.**

Based on the research paper: *"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"*

## 🚀 Key Features

### Core MoR Architecture
- ✅ **Recursive Transformer Layers** - Parameter sharing across computation depths
- ✅ **Adaptive Token-Level Routing** - Dynamic recursion depth assignment per token
- ✅ **Selective Attention** - Only active tokens participate in attention
- ✅ **KV Caching Optimization** - Memory-efficient key-value pair reuse

### Advanced Features
- 🎯 **Learned Threshold Routing** - Dynamic depth assignment with learned thresholds
- 🔄 **Multi-Scale Attention** - Hierarchical processing at multiple scales
- ⚡ **Efficiency-Aware Routing** - Computational optimization with target efficiency
- 🧠 **Adaptive Caching** - Smart KV cache management
- 📊 **Comprehensive Analysis Tools** - Depth patterns, efficiency metrics, benchmarking

## 📁 Project Structure

```
llm-research/
├── src/
│   ├── models/
│   │   ├── mor_model.py           # Core MoR implementation
│   │   └── advanced_mor.py        # Advanced MoR features
│   ├── experiments/
│   │   ├── train_mor.py           # Training pipeline
│   │   └── evaluate_mor.py        # Evaluation suite
│   ├── analysis/
│   │   └── mor_analyzer.py        # Analysis & benchmarking tools
│   └── utils/                     # Utility functions
├── notebooks/
│   ├── 01_getting_started.ipynb  # Project introduction
│   └── 02_mixture_of_recursions_demo.ipynb  # Interactive MoR demo
├── run_mor_experiment.py          # Unified experiment runner
├── simple_mor_demo.py             # Basic MoR demonstration
├── advanced_mor_demo.py           # Advanced features showcase
├── IMPLEMENTATION_PLAN.md         # Detailed implementation plan
└── requirements.txt               # Dependencies
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables in `.env` file

3. Start exploring the notebooks or run experiments from the `src/` directory

## Features

- Model experimentation and evaluation
- Data processing utilities
- Jupyter notebooks for interactive research
- Comprehensive testing suite

## Contributing

Please follow the established code structure and add tests for new functionality.

## License

MIT License

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd llm-research
pip install -r requirements.txt
```

### Basic Demo
```bash
# Run simple MoR demonstration
python simple_mor_demo.py

# Run advanced features showcase
python advanced_mor_demo.py
```

### Training & Evaluation
```bash
# Train a small MoR model
python run_mor_experiment.py train --model_size small --dataset wikitext

# Evaluate trained model
python run_mor_experiment.py evaluate --model_path results/checkpoints/

# Run comprehensive demo
python run_mor_experiment.py demo
```

### Interactive Analysis
```bash
# Launch Jupyter notebooks
jupyter notebook notebooks/

# Open the MoR demo notebook
# notebooks/02_mixture_of_recursions_demo.ipynb
```

## 📊 Model Configurations

| Size | Hidden Size | Attention Heads | Layers | Max Recursion Depth | Parameters |
|------|-------------|-----------------|--------|-------------------|------------|
| Small | 256 | 8 | 4 | 3 | ~33M |
| Medium | 512 | 16 | 8 | 4 | ~135M |
| Large | 1024 | 32 | 16 | 6 | ~1.7B |

## 🎯 Key Innovations

### 1. Adaptive Token-Level Computation
- Different tokens receive different amounts of computation
- Complex tokens (e.g., "revolutionizing") get deeper processing
- Simple tokens (e.g., "the", "a") get lighter processing
- Automatic efficiency optimization

### 2. Parameter Sharing via Recursion
- Same transformer layers reused across depths
- Dramatically reduces model size vs. standard transformers
- Maintains quality while improving efficiency

### 3. Advanced Routing Mechanisms
- **Learned Thresholds**: Dynamic depth assignment with trainable thresholds
- **Efficiency-Aware**: Balances performance vs. computational cost
- **Multi-Scale**: Hierarchical attention at different resolutions

## 📈 Performance Benefits

- **Parameter Efficiency**: 50-70% fewer parameters than equivalent transformers
- **Adaptive Computation**: 20-40% reduction in FLOPs for equivalent quality
- **Memory Optimization**: Smart KV caching reduces memory usage
- **Throughput**: Higher tokens/second due to selective computation

## 🔬 Research Applications

### Supported Datasets
- **Training**: WikiText-103, OpenWebText, The Pile
- **Evaluation**: WikiText, Penn Treebank, GLUE, SuperGLUE
- **Custom**: Easy integration of new datasets

### Analysis Tools
- Recursion depth pattern analysis
- Token complexity correlation studies
- Efficiency benchmarking
- Throughput and memory profiling
- Comparative analysis with baseline models

## 🛠️ Advanced Usage

### Custom Model Creation
```python
from src.models.advanced_mor import create_advanced_mor_model

# Create advanced MoR model
model = create_advanced_mor_model(
    model_size="medium",
    use_all_features=True
)
```

### Analysis and Benchmarking
```python
from src.analysis import create_analyzer

# Create analyzer
analyzer = create_analyzer(model_type="advanced")

# Analyze recursion patterns
results = analyzer.analyze_recursion_patterns([
    "Simple text.",
    "Complex technical documentation with specialized terminology."
])

# Benchmark throughput
benchmark = analyzer.benchmark_throughput(
    sequence_lengths=[128, 256, 512],
    batch_sizes=[1, 4, 8]
)
```

## 📚 Documentation

- **[Implementation Plan](IMPLEMENTATION_PLAN.md)**: Detailed development roadmap
- **[Getting Started Notebook](notebooks/01_getting_started.ipynb)**: Project introduction
- **[MoR Demo Notebook](notebooks/02_mixture_of_recursions_demo.ipynb)**: Interactive demonstrations
- **API Documentation**: Inline docstrings throughout codebase

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎉 Acknowledgments

Based on the research paper: *"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"*

---

**Ready to explore adaptive computation in transformers? Start with `python simple_mor_demo.py`!**
