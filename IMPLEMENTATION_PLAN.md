# Mixture-of-Recursions (MoR) Implementation Plan

## Paper Summary
**Title**: Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation

**Core Innovation**: MoR combines parameter sharing through recursive layers with adaptive computation via dynamic token-level recursion depths, achieving efficiency without sacrificing performance.

## Implementation Phases

### Phase 1: Core Architecture (Weeks 1-2)
1. **Basic Recursive Transformer Block**
   - Implement shared transformer layers that can be applied recursively
   - Create token-level recursion depth routing mechanism
   - Implement selective attention computation

2. **Router Network**
   - Design lightweight routing network for depth assignment
   - Implement token-level decision making
   - Add routing loss for training stability

### Phase 2: Efficiency Optimizations (Weeks 3-4)
1. **Selective KV Caching**
   - Implement dynamic KV pair caching based on active tokens
   - Memory-efficient attention computation
   - KV sharing variant for prefill optimization

2. **Memory Management**
   - Gradient checkpointing for recursive layers
   - Efficient tensor operations for variable-depth computation

### Phase 3: Training & Evaluation (Weeks 5-6)
1. **Training Pipeline**
   - Multi-objective loss (language modeling + routing)
   - Curriculum learning for recursion depth
   - Distributed training support

2. **Evaluation Framework**
   - Perplexity evaluation
   - Few-shot learning benchmarks
   - Efficiency metrics (FLOPs, memory, throughput)

## Required Dependencies

### Core ML Libraries
```python
torch>=2.0.0
transformers>=4.30.0
torch-audio>=2.0.0
accelerate>=0.20.0
```

### Efficiency & Optimization
```python
flash-attn>=2.0.0  # For efficient attention
triton>=2.0.0      # For custom kernels
xformers>=0.0.20   # Memory-efficient transformers
```

### Training & Evaluation
```python
datasets>=2.12.0
evaluate>=0.4.0
wandb>=0.15.0      # Experiment tracking
tensorboard>=2.13.0
```

### Utilities
```python
einops>=0.6.0      # Tensor operations
hydra-core>=1.3.0  # Configuration management
rich>=13.0.0       # Pretty printing
```

## Datasets for Training/Testing

### Primary Training Datasets
1. **OpenWebText** (~40GB)
   - Open-source recreation of GPT-2's training data
   - Good for initial pretraining experiments

2. **The Pile** (800GB subset)
   - Diverse text dataset
   - Use 10-50GB subset for experiments

3. **C4 (Colossal Clean Crawled Corpus)**
   - Clean web text from Common Crawl
   - Available through HuggingFace datasets

### Evaluation Datasets
1. **Language Modeling**
   - WikiText-103
   - Penn Treebank
   - HellaSwag

2. **Few-Shot Learning**
   - GLUE benchmark tasks
   - SuperGLUE tasks
   - BIG-bench (subset)

3. **Efficiency Benchmarks**
   - Custom synthetic datasets for throughput testing
   - Variable-length sequences for memory profiling

## Model Scales for Experiments
- **Small**: 135M parameters (12 layers, 768 hidden)
- **Medium**: 350M parameters (24 layers, 1024 hidden)
- **Large**: 1.3B parameters (24 layers, 2048 hidden)

## Success Metrics
1. **Performance**: Lower perplexity than baseline at equal FLOPs
2. **Efficiency**: Higher throughput and lower memory usage
3. **Adaptivity**: Meaningful recursion depth variation across tokens
4. **Scalability**: Benefits maintained across model sizes

## Risk Mitigation
- Start with smallest model scale (135M)
- Implement comprehensive logging and debugging
- Create ablation studies for each component
- Maintain baseline comparisons throughout development
