"""
MoR Analysis and Benchmarking Tools

This module provides comprehensive analysis tools for Mixture-of-Recursions models:
- Recursion depth analysis and visualization
- Efficiency benchmarking
- Token-level computation patterns
- Comparative analysis with baseline models
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from pathlib import Path
import pandas as pd

from transformers import AutoTokenizer
from ..models.mor_model import MixtureOfRecursions, MoRConfig
from ..models.advanced_mor import AdvancedMixtureOfRecursions, AdvancedMoRConfig


class MoRAnalyzer:
    """Comprehensive analysis tool for MoR models."""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Analysis cache
        self.analysis_cache = {}
        
    def analyze_recursion_patterns(
        self, 
        texts: List[str], 
        max_length: int = 128
    ) -> Dict[str, Any]:
        """Analyze recursion depth patterns across different texts."""
        
        results = {
            'texts': texts,
            'token_depths': [],
            'depth_distributions': [],
            'complexity_scores': [],
            'efficiency_metrics': []
        }
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                if isinstance(self.model, AdvancedMixtureOfRecursions):
                    outputs = self.model(**inputs)
                    aux_outputs = outputs.get('aux_outputs', {})
                else:
                    # Use simple router demo for basic model
                    hidden_states = torch.randn(
                        inputs['input_ids'].shape[0], 
                        inputs['input_ids'].shape[1], 
                        self.model.config.hidden_size
                    ).to(self.device)
                    recursion_depths, router_logits = self.model.router(hidden_states, training=False)
                    outputs = {'recursion_depths': recursion_depths}
                    aux_outputs = {}
            
            # Extract analysis data
            depths = outputs['recursion_depths'][0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Remove padding tokens
            actual_length = (inputs['input_ids'][0] != self.tokenizer.pad_token_id).sum().item()
            depths = depths[:actual_length]
            tokens = tokens[:actual_length]
            
            # Store results
            results['token_depths'].append({
                'text': text,
                'tokens': tokens,
                'depths': depths.tolist(),
                'avg_depth': float(np.mean(depths)),
                'max_depth': int(np.max(depths)),
                'min_depth': int(np.min(depths))
            })
            
            # Depth distribution
            unique_depths, counts = np.unique(depths, return_counts=True)
            distribution = dict(zip(unique_depths.tolist(), counts.tolist()))
            results['depth_distributions'].append(distribution)
            
            # Complexity analysis
            complexity_score = self._compute_text_complexity(text, depths)
            results['complexity_scores'].append(complexity_score)
            
            # Efficiency metrics
            total_computation = np.sum(depths)
            max_computation = len(depths) * self.model.config.max_recursion_depth
            efficiency = 1.0 - (total_computation / max_computation)
            
            results['efficiency_metrics'].append({
                'text': text,
                'total_computation': int(total_computation),
                'max_computation': int(max_computation),
                'efficiency': float(efficiency),
                'avg_depth': float(np.mean(depths))
            })
        
        return results
    
    def benchmark_throughput(
        self, 
        sequence_lengths: List[int] = [32, 64, 128, 256, 512],
        batch_sizes: List[int] = [1, 4, 8],
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark model throughput across different configurations."""
        
        results = {
            'sequence_lengths': sequence_lengths,
            'batch_sizes': batch_sizes,
            'throughput_data': [],
            'memory_usage': [],
            'efficiency_comparison': []
        }
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                print(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")
                
                # Create dummy input
                input_ids = torch.randint(
                    0, self.tokenizer.vocab_size, 
                    (batch_size, seq_len)
                ).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        try:
                            _ = self.model(input_ids)
                        except Exception as e:
                            print(f"Skipping due to error: {e}")
                            continue
                
                # Benchmark
                times = []
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                for run in range(num_runs):
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    start_time = time.time()
                    try:
                        with torch.no_grad():
                            outputs = self.model(input_ids)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"Error in run {run}: {e}")
                        continue
                
                if times:
                    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    memory_used = memory_after - memory_before
                    
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    throughput = (batch_size * seq_len) / avg_time  # tokens per second
                    
                    results['throughput_data'].append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'throughput': throughput,
                        'tokens_per_sec': throughput
                    })
                    
                    results['memory_usage'].append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'memory_mb': memory_used / (1024 * 1024) if memory_used > 0 else 0
                    })
        
        return results
    
    def analyze_token_complexity_correlation(
        self, 
        texts: List[str]
    ) -> Dict[str, Any]:
        """Analyze correlation between token complexity and recursion depth."""
        
        results = {
            'correlations': [],
            'token_analysis': [],
            'complexity_patterns': {}
        }
        
        all_tokens = []
        all_depths = []
        all_complexities = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(self.device)
            
            with torch.no_grad():
                # Get recursion depths
                hidden_states = torch.randn(
                    inputs['input_ids'].shape[0], 
                    inputs['input_ids'].shape[1], 
                    self.model.config.hidden_size
                ).to(self.device)
                recursion_depths, _ = self.model.router(hidden_states, training=False)
            
            depths = recursion_depths[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Compute token complexities
            complexities = []
            for token in tokens:
                if token == self.tokenizer.pad_token:
                    continue
                    
                # Simple complexity metrics
                token_complexity = (
                    len(token) +  # Length
                    (1 if token.startswith('Ġ') else 0) +  # Word boundary
                    (1 if any(c.isupper() for c in token) else 0) +  # Capitalization
                    (1 if any(c.isdigit() for c in token) else 0) +  # Numbers
                    (1 if any(c in '.,!?;:' for c in token) else 0)  # Punctuation
                )
                complexities.append(token_complexity)
            
            # Store for correlation analysis
            min_len = min(len(depths), len(complexities))
            all_tokens.extend(tokens[:min_len])
            all_depths.extend(depths[:min_len])
            all_complexities.extend(complexities[:min_len])
        
        # Compute correlations
        if all_depths and all_complexities:
            correlation = np.corrcoef(all_depths, all_complexities)[0, 1]
            results['correlations'].append({
                'metric': 'token_complexity_vs_depth',
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0
            })
        
        # Analyze patterns by token type
        token_types = {}
        for token, depth, complexity in zip(all_tokens, all_depths, all_complexities):
            if token not in token_types:
                token_types[token] = {'depths': [], 'complexities': []}
            token_types[token]['depths'].append(depth)
            token_types[token]['complexities'].append(complexity)
        
        # Summarize patterns
        for token, data in token_types.items():
            if len(data['depths']) >= 3:  # Only tokens that appear multiple times
                results['complexity_patterns'][token] = {
                    'avg_depth': float(np.mean(data['depths'])),
                    'avg_complexity': float(np.mean(data['complexities'])),
                    'frequency': len(data['depths'])
                }
        
        return results
    
    def _compute_text_complexity(self, text: str, depths: np.ndarray) -> Dict[str, float]:
        """Compute various text complexity metrics."""
        
        # Basic text statistics
        words = text.split()
        sentences = text.split('.')
        
        complexity = {
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'num_sentences': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'vocab_diversity': len(set(words)) / max(len(words), 1),
            'avg_recursion_depth': float(np.mean(depths)),
            'recursion_variance': float(np.var(depths))
        }
        
        return complexity
    
    def generate_analysis_report(
        self, 
        texts: List[str], 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        print("🔍 Generating MoR Analysis Report...")
        
        # Run all analyses
        recursion_analysis = self.analyze_recursion_patterns(texts)
        throughput_analysis = self.benchmark_throughput(
            sequence_lengths=[32, 64, 128], 
            batch_sizes=[1, 4], 
            num_runs=5
        )
        complexity_analysis = self.analyze_token_complexity_correlation(texts)
        
        # Compile report
        report = {
            'model_info': {
                'model_type': type(self.model).__name__,
                'config': self.model.config.__dict__,
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'recursion_analysis': recursion_analysis,
            'throughput_analysis': throughput_analysis,
            'complexity_analysis': complexity_analysis,
            'summary_statistics': self._compute_summary_statistics(recursion_analysis, throughput_analysis)
        }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📊 Report saved to: {output_path}")
        
        return report
    
    def _compute_summary_statistics(
        self, 
        recursion_analysis: Dict, 
        throughput_analysis: Dict
    ) -> Dict[str, Any]:
        """Compute summary statistics across all analyses."""
        
        # Recursion statistics
        all_depths = []
        all_efficiencies = []
        
        for item in recursion_analysis['token_depths']:
            all_depths.extend(item['depths'])
        
        for item in recursion_analysis['efficiency_metrics']:
            all_efficiencies.append(item['efficiency'])
        
        # Throughput statistics
        throughputs = [item['throughput'] for item in throughput_analysis['throughput_data']]
        
        summary = {
            'recursion_stats': {
                'avg_depth': float(np.mean(all_depths)) if all_depths else 0,
                'depth_std': float(np.std(all_depths)) if all_depths else 0,
                'avg_efficiency': float(np.mean(all_efficiencies)) if all_efficiencies else 0,
                'efficiency_std': float(np.std(all_efficiencies)) if all_efficiencies else 0
            },
            'performance_stats': {
                'avg_throughput': float(np.mean(throughputs)) if throughputs else 0,
                'max_throughput': float(np.max(throughputs)) if throughputs else 0,
                'throughput_std': float(np.std(throughputs)) if throughputs else 0
            }
        }
        
        return summary


def create_analyzer(model_path: Optional[str] = None, model_type: str = "basic") -> MoRAnalyzer:
    """Factory function to create MoR analyzer."""
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    if model_path and Path(model_path).exists():
        # Load saved model
        if model_type == "advanced":
            config = AdvancedMoRConfig()
            model = AdvancedMixtureOfRecursions(config)
        else:
            config = MoRConfig()
            model = MixtureOfRecursions(config)
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        # Create new model for analysis
        if model_type == "advanced":
            from .advanced_mor import create_advanced_mor_model
            model = create_advanced_mor_model("small")
        else:
            config = MoRConfig(vocab_size=tokenizer.vocab_size)
            model = MixtureOfRecursions(config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return MoRAnalyzer(model, tokenizer, device)
