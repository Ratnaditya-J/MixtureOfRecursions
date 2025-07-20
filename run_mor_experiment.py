#!/usr/bin/env python3
"""
Complete Mixture-of-Recursions (MoR) Experiment Runner

This script provides a unified interface to run MoR experiments including:
- Model training
- Evaluation
- Benchmarking
- Analysis
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from experiments.train_mor import main as train_main
from experiments.evaluate_mor import main as eval_main


def setup_experiment_directory(experiment_name: str):
    """Set up experiment directory structure."""
    base_dir = Path("./results") / experiment_name
    
    # Create directories
    directories = [
        base_dir / "checkpoints",
        base_dir / "logs",
        base_dir / "evaluations",
        base_dir / "plots"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return base_dir


def run_quick_demo():
    """Run a quick demonstration of MoR model."""
    print("=" * 60)
    print("MIXTURE-OF-RECURSIONS (MoR) QUICK DEMO")
    print("=" * 60)
    
    from models.mor_model import MixtureOfRecursions, MoRConfig
    import torch
    from transformers import AutoTokenizer
    
    # Create small model for demo
    config = MoRConfig(
        vocab_size=1000,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        max_recursion_depth=3,
        use_kv_sharing=True
    )
    
    model = MixtureOfRecursions(config)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Test input
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt", max_length=20, truncation=True)
    
    print(f"Input: {test_text}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs['logits'].shape}")
    print(f"Recursion depths: {outputs['recursion_depths'][0].tolist()}")
    print(f"Router loss: {outputs['router_loss'].item():.4f}")
    print(f"Average recursion depth: {outputs['recursion_depths'].float().mean().item():.2f}")
    
    print("\n✅ Demo completed successfully!")
    print("Next steps:")
    print("  1. Run training: python run_mor_experiment.py train --model_size small")
    print("  2. Run evaluation: python run_mor_experiment.py evaluate --model_path results/checkpoints/")
    print("  3. Open Jupyter notebook: jupyter notebook notebooks/02_mixture_of_recursions_demo.ipynb")


def main():
    parser = argparse.ArgumentParser(description='MoR Experiment Runner')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demo')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train MoR model')
    train_parser.add_argument('--experiment_name', default='mor_experiment')
    train_parser.add_argument('--model_size', default='small', choices=['small', 'medium', 'large'])
    train_parser.add_argument('--dataset', default='wikitext', choices=['wikitext', 'openwebtext'])
    train_parser.add_argument('--epochs', type=int, default=3)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--use_wandb', action='store_true')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate MoR model')
    eval_parser.add_argument('--model_path', required=True)
    eval_parser.add_argument('--eval_type', default='comprehensive')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup experiment environment')
    setup_parser.add_argument('--experiment_name', required=True)
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_quick_demo()
        
    elif args.command == 'setup':
        exp_dir = setup_experiment_directory(args.experiment_name)
        print(f"Experiment directory created: {exp_dir}")
        
    elif args.command == 'train':
        # Setup experiment directory
        exp_dir = setup_experiment_directory(args.experiment_name)
        
        # Prepare training arguments
        train_args = [
            '--dataset', args.dataset,
            '--model_size', args.model_size,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--save_dir', str(exp_dir / 'checkpoints')
        ]
        
        if args.use_wandb:
            train_args.append('--use_wandb')
        
        # Run training
        sys.argv = ['train_mor.py'] + train_args
        train_main()
        
    elif args.command == 'evaluate':
        # Prepare evaluation arguments
        eval_args = [
            '--model_path', args.model_path,
            '--eval_type', args.eval_type,
            '--output_path', './results/evaluation_report.json'
        ]
        
        # Run evaluation
        sys.argv = ['evaluate_mor.py'] + eval_args
        eval_main()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
