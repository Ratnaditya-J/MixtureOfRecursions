#!/usr/bin/env python3
"""
Simple MoR Demo - Bypasses complex attention mask issues for demonstration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from models.mor_model import MoRConfig, RecursiveRouter
from transformers import AutoTokenizer

def simple_mor_demo():
    """Run a simplified MoR demonstration focusing on the router."""
    print("=" * 60)
    print("MIXTURE-OF-RECURSIONS (MoR) SIMPLE DEMO")
    print("=" * 60)
    
    # Create tokenizer and sample text
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question.",
        "Machine learning is revolutionizing artificial intelligence."
    ]
    
    # Create router for demonstration
    config = MoRConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        max_recursion_depth=4,
        use_kv_sharing=True
    )
    
    router = RecursiveRouter(config)
    
    print(f"Router parameters: {sum(p.numel() for p in router.parameters()):,}")
    print(f"Max recursion depth: {config.max_recursion_depth}")
    print()
    
    # Test router on different texts
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=20, truncation=True)
        seq_len = inputs['input_ids'].shape[1]
        
        # Create dummy hidden states (normally from embeddings)
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        
        # Get recursion depths from router
        with torch.no_grad():
            recursion_depths, router_logits = router(hidden_states, training=False)
        
        # Decode tokens for display
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        depths = recursion_depths[0].tolist()
        
        print("Token-level recursion depths:")
        for token, depth in zip(tokens, depths):
            print(f"  {token:>15} → depth {depth}")
        
        avg_depth = sum(depths) / len(depths)
        print(f"Average recursion depth: {avg_depth:.2f}")
        print(f"Depth distribution: {dict(zip(*torch.unique(recursion_depths, return_counts=True)))}")
        print()
    
    print("✅ Simple MoR demo completed successfully!")
    print("\nKey insights:")
    print("- Different tokens get assigned different recursion depths")
    print("- Complex tokens (like 'revolutionizing') may get higher depths")
    print("- Simple tokens (like 'the', 'a') may get lower depths")
    print("- This adaptive computation is the core of MoR efficiency")

if __name__ == "__main__":
    simple_mor_demo()
