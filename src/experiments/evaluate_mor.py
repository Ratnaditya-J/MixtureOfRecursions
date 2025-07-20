"""
Evaluation script for Mixture-of-Recursions (MoR) model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mor_model import MixtureOfRecursions, MoRConfig


class MoREvaluator:
    """Evaluator class for MoR model."""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=self.device)
        self.config = checkpoint['config']
        self.model = MixtureOfRecursions(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def evaluate_perplexity(self, dataset_name: str, split: str = "test", max_samples: int = 1000):
        """Evaluate perplexity on a dataset."""
        # Load dataset
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")[split]
        elif dataset_name == "ptb":
            dataset = load_dataset("ptb_text_only")[split]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples for faster evaluation
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=False)
        
        total_loss = 0
        total_tokens = 0
        recursion_depths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating perplexity"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Create labels
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                depths = outputs["recursion_depths"]
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = nn.CrossEntropyLoss(reduction='sum')(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Count valid tokens
                valid_tokens = (shift_labels != -100).sum().item()
                
                total_loss += loss.item()
                total_tokens += valid_tokens
                
                # Collect recursion depth statistics
                valid_depths = depths[attention_mask == 1]
                recursion_depths.extend(valid_depths.cpu().numpy().tolist())
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        # Recursion depth statistics
        depth_stats = {
            'mean_depth': np.mean(recursion_depths),
            'std_depth': np.std(recursion_depths),
            'min_depth': np.min(recursion_depths),
            'max_depth': np.max(recursion_depths),
            'depth_distribution': {
                str(i): (np.array(recursion_depths) == i).sum() / len(recursion_depths)
                for i in range(self.config.min_recursion_depth, self.config.max_recursion_depth + 1)
            }
        }
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'recursion_stats': depth_stats
        }
    
    def evaluate_throughput(self, sequence_lengths: list = [128, 256, 512, 1024]):
        """Evaluate model throughput at different sequence lengths."""
        throughput_results = {}
        
        for seq_len in sequence_lengths:
            if seq_len > self.config.max_position_embeddings:
                continue
                
            # Create dummy input
            batch_size = 8
            input_ids = torch.randint(
                0, self.config.vocab_size, 
                (batch_size, seq_len), 
                device=self.device
            )
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(input_ids)
            
            # Measure throughput
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            num_runs = 20
            with torch.no_grad():
                for _ in range(num_runs):
                    outputs = self.model(input_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_runs
            tokens_per_second = (batch_size * seq_len) / avg_time_per_batch
            
            # Get average recursion depth for this sequence length
            avg_depth = outputs["recursion_depths"].float().mean().item()
            
            throughput_results[seq_len] = {
                'avg_time_per_batch': avg_time_per_batch,
                'tokens_per_second': tokens_per_second,
                'avg_recursion_depth': avg_depth
            }
            
            print(f"Seq len {seq_len}: {tokens_per_second:.1f} tokens/sec, avg depth: {avg_depth:.2f}")
        
        return throughput_results
    
    def evaluate_few_shot(self, task: str = "sentiment", num_shots: int = 5):
        """Evaluate few-shot learning performance."""
        # This is a simplified example - in practice you'd use proper few-shot datasets
        if task == "sentiment":
            # Create simple sentiment classification examples
            examples = [
                ("This movie is amazing!", "positive"),
                ("I hate this film.", "negative"),
                ("Great acting and plot.", "positive"),
                ("Boring and predictable.", "negative"),
                ("Absolutely wonderful!", "positive")
            ]
            
            test_cases = [
                "This is a fantastic movie!",
                "Terrible acting and story.",
                "I enjoyed watching this.",
                "Not worth my time."
            ]
            
            correct = 0
            total = len(test_cases)
            
            for test_text in test_cases:
                # Create few-shot prompt
                prompt = ""
                for text, label in examples[:num_shots]:
                    prompt += f"Text: {text}\nSentiment: {label}\n\n"
                prompt += f"Text: {test_text}\nSentiment:"
                
                # Tokenize and generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs["logits"][0, -1, :]  # Last token logits
                    
                    # Get tokens for "positive" and "negative"
                    pos_token = self.tokenizer.encode(" positive")[0]
                    neg_token = self.tokenizer.encode(" negative")[0]
                    
                    pos_prob = torch.softmax(logits, dim=-1)[pos_token].item()
                    neg_prob = torch.softmax(logits, dim=-1)[neg_token].item()
                    
                    prediction = "positive" if pos_prob > neg_prob else "negative"
                    
                    # Simple ground truth (this would be more sophisticated in practice)
                    ground_truth = "positive" if any(word in test_text.lower() 
                                                   for word in ["fantastic", "enjoyed", "great", "amazing"]) else "negative"
                    
                    if prediction == ground_truth:
                        correct += 1
            
            accuracy = correct / total
            return {"few_shot_accuracy": accuracy, "task": task, "num_shots": num_shots}
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def generate_comprehensive_report(self, output_path: str):
        """Generate a comprehensive evaluation report."""
        print("Starting comprehensive evaluation...")
        
        results = {
            "model_config": self.config.__dict__,
            "evaluation_results": {}
        }
        
        # Perplexity evaluation
        print("Evaluating perplexity on WikiText-103...")
        perplexity_results = self.evaluate_perplexity("wikitext", "test")
        results["evaluation_results"]["perplexity"] = perplexity_results
        
        # Throughput evaluation
        print("Evaluating throughput...")
        throughput_results = self.evaluate_throughput()
        results["evaluation_results"]["throughput"] = throughput_results
        
        # Few-shot evaluation
        print("Evaluating few-shot performance...")
        few_shot_results = self.evaluate_few_shot("sentiment", 5)
        results["evaluation_results"]["few_shot"] = few_shot_results
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Perplexity: {perplexity_results['perplexity']:.2f}")
        print(f"Average recursion depth: {perplexity_results['recursion_stats']['mean_depth']:.2f}")
        print(f"Few-shot accuracy: {few_shot_results['few_shot_accuracy']:.3f}")
        print(f"Peak throughput: {max(t['tokens_per_second'] for t in throughput_results.values()):.1f} tokens/sec")
        print(f"Results saved to: {output_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MoR model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--output_path', default='./results/evaluation_report.json')
    parser.add_argument('--eval_type', default='comprehensive', 
                       choices=['perplexity', 'throughput', 'few_shot', 'comprehensive'])
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MoREvaluator(args.model_path)
    
    if args.eval_type == 'comprehensive':
        evaluator.generate_comprehensive_report(args.output_path)
    elif args.eval_type == 'perplexity':
        results = evaluator.evaluate_perplexity("wikitext", "test")
        print(f"Perplexity: {results['perplexity']:.2f}")
    elif args.eval_type == 'throughput':
        results = evaluator.evaluate_throughput()
        for seq_len, metrics in results.items():
            print(f"Seq {seq_len}: {metrics['tokens_per_second']:.1f} tok/s")
    elif args.eval_type == 'few_shot':
        results = evaluator.evaluate_few_shot()
        print(f"Few-shot accuracy: {results['few_shot_accuracy']:.3f}")


if __name__ == "__main__":
    main()
