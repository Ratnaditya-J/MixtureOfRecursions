"""
Training script for Mixture-of-Recursions (MoR) model.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mor_model import MixtureOfRecursions, MoRConfig
from utils.config import Config


class MoRTrainer:
    """Trainer class for MoR model."""

    def __init__(self, config: MoRConfig, model_config: dict):
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = MixtureOfRecursions(config).to(self.device)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
        )

        # Initialize scheduler
        self.scheduler = None

    def prepare_data(self, dataset_name: str, max_length: int = 512):
        """Prepare training data."""
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
            train_dataset = dataset["train"]
        elif dataset_name == "openwebtext":
            dataset = load_dataset(
                "openwebtext", split="train[:1%]"
            )  # Use small subset
            train_dataset = dataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        # Tokenize dataset
        tokenized_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=train_dataset.column_names
        )

        return tokenized_dataset

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_router_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Create labels (shifted input_ids for language modeling)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding tokens

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            router_loss = outputs["router_loss"]

            # Compute language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            lm_loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Total loss combines language modeling and router losses
            total_loss_batch = lm_loss + router_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Update metrics
            total_loss += total_loss_batch.item()
            total_lm_loss += lm_loss.item()
            total_router_loss += router_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "LM Loss": f"{lm_loss.item():.4f}",
                    "Router Loss": f"{router_loss.item():.4f}",
                    "Total Loss": f"{total_loss_batch.item():.4f}",
                }
            )

            # Log to wandb
            if batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train/lm_loss": lm_loss.item(),
                        "train/router_loss": router_loss.item(),
                        "train/total_loss": total_loss_batch.item(),
                        "train/avg_recursion_depth": outputs["recursion_depths"]
                        .float()
                        .mean()
                        .item(),
                    }
                )

        return {
            "total_loss": total_loss / len(dataloader),
            "lm_loss": total_lm_loss / len(dataloader),
            "router_loss": total_router_loss / len(dataloader),
        }

    def evaluate(self, dataloader: DataLoader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_lm_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]

                # Compute perplexity
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                lm_loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                total_lm_loss += lm_loss.item()

        avg_loss = total_lm_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))

        return {"eval_loss": avg_loss, "perplexity": perplexity.item()}

    def save_model(self, save_path: str):
        """Save model checkpoint."""
        os.makedirs(save_path, exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            os.path.join(save_path, "model.pt"),
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description="Train MoR model")
    parser.add_argument(
        "--dataset", default="wikitext", choices=["wikitext", "openwebtext"]
    )
    parser.add_argument(
        "--model_size", default="small", choices=["small", "medium", "large"]
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_dir", default="./results/mor_checkpoints")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Model configurations
    model_configs = {
        "small": MoRConfig(
            vocab_size=50257,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            max_recursion_depth=3,
        ),
        "medium": MoRConfig(
            vocab_size=50257,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=24,
            max_recursion_depth=4,
        ),
        "large": MoRConfig(
            vocab_size=50257,
            hidden_size=1536,
            num_attention_heads=24,
            num_hidden_layers=36,
            max_recursion_depth=5,
        ),
    }

    config = model_configs[args.model_size]

    # Training configuration
    train_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="mor-training",
            config={**config.__dict__, **train_config, **vars(args)},
        )

    # Initialize trainer
    trainer = MoRTrainer(config, train_config)

    # Prepare data
    print("Preparing training data...")
    train_dataset = trainer.prepare_data(args.dataset, args.max_length)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Setup scheduler
    total_steps = len(train_dataloader) * args.epochs
    trainer.scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_dataloader, epoch)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  LM Loss: {train_metrics['lm_loss']:.4f}")
        print(f"  Router Loss: {train_metrics['router_loss']:.4f}")

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics["total_loss"],
                    "train/epoch_lm_loss": train_metrics["lm_loss"],
                    "train/epoch_router_loss": train_metrics["router_loss"],
                }
            )

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            trainer.save_model(save_path)
            print(f"Model saved to {save_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
