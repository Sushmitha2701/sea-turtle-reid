#!/usr/bin/env python3
"""
Main Training Script for Sea Turtle Re-Identification

Usage:
    python scripts/train.py --config configs/resnet50.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_factory import create_model
from src.evaluation.metrics import evaluate, print_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description='Train wildlife re-ID model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Training Configuration: {args.config}")
    print(f"{'='*60}\n")
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print(f"Creating model: {config['model']['architecture']}")
    model = create_model(
        architecture=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        feature_dim=config['model']['feature_dim']
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Training configuration
    print("Training configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}\n")
    
    print(f"{'='*60}")
    print("Training will begin...")
    print(f"{'='*60}\n")
    print("NOTE: This is a template training script.")
    print("Please implement the full training loop based on your notebook.")
    print("Key components needed:")
    print("  1. Data loading with temporal splits")
    print("  2. Loss function (CrossEntropy + Triplet + Center)")
    print("  3. Optimizer with learning rate schedule")
    print("  4. Training loop with validation")
    print("  5. Model checkpointing")
    print("  6. TensorBoard logging")
    
    # Save config
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"\nConfiguration saved to: {config_save_path}")


if __name__ == '__main__':
    main()
