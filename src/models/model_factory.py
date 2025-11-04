"""
Model Factory for Wildlife Re-Identification

Provides unified interface for loading different architectures:
- ResNet-18, ResNet-50
- OSNet variants
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings


class ReIDModel(nn.Module):
    """Base class for re-identification models."""
    
    def __init__(self, num_classes: int, feature_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
    def forward(self, x):
        raise NotImplementedError
        
    def extract_features(self, x):
        """Extract feature embeddings without classification."""
        raise NotImplementedError


def create_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    feature_dim: int = 512,
    **kwargs
) -> ReIDModel:
    """
    Create a re-identification model.
    
    Args:
        architecture: Model architecture name ('resnet18', 'resnet50', 'osnet')
        num_classes: Number of individual identities
        pretrained: Whether to use ImageNet pretrained weights
        feature_dim: Dimension of feature embeddings
        **kwargs: Additional architecture-specific arguments
        
    Returns:
        Initialized model
    """
    architecture = architecture.lower()
    
    if architecture == 'resnet18':
        from .resnet import ResNet18ReID
        model = ResNet18ReID(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            **kwargs
        )
        
    elif architecture == 'resnet50':
        from .resnet import ResNet50ReID
        model = ResNet50ReID(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            **kwargs
        )
        
    elif architecture == 'osnet':
        from .osnet import OSNetReID
        model = OSNetReID(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_dim=feature_dim,
            **kwargs
        )
        
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: 'resnet18', 'resnet50', 'osnet'"
        )
    
    return model


def load_model(
    architecture: str,
    checkpoint: str,
    num_classes: Optional[int] = None,
    device: str = 'cuda',
    **kwargs
) -> ReIDModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        architecture: Model architecture name
        checkpoint: Path to checkpoint file
        num_classes: Number of classes (auto-detected from checkpoint if None)
        device: Device to load model on
        **kwargs: Additional model arguments
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint, map_location=device)
    
    # Auto-detect num_classes if not provided
    if num_classes is None:
        if 'num_classes' in checkpoint_data:
            num_classes = checkpoint_data['num_classes']
        elif 'model_config' in checkpoint_data:
            num_classes = checkpoint_data['model_config']['num_classes']
        else:
            # Try to infer from state dict
            state_dict = checkpoint_data.get('state_dict', checkpoint_data)
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
            
            if num_classes is None:
                raise ValueError(
                    "Could not auto-detect num_classes. Please provide explicitly."
                )
    
    # Create model
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,  # Don't load ImageNet weights
        **kwargs
    )
    
    # Load weights
    if 'state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['state_dict'])
    else:
        model.load_state_dict(checkpoint_data)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint}")
    print(f"  Architecture: {architecture}")
    print(f"  Num classes: {num_classes}")
    print(f"  Device: {device}")
    
    return model


def get_model_info(architecture: str) -> Dict[str, Any]:
    """
    Get information about a model architecture.
    
    Args:
        architecture: Model architecture name
        
    Returns:
        Dictionary with model information
    """
    info = {
        'resnet18': {
            'parameters': '11.4M',
            'gflops': 1.8,
            'feature_dim': 512,
            'backbone_dim': 512,
            'typical_training_time': '29 min',
            'memory_usage': '8.4GB',
            'best_for': 'Efficient deployment, good balance',
        },
        'resnet50': {
            'parameters': '24.7M',
            'gflops': 4.1,
            'feature_dim': 512,
            'backbone_dim': 2048,
            'typical_training_time': '47 min',
            'memory_usage': '14.1GB',
            'best_for': 'Maximum accuracy, detailed features',
        },
        'osnet': {
            'parameters': '2.2M',
            'gflops': 0.98,
            'feature_dim': 512,
            'backbone_dim': 512,
            'typical_training_time': '21 min',
            'memory_usage': '6.2GB',
            'best_for': 'Resource-constrained, edge deployment',
        }
    }
    
    architecture = architecture.lower()
    if architecture not in info:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return info[architecture]


def print_model_comparison():
    """Print comparison table of available architectures."""
    architectures = ['resnet18', 'resnet50', 'osnet']
    
    print("\n" + "="*80)
    print("Model Architecture Comparison")
    print("="*80)
    print(f"{'Architecture':<12} {'Parameters':<12} {'GFLOPs':<10} {'Training Time':<15} {'Memory':<10}")
    print("-"*80)
    
    for arch in architectures:
        info = get_model_info(arch)
        print(f"{arch:<12} {info['parameters']:<12} {info['gflops']:<10} "
              f"{info['typical_training_time']:<15} {info['memory_usage']:<10}")
    
    print("="*80)
    print("\nRecommendations:")
    for arch in architectures:
        info = get_model_info(arch)
        print(f"  {arch.upper()}: {info['best_for']}")
    print()


if __name__ == "__main__":
    print("Model Factory - Usage Examples\n")
    
    # Print comparison
    print_model_comparison()
    
    # Create example models
    print("\nCreating example models:")
    for arch in ['resnet18', 'resnet50', 'osnet']:
        try:
            model = create_model(arch, num_classes=438, pretrained=False)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  {arch}: {total_params:,} parameters")
        except Exception as e:
            print(f"  {arch}: Error - {e}")
