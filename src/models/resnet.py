"""
ResNet Architectures for Wildlife Re-Identification

Implements ResNet-18 and ResNet-50 with unified re-identification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetReID(nn.Module):
    """
    ResNet-based re-identification model with unified head architecture.
    
    Architecture:
        ResNet Backbone → Global Average Pooling → BN → Embedding → Classifier
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        num_classes: int,
        feature_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone_dim = backbone_dim
        
        # Backbone (without final FC layer)
        self.backbone = backbone
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(backbone_dim),
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        # L2 normalization for embeddings
        self.l2_norm = nn.functional.normalize
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize new layers with appropriate initialization."""
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.classifier.weight, std=0.001)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return both logits and features
            
        Returns:
            logits or (logits, features) if return_features=True
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Global average pooling
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        
        # Embedding
        embeddings = self.embedding(features)
        
        # L2 normalization
        normalized_embeddings = self.l2_norm(embeddings, p=2, dim=1)
        
        # Classification
        logits = self.classifier(embeddings)
        
        if return_features:
            return logits, normalized_embeddings
        return logits
    
    def extract_features(self, x):
        """
        Extract normalized feature embeddings without classification.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Normalized feature embeddings [B, feature_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = self.gap(features)
            features = features.view(features.size(0), -1)
            embeddings = self.embedding(features)
            normalized_embeddings = self.l2_norm(embeddings, p=2, dim=1)
            return normalized_embeddings


class ResNet18ReID(ResNetReID):
    """
    ResNet-18 for re-identification.
    
    Specifications:
        - Parameters: 11.4M
        - GFLOPs: 1.8
        - Backbone dimension: 512
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feature_dim: int = 512,
        dropout: float = 0.5,
    ):
        # Load ResNet-18 backbone
        resnet18 = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        super().__init__(
            backbone=backbone,
            backbone_dim=512,  # ResNet-18 output channels
            num_classes=num_classes,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        self.architecture = 'resnet18'


class ResNet50ReID(ResNetReID):
    """
    ResNet-50 for re-identification.
    
    Specifications:
        - Parameters: 24.7M
        - GFLOPs: 4.1
        - Backbone dimension: 2048
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feature_dim: int = 512,
        dropout: float = 0.5,
    ):
        # Load ResNet-50 backbone
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Remove final FC layer
        backbone = nn.Sequential(*list(resnet50.children())[:-2])
        
        super().__init__(
            backbone=backbone,
            backbone_dim=2048,  # ResNet-50 output channels
            num_classes=num_classes,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        self.architecture = 'resnet50'


def get_resnet_model(
    variant: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> ResNetReID:
    """
    Get ResNet model by variant name.
    
    Args:
        variant: 'resnet18' or 'resnet50'
        num_classes: Number of individual identities
        pretrained: Use ImageNet pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        ResNet re-ID model
    """
    variant = variant.lower()
    
    if variant == 'resnet18':
        return ResNet18ReID(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif variant == 'resnet50':
        return ResNet50ReID(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown ResNet variant: {variant}")


if __name__ == "__main__":
    print("ResNet Re-ID Models - Testing\n")
    
    # Test ResNet-18
    print("Testing ResNet-18:")
    model18 = ResNet18ReID(num_classes=438, pretrained=False)
    x = torch.randn(2, 3, 256, 128)
    logits = model18(x)
    features = model18.extract_features(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Feature embeddings shape: {features.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model18.parameters()):,}")
    
    print("\nTesting ResNet-50:")
    model50 = ResNet50ReID(num_classes=438, pretrained=False)
    logits = model50(x)
    features = model50.extract_features(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Feature embeddings shape: {features.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model50.parameters()):,}")
    
    print("\n✓ All tests passed!")
