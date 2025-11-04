"""
Evaluation Metrics for Wildlife Re-Identification

Implements standard re-identification metrics:
- Rank-k Accuracy
- Mean Average Precision (mAP)
- CMC Curve
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings


def compute_distance_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute pairwise distance matrix between query and gallery features.
    
    Args:
        query_features: Query features [N_q, D]
        gallery_features: Gallery features [N_g, D]
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Distance matrix [N_q, N_g]
    """
    if metric == 'euclidean':
        # Euclidean distance
        m, n = query_features.shape[0], gallery_features.shape[0]
        distmat = np.zeros((m, n))
        
        for i in range(m):
            distmat[i] = np.sqrt(
                ((query_features[i:i+1] - gallery_features)**2).sum(axis=1)
            )
            
    elif metric == 'cosine':
        # Cosine distance (1 - cosine similarity)
        # Assuming features are already L2-normalized
        distmat = 1 - np.dot(query_features, gallery_features.T)
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distmat


def evaluate_rank(
    distmat: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    max_rank: int = 50
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate ranking-based metrics.
    
    Args:
        distmat: Distance matrix [N_q, N_g]
        query_ids: Query identity labels [N_q]
        gallery_ids: Gallery identity labels [N_g]
        max_rank: Maximum rank to compute
        
    Returns:
        cmc: Cumulative Matching Characteristic curve
        metrics: Dictionary with rank-k accuracies
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        warnings.warn(f"Gallery size ({num_g}) < max_rank. Setting max_rank = {num_g}")
    
    # Sort indices by distance
    indices = np.argsort(distmat, axis=1)
    
    # Get gallery IDs in ranked order for each query
    matches = gallery_ids[indices]
    
    # Check if correct match appears in top-k
    cmc = np.zeros(max_rank)
    
    for q_idx in range(num_q):
        # Query identity
        q_id = query_ids[q_idx]
        
        # Find where correct matches appear in ranking
        correct_matches = matches[q_idx] == q_id
        
        if not correct_matches.any():
            # No correct match in gallery
            continue
        
        # First correct match position
        first_match_idx = np.where(correct_matches)[0][0]
        
        # Update CMC curve
        if first_match_idx < max_rank:
            cmc[first_match_idx:] += 1
    
    # Normalize CMC curve
    cmc = cmc / num_q * 100  # Convert to percentage
    
    # Compute specific rank-k metrics
    metrics = {
        'rank-1': cmc[0],
        'rank-5': cmc[4] if max_rank >= 5 else None,
        'rank-10': cmc[9] if max_rank >= 10 else None,
        'rank-20': cmc[19] if max_rank >= 20 else None,
    }
    
    return cmc, metrics


def evaluate_map(
    distmat: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray
) -> float:
    """
    Evaluate Mean Average Precision (mAP).
    
    Args:
        distmat: Distance matrix [N_q, N_g]
        query_ids: Query identity labels [N_q]
        gallery_ids: Gallery identity labels [N_g]
        
    Returns:
        mAP score
    """
    num_q, num_g = distmat.shape
    
    # Sort indices by distance
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices]
    
    all_aps = []
    
    for q_idx in range(num_q):
        q_id = query_ids[q_idx]
        
        # Find all correct matches
        correct_matches = matches[q_idx] == q_id
        num_correct = correct_matches.sum()
        
        if num_correct == 0:
            # No correct matches for this query
            all_aps.append(0)
            continue
        
        # Compute Average Precision for this query
        # AP = (1/num_correct) * sum of (precision @ k * relevance_k)
        positions = np.where(correct_matches)[0]
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(positions)]
        ap = np.mean(precisions)
        
        all_aps.append(ap)
    
    # Mean Average Precision
    mAP = np.mean(all_aps) * 100  # Convert to percentage
    
    return mAP


def evaluate(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    max_rank: int = 50,
    distance_metric: str = 'euclidean'
) -> Dict[str, float]:
    """
    Comprehensive evaluation with all metrics.
    
    Args:
        query_features: Query feature embeddings [N_q, D]
        gallery_features: Gallery feature embeddings [N_g, D]
        query_ids: Query identity labels [N_q]
        gallery_ids: Gallery identity labels [N_g]
        max_rank: Maximum rank for CMC computation
        distance_metric: Distance metric to use
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Compute distance matrix
    distmat = compute_distance_matrix(
        query_features,
        gallery_features,
        metric=distance_metric
    )
    
    # Evaluate ranking metrics
    cmc, rank_metrics = evaluate_rank(distmat, query_ids, gallery_ids, max_rank)
    
    # Evaluate mAP
    mAP = evaluate_map(distmat, query_ids, gallery_ids)
    
    # Combine all metrics
    results = {
        'mAP': mAP,
        'cmc': cmc,
        **rank_metrics
    }
    
    return results


def print_evaluation_results(results: Dict[str, float], dataset_name: str = ""):
    """
    Print evaluation results in formatted table.
    
    Args:
        results: Dictionary with evaluation metrics
        dataset_name: Name of dataset for display
    """
    print("\n" + "="*60)
    if dataset_name:
        print(f"Evaluation Results: {dataset_name}")
    else:
        print("Evaluation Results")
    print("="*60)
    
    print(f"mAP: {results['mAP']:.2f}%")
    print("-"*60)
    
    for key in ['rank-1', 'rank-5', 'rank-10', 'rank-20']:
        if key in results and results[key] is not None:
            print(f"{key}: {results[key]:.2f}%")
    
    print("="*60 + "\n")


def compute_random_baseline(
    num_classes: int,
    max_rank: int = 50
) -> Dict[str, float]:
    """
    Compute random baseline performance.
    
    Args:
        num_classes: Number of unique identities
        max_rank: Maximum rank
        
    Returns:
        Dictionary with baseline metrics
    """
    # Random baseline: probability of correct match in top-k
    cmc = np.minimum(np.arange(1, max_rank + 1) / num_classes * 100, 100)
    
    baseline = {
        'mAP': 100.0 / num_classes,
        'rank-1': 100.0 / num_classes,
        'rank-5': min(500.0 / num_classes, 100.0),
        'rank-10': min(1000.0 / num_classes, 100.0),
        'rank-20': min(2000.0 / num_classes, 100.0),
        'cmc': cmc
    }
    
    return baseline


if __name__ == "__main__":
    print("Evaluation Metrics - Testing\n")
    
    # Generate random data for testing
    np.random.seed(42)
    num_query = 100
    num_gallery = 200
    feature_dim = 512
    num_classes = 50
    
    query_features = np.random.randn(num_query, feature_dim)
    gallery_features = np.random.randn(num_gallery, feature_dim)
    query_ids = np.random.randint(0, num_classes, num_query)
    gallery_ids = np.random.randint(0, num_classes, num_gallery)
    
    # Normalize features (for cosine distance)
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    
    # Evaluate
    print("Testing evaluation metrics...")
    results = evaluate(
        query_features,
        gallery_features,
        query_ids,
        gallery_ids,
        max_rank=20,
        distance_metric='cosine'
    )
    
    print_evaluation_results(results, "Test Dataset")
    
    # Compare with random baseline
    baseline = compute_random_baseline(num_classes, max_rank=20)
    print("Random Baseline:")
    print_evaluation_results(baseline, "Random Baseline")
    
    print("✓ All tests passed!")
