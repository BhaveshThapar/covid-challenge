"""
Domain Adaptation utilities for cross-source generalization.

Includes:
  - Gradient Reversal Layer (GRL) for adversarial training
  - Domain Discriminator for learning domain-invariant features
  - Unsupervised domain discovery via k-means clustering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, ICML 2015).
    During forward pass, acts as identity.
    During backward pass, negates gradients scaled by lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for the GRL function."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """
    Predicts which latent domain a feature vector belongs to.
    Used with GRL for adversarial domain adaptation.
    """

    def __init__(self, embed_dim, hidden_dim=256, n_domains=4, dropout=0.3):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_domains),
        )

    def set_lambda(self, lambda_):
        """Set the GRL lambda for scheduling."""
        self.grl.set_lambda(lambda_)

    def forward(self, features):
        """
        Args:
            features: (B, embed_dim) scan-level embeddings
        Returns:
            domain_logits: (B, n_domains)
        """
        reversed_features = self.grl(features)
        return self.net(reversed_features)


def compute_domain_lambda(epoch, max_epochs, gamma=10.0):
    """
    Progressive domain loss weighting schedule.
    Starts at 0, increases to 1 following the schedule from Ganin et al.
    lambda = 2 / (1 + exp(-gamma * p)) - 1, where p = epoch / max_epochs
    """
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def discover_domains_kmeans(features, n_clusters=4):
    """
    Discover latent domains in training data via k-means clustering.

    Args:
        features: (N, D) numpy array of scan-level embeddings
        n_clusters: number of domain clusters to discover

    Returns:
        labels: (N,) cluster assignments
        centroids: (n_clusters, D) cluster centroids
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, max_iter=300)
    labels = kmeans.fit_predict(features_scaled)

    # Transform centroids back
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, centroids
