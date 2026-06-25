"""Neural network models for the hybrid recommendation system."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

logger = logging.getLogger(__name__)


class SimilarityScorer(nn.Module):
    """MLP that scores similarity between two fused product vectors, output in [0, 1]."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([x1, x2], dim=1)
        return self.model(pair)


class AttentionFusion(nn.Module):
    """
    Fuses multiple feature vectors via a Q-K-V attention mechanism.

    Each feature type is first projected to `fusion_dim`, then cross-feature
    attention weights are computed and used to produce a single fused vector.
    """

    def __init__(self, input_dims: list, fusion_dim: int) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.num_features = len(input_dims)

        self.projection_layers = nn.ModuleList(
            [nn.Linear(dim, fusion_dim) for dim in input_dims]
        )
        self.query = nn.Linear(fusion_dim, fusion_dim)
        self.key = nn.Linear(fusion_dim, fusion_dim)
        self.value = nn.Linear(fusion_dim, fusion_dim)
        self.scale = fusion_dim ** 0.5
        self.output_layer = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, features: list) -> tuple:
        """
        Args:
            features: List of tensors, each (batch_size, feature_dim_i).

        Returns:
            fused_output: (batch_size, fusion_dim)
            attention_weights: (batch_size, num_features, num_features)
        """
        if not isinstance(features, list) or len(features) != self.num_features:
            raise ValueError(
                f"AttentionFusion expected a list of {self.num_features} tensors, "
                f"got {type(features)} of length "
                f"{len(features) if isinstance(features, list) else 'N/A'}"
            )

        projected = [layer(feat) for layer, feat in zip(self.projection_layers, features)]
        stacked = torch.stack(projected, dim=1)  # (B, num_features, fusion_dim)

        queries = self.query(stacked)
        keys = self.key(stacked)
        values = self.value(stacked)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        weighted_sum = torch.matmul(attention_weights, values)  # (B, num_features, fusion_dim)
        fused_output = weighted_sum.sum(dim=1)                  # (B, fusion_dim)
        fused_output = self.output_layer(fused_output)

        return fused_output, attention_weights


class HybridRecommender(nn.Module):
    """
    Combines five feature types (text, GNN, category, numerical, aspects) via
    attention fusion and outputs a binary interaction prediction.
    """

    def __init__(
        self,
        config: dict,
        embedding_dim: int,
        gnn_dim: int,
        category_dim: int,
        numerical_dim: int,
        aspects_dim: int,
    ) -> None:
        super().__init__()

        fusion_dim: int = config["hybrid"]["fusion_output_dim"]
        hidden_dim: int = config["hybrid"].get("hidden_dim", fusion_dim)
        dropout: float = config["gnn"].get("dropout", 0.2)

        # Per-feature projection layers
        self.embedding_layer = nn.Linear(embedding_dim, fusion_dim)
        self.gnn_layer = nn.Linear(gnn_dim, fusion_dim)
        self.category_layer = nn.Linear(category_dim, fusion_dim)
        self.numerical_layer = nn.Linear(numerical_dim, fusion_dim)
        self.aspects_layer = nn.Linear(aspects_dim, fusion_dim)

        # Attention fusion over the five projected features
        self.fusion_layer = AttentionFusion([fusion_dim] * 5, fusion_dim)

        # Binary classifier on top of the fused vector
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("HybridRecommender initialised — %d trainable parameters", total)

    def forward(
        self,
        embedding_feat: torch.Tensor,
        gnn_feat: torch.Tensor,
        category_feat: torch.Tensor,
        numerical_feat: torch.Tensor,
        aspects_feat: torch.Tensor,
    ) -> tuple:
        """
        Returns:
            prediction: (batch_size, 1) sigmoid output
            attention_weights: (batch_size, num_features, num_features)
            fused_output: (batch_size, fusion_dim)
        """
        embedded = self.embedding_layer(embedding_feat)
        gnn_out = self.gnn_layer(gnn_feat)
        category_out = self.category_layer(category_feat)
        numerical_out = self.numerical_layer(numerical_feat)
        aspects_out = self.aspects_layer(aspects_feat)

        features = [embedded, gnn_out, category_out, numerical_out, aspects_out]
        fused_output, attention_weights = self.fusion_layer(features)

        prediction = self.predictor(fused_output)
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)

        return prediction, attention_weights, fused_output

    def get_feature_importance(self) -> dict:
        """Run a dummy forward pass and return average attention weights per feature."""
        feature_names = ["text", "graph", "category", "numerical", "aspects"]
        dims = [
            self.embedding_layer.in_features,
            self.gnn_layer.in_features,
            self.category_layer.in_features,
            self.numerical_layer.in_features,
            self.aspects_layer.in_features,
        ]
        with torch.no_grad():
            dummy_feats = [torch.zeros(1, d) for d in dims]
            _, weights, _ = self.forward(*dummy_feats)
            avg = weights.mean(dim=0).mean(dim=0)  # (num_features,)
        return {name: float(w) for name, w in zip(feature_names, avg)}


class GNNModel(nn.Module):
    """
    Graph Neural Network (GCN or GAT) that encodes product relationship signals.

    Takes text embeddings as node features, runs message passing over the
    product similarity graph, and outputs per-node embeddings.
    """

    def __init__(self, config: dict, input_dim: int) -> None:
        super().__init__()
        self.gnn_type = config.get("type", "gcn")
        hidden_channels: int = config.get("hidden_channels", 128)
        num_layers: int = config.get("num_layers", 2)
        output_dim: int = config.get("output_dim", 64)
        dropout: float = config.get("dropout", 0.2)

        self.layers = nn.ModuleList()

        if self.gnn_type == "gcn":
            self.layers.append(GCNConv(input_dim, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, output_dim))
        elif self.gnn_type == "gat":
            self.layers.append(GATConv(input_dim, hidden_channels, heads=4, dropout=dropout))
            for _ in range(num_layers - 2):
                self.layers.append(
                    GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
                )
            self.layers.append(
                GATConv(hidden_channels * 4, output_dim, heads=1, concat=False, dropout=dropout)
            )
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type!r}. Use 'gcn' or 'gat'.")

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x
