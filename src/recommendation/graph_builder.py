"""Product similarity graph builder for use with PyTorch Geometric GNNs."""

import logging

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a sparse k-NN product similarity graph from text embeddings.

    Optionally restricts edges to products in the same category so the GNN
    learns within-category relational signals only.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.similarity_threshold: float = config.get("similarity_threshold", 0.3)
        self.k: int = config.get("k_neighbors", 10)

    def build_product_graph(
        self,
        embeddings: np.ndarray,
        category_features: np.ndarray | None = None,
        k: int | None = None,
    ) -> tuple:
        """
        Build a product similarity graph.

        Args:
            embeddings: (N, D) array of product embeddings.
            category_features: Optional (N, C) one-hot category matrix.
                               When provided, edges are restricted to same-category pairs.
            k: Number of nearest neighbours per node (overrides config value if given).

        Returns:
            edge_index: (2, E) LongTensor of directed edges.
            edge_attr: (E,) FloatTensor of similarity weights.
        """
        k = k or self.k
        similarity = cosine_similarity(embeddings)  # (N, N)

        if category_features is not None:
            cat_id = category_features.argmax(axis=1)         # (N,)
            same_cat = cat_id[:, None] == cat_id[None, :]     # (N, N) bool
            similarity = similarity * same_cat.astype(float)

        similarity_sparse = self._top_k_sparse(similarity, k)

        rows, cols = similarity_sparse.nonzero()
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(similarity_sparse[rows, cols], dtype=torch.float)

        logger.info(
            "Graph built: %d nodes, %d edges (k=%d, threshold=%.2f)",
            embeddings.shape[0], edge_index.shape[1], k, self.similarity_threshold,
        )
        return edge_index, edge_attr

    def _top_k_sparse(self, similarity_matrix: np.ndarray, k: int) -> np.ndarray:
        """Return a sparse matrix keeping only the top-k neighbours per row."""
        n = similarity_matrix.shape[0]
        sparse = np.zeros_like(similarity_matrix)

        for i in range(n):
            top_idx = np.argsort(-similarity_matrix[i])[:k + 1]  # +1 includes self
            for j in top_idx:
                if i != j and similarity_matrix[i, j] > self.similarity_threshold:
                    sparse[i, j] = similarity_matrix[i, j]

        return sparse
