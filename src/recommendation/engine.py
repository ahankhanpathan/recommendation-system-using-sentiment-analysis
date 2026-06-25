"""
Central inference layer that turns raw dataframe rows into fused feature vectors,
exposes similarity / category recommendation helpers, and can be dropped into
Streamlit or any other serving stack.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401

from src.recommendation.feature_engineer import FeatureEngineer
from src.recommendation.graph_builder import GraphBuilder
from src.recommendation.models import HybridRecommender, GNNModel, SimilarityScorer
from src.recommendation.transformer_embedder import TransformerEmbedder

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """High-level orchestrator around the trained HybridRecommender stack."""

    def __init__(
        self,
        hybrid_model: HybridRecommender,
        gnn_model: GNNModel,
        feature_engineer: FeatureEngineer,
        embedder: TransformerEmbedder,
        graph_builder: GraphBuilder,
        *,
        fused_feature_dim: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.hybrid = hybrid_model.to(self.device).eval()
        self.gnn = gnn_model.to(self.device).eval()

        self.fe = feature_engineer
        self.embedder = embedder
        self.graph = graph_builder

        self.similarity_scorer = SimilarityScorer(fused_feature_dim).to(self.device).eval()

        logger.info("RecommendationEngine ready on %s", self.device)

    # ---------------------------------------------------------------------- #
    # INTERNAL UTILITIES                                                       #
    # ---------------------------------------------------------------------- #

    @torch.inference_mode()
    def _build_gnn_embeddings(
        self, text_emb: np.ndarray, cat_feats: np.ndarray
    ) -> torch.Tensor:
        """Run the frozen GNN over a freshly-built similarity graph."""
        edge_index, _ = self.graph.build_product_graph(text_emb, cat_feats)
        x = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)
        return self.gnn(x, edge_index)  # (N, gnn_dim)

    @torch.inference_mode()
    def _fuse(
        self,
        text_emb: np.ndarray,
        gnn_emb: torch.Tensor,
        category: np.ndarray,
        numeric: np.ndarray,
        aspects: np.ndarray,
    ) -> torch.Tensor:
        """Pass all feature types through the frozen HybridRecommender fusion layer."""
        x_text = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
        x_cat = torch.tensor(category, dtype=torch.float32, device=self.device)
        x_num = torch.tensor(numeric, dtype=torch.float32, device=self.device)
        x_asp = torch.tensor(aspects, dtype=torch.float32, device=self.device)

        _pred, _attn, fused = self.hybrid(x_text, gnn_emb, x_cat, x_num, x_asp)
        return fused  # (N, fused_feature_dim)

    # ---------------------------------------------------------------------- #
    # PUBLIC API                                                               #
    # ---------------------------------------------------------------------- #

    def get_product_representations(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turn a dataframe slice into fused feature vectors (shape: N × fusion_dim).

        Pipeline:
        1. Sentence-transformer embeddings for the `all_text` column.
        2. Classical one-hot / numeric / aspect vectors via FeatureEngineer.
        3. Build similarity graph → GNN for relational signal.
        4. HybridRecommender attention fusion layer.
        """
        logger.info("Generating text embeddings for %d products …", len(df))
        text_emb = self.embedder.generate_embeddings(df["all_text"].tolist())  # (N, 384)

        feats = self.fe.transform_features(df)
        cat_np = feats["category_features"].values.astype("float32")
        num_np = feats["numerical_features"].values.astype("float32")
        asp_np = feats["aspects_matrix"].astype("float32")

        logger.info("Building graph and running GNN pass …")
        gnn_emb = self._build_gnn_embeddings(text_emb, cat_np)  # (N, 64)

        logger.info("Fusing features …")
        fused = self._fuse(text_emb, gnn_emb, cat_np, num_np, asp_np)

        return fused.cpu().numpy()  # (N, fused_feature_dim)

    def recommend_similar_products(
        self, product_idx: int, df: pd.DataFrame, *, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Nearest-neighbour recommendations within the same category.

        Returns a dataframe with original columns plus a `similarity_score` column.
        """
        logger.info("Generating similar-item recommendations for row %d", product_idx)

        cat = df.loc[product_idx, "main_category"]
        all_fused = self.get_product_representations(df)  # (N, F)
        query = np.repeat(all_fused[[product_idx]], len(df), axis=0)

        with torch.inference_mode():
            scores = (
                self.similarity_scorer(
                    torch.tensor(query, device=self.device),
                    torch.tensor(all_fused, device=self.device),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        mask = (df["main_category"] == cat).to_numpy()
        scores[~mask] = -1.0
        scores[product_idx] = -1.0

        idx = np.argsort(-scores)[:top_n]
        return (
            df.loc[idx]
            .assign(similarity_score=scores[idx])
            .reset_index(drop=True)
        )

    def recommend_by_category(
        self, category: str, df: pd.DataFrame, *, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Top-N items in a given category ranked by centroid-similarity × sentiment score.
        """
        logger.info("Generating category recommendations for '%s'", category)
        subset = df[df["main_category"].str.lower() == category.lower()].copy()
        if subset.empty:
            logger.warning("No products found for category '%s'", category)
            return pd.DataFrame()

        reps = self.get_product_representations(subset)
        centroid = reps.mean(axis=0, keepdims=True)

        with torch.inference_mode():
            sims = (
                self.similarity_scorer(
                    torch.tensor(np.repeat(centroid, len(subset), axis=0), device=self.device),
                    torch.tensor(reps, device=self.device),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        weights = subset.get("score", pd.Series(np.ones(len(subset)))).to_numpy(float)
        final = sims * weights
        subset = subset.assign(weighted_score=final, similarity_score=sims)

        return (
            subset.sort_values("weighted_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


# --------------------------------------------------------------------------- #
# Convenience factory                                                          #
# --------------------------------------------------------------------------- #

def build_engine_from_trained_folder(
    model_dir: str,
    *,
    fused_feature_dim: int = 128,
) -> RecommendationEngine:
    """
    Recreate the full inference stack from a saved model directory.

    The directory must contain:
    ``hybrid_model.pt``, ``gnn_model.pt``, ``feature_engineer.pkl``,
    ``mlb.pkl``, and ``config.json``.
    """
    import json
    from pathlib import Path

    import dill as pickle

    path = Path(model_dir)
    with open(path / "config.json") as f:
        cfg = json.load(f)

    with (path / "feature_engineer.pkl").open("rb") as f:
        fe: FeatureEngineer = pickle.load(f)
    with (path / "mlb.pkl").open("rb") as f:
        fe.mlb = pickle.load(f)

    hybrid = HybridRecommender(
        cfg,
        embedding_dim=384,
        gnn_dim=cfg["gnn"]["output_dim"],
        category_dim=int(fe.category_columns.size),
        numerical_dim=3,
        aspects_dim=int(fe.mlb.classes_.size),
    )
    hybrid.load_state_dict(torch.load(path / "hybrid_model.pt", map_location="cpu"))
    hybrid.eval()

    gnn = GNNModel(cfg["gnn"], input_dim=384)
    gnn.load_state_dict(torch.load(path / "gnn_model.pt", map_location="cpu"))
    gnn.eval()

    embedder = TransformerEmbedder(cfg["transformer"])
    graph = GraphBuilder(cfg["gnn"])

    return RecommendationEngine(
        hybrid,
        gnn,
        fe,
        embedder,
        graph,
        fused_feature_dim=fused_feature_dim,
    )
