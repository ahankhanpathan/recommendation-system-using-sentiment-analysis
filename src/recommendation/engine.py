# engine.py
from __future__ import annotations
# ----------------------------------------------------------------------------------
# Central “glue” layer that turns raw dataframe rows into fused feature vectors,
# exposes similarity / category recommendation helpers, and can be dropped into
# Streamlit or any other serving stack.
# ----------------------------------------------------------------------------------


import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity  # noqa

# --- Local project imports --------------------------------------------------------

from src.recommendation.feature_engineer import FeatureEngineer
from src.recommendation.models import HybridRecommender, GNNModel, SimilarityScorer

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

        # Core models
        self.hybrid = hybrid_model.to(self.device).eval()
        self.gnn = gnn_model.to(self.device).eval()

        # Tooling
        self.fe = feature_engineer
        self.embedder = embedder
        self.graph = graph_builder

        # Lightweight MLP to turn two fused vectors into a similarity score [0-1]
        self.scorer = SimilarityScorer(fused_feature_dim).to(self.device).eval()

        logger.info("RecommendationEngine ready on %s", self.device)

    # ------------------------------------------------------------------------- #
    # INTERNAL UTILITIES                                                         #
    # ------------------------------------------------------------------------- #
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
        """Pass everything through the (frozen) HybridRecommender and get fused vecs."""
        x_text = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
        x_cat = torch.tensor(category, dtype=torch.float32, device=self.device)
        x_num = torch.tensor(numeric, dtype=torch.float32, device=self.device)
        x_asp = torch.tensor(aspects, dtype=torch.float32, device=self.device)

        pred, attn, fused = self.hybrid(
            x_text,
            gnn_emb,
            x_cat,
            x_num,
            x_asp,
        )
        # We only need the fused representation here
        return fused  # (N, fused_feature_dim)

    # ------------------------------------------------------------------------- #
    # PUBLIC API                                                                 #
    # ------------------------------------------------------------------------- #
    def get_product_representations(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turn a dataframe slice into **fused feature vectors** (shape: N × 128).

        The pipeline:
        1. Sentence-transformer embeddings for `all_text`
        2. Classical one-hots / numeric / aspect vectors via FeatureEngineer
        3. Build similarity graph ➜ GNN for relational signal
        4. HybridRecommender fusion layer
        """
        logger.info("Generating text embeddings …")
        text_emb = self.embedder.generate_embeddings(df["all_text"])  # (N, 384)

        feats = self.fe.transform_features(df)
        cat_np = feats["category_features"].values.astype("float32")  # one-hot
        num_np = feats["numerical_features"].values.astype("float32")
        asp_np = feats["aspects_matrix"].astype("float32")

        logger.info("Building graph + GNN pass …")
        gnn_emb = self._build_gnn_embeddings(text_emb, cat_np)  # (N, 64)

        logger.info("Fusing features …")
        fused = self._fuse(text_emb, gnn_emb, cat_np, num_np, asp_np)  # torch.Tensor

        return fused.cpu().numpy()  # (N, fused_feature_dim)

    # --------------------------------------------------------------------- #
    # Recommendation helpers                                                #
    # --------------------------------------------------------------------- #
    def recommend_similar_products(
        self, product_idx: int, df: pd.DataFrame, *, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Nearest-neighbour style recommendations **within the same category**.

        Returns a dataframe with original product columns + `similarity_score`.
        """
        logger.info("Similar-items for row %d", product_idx)

        cat = df.loc[product_idx, "main_category"]
        all_fused = self.get_product_representations(df)  # (N, F)
        query = all_fused[[product_idx]].repeat(len(df), axis=0)

        with torch.inference_mode():
            scores = (
                self.scorer(
                    torch.tensor(query, device=self.device),
                    torch.tensor(all_fused, device=self.device),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # same-category mask & drop self
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
        Pick the **top‐N** items in a given category by similarity-to-centroid × score.
        """
        logger.info("Category recommendations for %s", category)
        subset = df[df["main_category"].str.lower() == category.lower()].copy()
        if subset.empty:
            logger.warning("No rows for category %s", category)
            return pd.DataFrame()

        reps = self.get_product_representations(subset)
        centroid = reps.mean(axis=0, keepdims=True)

        with torch.inference_mode():
            sims = (
                self.scorer(
                    torch.tensor(np.repeat(centroid, len(subset), axis=0), device=self.device),
                    torch.tensor(reps, device=self.device),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # combine similarity with original sentiment score if available
        weights = subset.get("score", pd.Series(np.ones(len(subset)))).to_numpy(float)
        final = sims * weights
        subset = subset.assign(weighted_score=final, similarity_score=sims)

        return subset.sort_values("weighted_score", ascending=False).head(top_n).reset_index(
            drop=True
        )


# ---------------------------------------------------------------------------- #
# Simple convenience factory if you need to wire everything from a single call #
# ---------------------------------------------------------------------------- #
def build_engine_from_trained_folder(
    model_dir: str,
    config: dict,
    *,
    fused_feature_dim: int = 128,
) -> RecommendationEngine:
    """
    Utility to recreate the full stack from the saved `model_dir`.

    Assumes:
    - `model_dir` has  *hybrid_model.pt* , *gnn_model.pt* ,
      *feature_engineer.pkl* , *mlb.pkl* , and *config.json*
    """
    import dill as pickle
    from pathlib import Path
    import json

    path = Path(model_dir)
    with open(path / "config.json") as f:
        cfg = json.load(f)

    # ----- feature engineer ---------------------------------------------------
    with (path / "feature_engineer.pkl").open("rb") as f:
        fe: FeatureEngineer = pickle.load(f)
    with (path / "mlb.pkl").open("rb") as f:
        fe.mlb = pickle.load(f)

    # ----- models -------------------------------------------------------------
    hybrid = HybridRecommender(
        cfg,
        embedding_dim=384,
        gnn_dim=64,
        category_dim=fe.category_columns.size,
        numerical_dim=3,
        aspects_dim=fe.mlb.classes_.size,
    )
    hybrid.load_state_dict(torch.load(path / "hybrid_model.pt", map_location="cpu"))
    hybrid.eval()

    gnn = GNNModel(cfg["gnn"], input_dim=384)
    gnn.load_state_dict(torch.load(path / "gnn_model.pt", map_location="cpu"))
    gnn.eval()

    # ----- helpers ------------------------------------------------------------
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
# engine.py
# ----------------------------------------------------------------------------------
# Central “glue” layer that turns raw dataframe rows into fused feature vectors,
# exposes similarity / category recommendation helpers, and can be dropped into
# Streamlit or any other serving stack.
# ----------------------------------------------------------------------------------


import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity  # noqa

# --- Local project imports --------------------------------------------------------

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

        # Core models
        self.hybrid = hybrid_model.to(self.device).eval()
        self.gnn = gnn_model.to(self.device).eval()

        # Tooling
        self.fe = feature_engineer
        self.embedder = embedder
        self.graph = graph_builder

        # Lightweight MLP to turn two fused vectors into a similarity score [0-1]
        self.scorer = SimilarityScorer(fused_feature_dim).to(self.device).eval()

        logger.info("RecommendationEngine ready on %s", self.device)

    # ------------------------------------------------------------------------- #
    # INTERNAL UTILITIES                                                         #
    # ------------------------------------------------------------------------- #
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
        """Pass everything through the (frozen) HybridRecommender and get fused vecs."""
        x_text = torch.tensor(text_emb, dtype=torch.float32, device=self.device)
        x_cat = torch.tensor(category, dtype=torch.float32, device=self.device)
        x_num = torch.tensor(numeric, dtype=torch.float32, device=self.device)
        x_asp = torch.tensor(aspects, dtype=torch.float32, device=self.device)

        pred, attn, fused = self.hybrid(
            x_text,
            gnn_emb,
            x_cat,
            x_num,
            x_asp,
        )
        # We only need the fused representation here
        return fused  # (N, fused_feature_dim)

    # ------------------------------------------------------------------------- #
    # PUBLIC API                                                                 #
    # ------------------------------------------------------------------------- #
    def get_product_representations(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turn a dataframe slice into **fused feature vectors** (shape: N × 128).

        The pipeline:
        1. Sentence-transformer embeddings for `all_text`
        2. Classical one-hots / numeric / aspect vectors via FeatureEngineer
        3. Build similarity graph ➜ GNN for relational signal
        4. HybridRecommender fusion layer
        """
        logger.info("Generating text embeddings …")
        text_emb = self.embedder.generate_embeddings(df["all_text"])  # (N, 384)

        feats = self.fe.transform_features(df)
        cat_np = feats["category_features"].values.astype("float32")  # one-hot
        num_np = feats["numerical_features"].values.astype("float32")
        asp_np = feats["aspects_matrix"].astype("float32")

        logger.info("Building graph + GNN pass …")
        gnn_emb = self._build_gnn_embeddings(text_emb, cat_np)  # (N, 64)

        logger.info("Fusing features …")
        fused = self._fuse(text_emb, gnn_emb, cat_np, num_np, asp_np)  # torch.Tensor

        return fused.cpu().numpy()  # (N, fused_feature_dim)

    # --------------------------------------------------------------------- #
    # Recommendation helpers                                                #
    # --------------------------------------------------------------------- #
    def recommend_similar_products(
        self, product_idx: int, df: pd.DataFrame, *, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Nearest-neighbour style recommendations **within the same category**.

        Returns a dataframe with original product columns + `similarity_score`.
        """
        logger.info("Similar-items for row %d", product_idx)

        cat = df.loc[product_idx, "main_category"]
        all_fused = self.get_product_representations(df)  # (N, F)
        query = all_fused[[product_idx]].repeat(len(df), axis=0)

        with torch.inference_mode():
            scores = (
                self.scorer(
                    torch.tensor(query, device=self.device),
                    torch.tensor(all_fused, device=self.device),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # same-category mask & drop self
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
        Pick the **top‐N** items in a given category by similarity-to-centroid × score.
        """
        logger.info("Category recommendations for %s", category)
        subset = df[df["main_category"].str.lower() == category.lower()].copy()
        if subset.empty:
            logger.warning("No rows for category %s", category)
            return pd.DataFrame()

        reps = self.get_product_representations(subset)
        centroid = reps.mean(axis=0, keepdims=True)

        with torch.inference_mode():
            sims = (
                self.scorer(
                    torch.tensor(np.repeat(centroid, len(subset), axis=0), device=self.device),
                    torch.tensor(reps, device=self.device),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # combine similarity with original sentiment score if available
        weights = subset.get("score", pd.Series(np.ones(len(subset)))).to_numpy(float)
        final = sims * weights
        subset = subset.assign(weighted_score=final, similarity_score=sims)

        return subset.sort_values("weighted_score", ascending=False).head(top_n).reset_index(
            drop=True
        )


# ---------------------------------------------------------------------------- #
# Simple convenience factory if you need to wire everything from a single call #
# ---------------------------------------------------------------------------- #
def build_engine_from_trained_folder(
    model_dir: str,
    config: dict,
    *,
    fused_feature_dim: int = 128,
) -> RecommendationEngine:
    """
    Utility to recreate the full stack from the saved `model_dir`.

    Assumes:
    - `model_dir` has  *hybrid_model.pt* , *gnn_model.pt* ,
      *feature_engineer.pkl* , *mlb.pkl* , and *config.json*
    """
    import dill as pickle
    from pathlib import Path
    import json

    path = Path(model_dir)
    with open(path / "config.json") as f:
        cfg = json.load(f)

    # ----- feature engineer ---------------------------------------------------
    with (path / "feature_engineer.pkl").open("rb") as f:
        fe: FeatureEngineer = pickle.load(f)
    with (path / "mlb.pkl").open("rb") as f:
        fe.mlb = pickle.load(f)

    # ----- models -------------------------------------------------------------
    hybrid = HybridRecommender(
        cfg,
        embedding_dim=384,
        gnn_dim=64,
        category_dim=fe.category_columns.size,
        numerical_dim=3,
        aspects_dim=fe.mlb.classes_.size,
    )
    hybrid.load_state_dict(torch.load(path / "hybrid_model.pt", map_location="cpu"))
    hybrid.eval()

    gnn = GNNModel(cfg["gnn"], input_dim=384)
    gnn.load_state_dict(torch.load(path / "gnn_model.pt", map_location="cpu"))
    gnn.eval()

    # ----- helpers ------------------------------------------------------------
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
