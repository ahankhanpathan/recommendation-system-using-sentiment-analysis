import torch
import dill as pickle  # Use dill for robust deserialization
import json
from pathlib import Path

from src.recommendation.models import HybridRecommender, GNNModel
from src.recommendation.feature_engineer import FeatureEngineer
from src.recommendation.transformer_embedder import TransformerEmbedder
from src.recommendation.graph_builder import GraphBuilder


def load_trained_model(model_dir):
    model_dir = Path(model_dir) 

    with (model_dir / "config.json").open() as f:
        config = json.load(f)

    with (model_dir / "feature_engineer.pkl").open("rb") as f:
        feature_engineer = pickle.load(f)

    with (model_dir / "mlb.pkl").open("rb") as f:
        feature_engineer.mlb = pickle.load(f)

    hybrid_model = HybridRecommender(
        config,
        embedding_dim=384,
        gnn_dim=64,
        category_dim=26,
        numerical_dim=3,
        aspects_dim=8490
    )
    hybrid_model.load_state_dict(torch.load(model_dir / "hybrid_model.pt"))
    hybrid_model.eval()

    gnn = GNNModel(config["gnn"], input_dim=384)
    gnn.load_state_dict(torch.load(model_dir / "gnn_model.pt"))
    gnn.eval()

    transformer_embedder = TransformerEmbedder(config)
    graph_builder = GraphBuilder(config)

    return hybrid_model, gnn, feature_engineer, transformer_embedder, graph_builder
