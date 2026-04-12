"""Training loop and utilities for the HybridRecommender model."""

import json
import logging
import os
import time
from datetime import datetime

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.recommendation.models import GNNModel, HybridRecommender

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates data preparation, training, saving, and plotting."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = config.get("model_dir", config.get("paths", {}).get("model_dir", "models"))
        self.log_dir = config.get("log_dir", config.get("paths", {}).get("log_dir", "logs"))

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history: dict = {
            "train_loss": [],
            "val_loss": [],
            "feature_weights": [],
            "epochs": 0,
        }

    def prepare_training_data(
        self,
        df,
        train_indices,
        test_indices,
        feature_engineer,
        transformer_embedder,
        graph_builder,
    ) -> tuple:
        """Generate embeddings, build graphs, and return TensorDatasets for train/test."""
        train_df = df.iloc[train_indices].copy().reset_index(drop=True)
        test_df = df.iloc[test_indices].copy().reset_index(drop=True)

        logger.info("Generating text embeddings …")
        train_embeddings = transformer_embedder.generate_embeddings(train_df["all_text"].tolist())
        test_embeddings = transformer_embedder.generate_embeddings(test_df["all_text"].tolist())

        logger.info("Engineering features …")
        train_features = feature_engineer.engineer_features(train_df)
        test_features = feature_engineer.transform_features(test_df)

        logger.info("Building product graph and running GNN …")
        train_edge_index, _ = graph_builder.build_product_graph(
            train_embeddings,
            train_features["category_features"].values,
        )

        gnn = GNNModel(self.config["gnn"], input_dim=train_embeddings.shape[1]).to(self.device)
        gnn.eval()
        with torch.no_grad():
            train_emb_t = torch.tensor(train_embeddings, dtype=torch.float).to(self.device)
            train_gnn_emb = gnn(train_emb_t, train_edge_index.to(self.device)).cpu().numpy()

        train_targets = (train_df["rating"] > 3).astype(int).values
        train_data = TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float),
            torch.tensor(train_gnn_emb, dtype=torch.float),
            torch.tensor(train_features["category_features"].values, dtype=torch.float),
            torch.tensor(train_features["numerical_features"].values, dtype=torch.float),
            torch.tensor(train_features["aspects_matrix"], dtype=torch.float),
            torch.tensor(train_targets, dtype=torch.float),
        )

        test_edge_index, _ = graph_builder.build_product_graph(
            test_embeddings,
            test_features["category_features"].values,
        )
        with torch.no_grad():
            test_emb_t = torch.tensor(test_embeddings, dtype=torch.float).to(self.device)
            test_gnn_emb = gnn(test_emb_t, test_edge_index.to(self.device)).cpu().numpy()

        test_targets = (test_df["rating"] > 3).astype(int).values
        test_data = TensorDataset(
            torch.tensor(test_embeddings, dtype=torch.float),
            torch.tensor(test_gnn_emb, dtype=torch.float),
            torch.tensor(test_features["category_features"].values, dtype=torch.float),
            torch.tensor(test_features["numerical_features"].values, dtype=torch.float),
            torch.tensor(test_features["aspects_matrix"], dtype=torch.float),
            torch.tensor(test_targets, dtype=torch.float),
        )

        return train_data, test_data, gnn

    def train_model(
        self,
        model: HybridRecommender,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
    ) -> HybridRecommender:
        """Run the training loop and populate `self.history`."""
        model.to(self.device)
        criterion = torch.nn.BCELoss()
        lr = self.config.get("learning_rate", self.config.get("hybrid", {}).get("learning_rate", 1e-3))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            feature_weights_sum = None
            num_batches = 0

            for batch in train_dataloader:
                emb, gnn_f, cat, num, asp, targets = [t.to(self.device) for t in batch]
                optimizer.zero_grad()

                preds, attn_weights, _ = model(emb, gnn_f, cat, num, asp)
                loss = criterion(preds, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                batch_w = attn_weights.mean(dim=1).sum(dim=0)
                feature_weights_sum = (
                    batch_w if feature_weights_sum is None else feature_weights_sum + batch_w
                )

            avg_train_loss = train_loss / num_batches
            avg_feature_weights = feature_weights_sum / num_batches

            model.eval()
            val_loss = 0.0
            num_val = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    emb, gnn_f, cat, num, asp, targets = [t.to(self.device) for t in batch]
                    preds, _, _ = model(emb, gnn_f, cat, num, asp)
                    val_loss += criterion(preds, targets.unsqueeze(1)).item()
                    num_val += 1

            avg_val_loss = val_loss / num_val

            logger.info(
                "Epoch %d/%d — train loss: %.4f  val loss: %.4f",
                epoch + 1, epochs, avg_train_loss, avg_val_loss,
            )

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["feature_weights"].append(
                avg_feature_weights.detach().cpu().numpy().tolist()
            )
            self.history["epochs"] = epoch + 1

        return model

    def save_model(
        self,
        model: HybridRecommender,
        gnn: GNNModel,
        feature_engineer,
        timestamp: str | None = None,
    ) -> str:
        """Save model weights, feature engineer, config, and training history."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(self.model_dir, f"hybrid_recommender_{timestamp}")
        os.makedirs(model_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_path, "hybrid_model.pt"))
        torch.save(gnn.state_dict(), os.path.join(model_path, "gnn_model.pt"))

        if hasattr(feature_engineer, "mlb") and feature_engineer.mlb is not None:
            with open(os.path.join(model_path, "mlb.pkl"), "wb") as f:
                pickle.dump(feature_engineer.mlb, f)

        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        with open(os.path.join(model_path, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=4)

        with open(os.path.join(model_path, "feature_engineer.pkl"), "wb") as f:
            pickle.dump(feature_engineer, f)

        logger.info("Model saved to %s", model_path)
        return model_path

    def plot_training_history(self, output_path: str | None = None) -> None:
        """Plot training/validation loss and per-feature attention weights over epochs."""
        epochs = list(range(1, self.history["epochs"] + 1))
        feature_names = ["Text", "Graph", "Category", "Numerical", "Aspects"]
        feature_weights = np.array(self.history["feature_weights"])

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        axs[0].plot(epochs, self.history["train_loss"], "b-", label="Training Loss")
        axs[0].plot(epochs, self.history["val_loss"], "r-", label="Validation Loss")
        axs[0].set_title("Training and Validation Loss")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        for i, name in enumerate(feature_names):
            axs[1].plot(epochs, feature_weights[:, i], label=name)
        axs[1].set_title("Feature Importance Weights Over Time")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Weight")
        axs[1].legend()

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            logger.info("Training history plot saved to %s", output_path)
        plt.show()
