"""Training loop and utilities for the HybridRecommender model."""

import copy
import json
import logging
import os
from datetime import datetime

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.recommendation.models import GNNModel, HybridRecommender

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Early Stopping                                                               #
# --------------------------------------------------------------------------- #

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Saves the best model weights internally and restores them at the end so
    the returned model is always the best checkpoint, not the last epoch.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best_loss = float("inf")
        self._best_weights: dict | None = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Call at the end of each epoch.

        Returns True if training should stop.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._best_weights = copy.deepcopy(model.state_dict())
            self._counter = 0
            logger.info("EarlyStopping: new best val_loss=%.4f", val_loss)
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: no improvement (%d / %d)", self._counter, self.patience
            )

        return self._counter >= self.patience

    def restore_best(self, model: torch.nn.Module) -> None:
        """Load the best weights back into the model."""
        if self._best_weights is not None:
            model.load_state_dict(self._best_weights)
            logger.info("EarlyStopping: best weights restored (val_loss=%.4f)", self._best_loss)


# --------------------------------------------------------------------------- #
# Training Pipeline                                                            #
# --------------------------------------------------------------------------- #

class TrainingPipeline:
    """Orchestrates data preparation, training, saving, and plotting."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = config.get("paths", {}).get("model_dir", config.get("model_dir", "models"))
        self.log_dir = config.get("paths", {}).get("log_dir", config.get("log_dir", "logs"))

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history: dict = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "feature_weights": [],
            "epochs": 0,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

    # ---------------------------------------------------------------------- #
    # Data preparation                                                         #
    # ---------------------------------------------------------------------- #

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

    # ---------------------------------------------------------------------- #
    # Training loop                                                            #
    # ---------------------------------------------------------------------- #

    def train_model(
        self,
        model: HybridRecommender,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
    ) -> HybridRecommender:
        """
        Run the training loop with:
        - ReduceLROnPlateau learning-rate scheduling
        - Gradient clipping (max_norm configurable)
        - Early stopping with best-weight restoration
        """
        hybrid_cfg = self.config.get("hybrid", {})
        lr: float = hybrid_cfg.get("learning_rate", 1e-3)
        max_grad_norm: float = hybrid_cfg.get("max_grad_norm", 1.0)
        patience: int = hybrid_cfg.get("early_stopping_patience", 5)
        lr_patience: int = hybrid_cfg.get("lr_scheduler_patience", 2)
        lr_factor: float = hybrid_cfg.get("lr_scheduler_factor", 0.5)
        min_lr: float = hybrid_cfg.get("min_lr", 1e-6)

        model.to(self.device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            # ---------------------------------------------------------------- #
            # Training pass                                                     #
            # ---------------------------------------------------------------- #
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

                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                batch_w = attn_weights.mean(dim=1).sum(dim=0)
                feature_weights_sum = (
                    batch_w if feature_weights_sum is None else feature_weights_sum + batch_w
                )

            avg_train_loss = train_loss / num_batches
            avg_feature_weights = feature_weights_sum / num_batches

            # ---------------------------------------------------------------- #
            # Validation pass                                                   #
            # ---------------------------------------------------------------- #
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
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d — train: %.4f  val: %.4f  lr: %.2e",
                epoch + 1, epochs, avg_train_loss, avg_val_loss, current_lr,
            )

            # ---------------------------------------------------------------- #
            # Scheduler + history                                               #
            # ---------------------------------------------------------------- #
            scheduler.step(avg_val_loss)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["learning_rates"].append(current_lr)
            self.history["feature_weights"].append(
                avg_feature_weights.detach().cpu().numpy().tolist()
            )
            self.history["epochs"] = epoch + 1

            if avg_val_loss < self.history["best_val_loss"]:
                self.history["best_val_loss"] = avg_val_loss
                self.history["best_epoch"] = epoch + 1

            # ---------------------------------------------------------------- #
            # Early stopping                                                    #
            # ---------------------------------------------------------------- #
            if early_stopping.step(avg_val_loss, model):
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        early_stopping.restore_best(model)
        logger.info(
            "Training complete. Best epoch: %d, best val_loss: %.4f",
            self.history["best_epoch"],
            self.history["best_val_loss"],
        )
        return model

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

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

    # ---------------------------------------------------------------------- #
    # Visualisation                                                            #
    # ---------------------------------------------------------------------- #

    def plot_training_history(self, output_path: str | None = None) -> None:
        """
        Plot three panels:
        1. Training vs validation loss with best-epoch marker
        2. Learning rate schedule over epochs
        3. Per-feature attention weights over time
        """
        epochs = list(range(1, self.history["epochs"] + 1))
        feature_names = ["Text", "Graph", "Category", "Numerical", "Aspects"]
        feature_weights = np.array(self.history["feature_weights"])

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # --- Loss ---
        axs[0].plot(epochs, self.history["train_loss"], "b-o", markersize=4, label="Train Loss")
        axs[0].plot(epochs, self.history["val_loss"], "r-o", markersize=4, label="Val Loss")
        best_ep = self.history.get("best_epoch", 0)
        if best_ep:
            axs[0].axvline(best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best epoch ({best_ep})")
        axs[0].set_title("Training and Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("BCE Loss")
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # --- Learning rate ---
        if self.history.get("learning_rates"):
            axs[1].plot(epochs, self.history["learning_rates"], "g-o", markersize=4)
            axs[1].set_title("Learning Rate Schedule")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Learning Rate")
            axs[1].set_yscale("log")
            axs[1].grid(alpha=0.3)

        # --- Feature weights ---
        for i, name in enumerate(feature_names):
            axs[2].plot(epochs, feature_weights[:, i], marker="o", markersize=3, label=name)
        axs[2].set_title("Feature Importance Weights Over Time")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Attention Weight")
        axs[2].legend()
        axs[2].grid(alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
            logger.info("Training history plot saved to %s", output_path)
        plt.show()
