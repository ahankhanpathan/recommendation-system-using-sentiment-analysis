# --- Standard library ---
import os
import json
import time

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score


from src.recommendation.models import HybridRecommender, GNNModel

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dir = config.get('model_dir', 'models')
        self.log_dir = config.get('log_dir', 'logs')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'feature_weights': [],
            'epochs': 0
        }

    def prepare_training_data(self, df, train_indices, test_indices, feature_engineer, transformer_embedder, graph_builder):
        train_df = df.iloc[train_indices].copy().reset_index(drop=True)
        test_df = df.iloc[test_indices].copy().reset_index(drop=True)

        print("Generating text embeddings...")
        train_embeddings = transformer_embedder.generate_embeddings(train_df['all_text'])
        test_embeddings = transformer_embedder.generate_embeddings(test_df['all_text'])

        print("Engineering features...")
        train_features = feature_engineer.engineer_features(train_df)
        test_features = feature_engineer.transform_features(test_df)

        print("Building product graph...")
        train_edge_index, train_edge_attr = graph_builder.build_product_graph(
            train_embeddings,
            train_features['category_features'].values
        )

        gnn_input_dim = train_embeddings.shape[1]
        gnn = GNNModel(self.config['gnn'], input_dim=gnn_input_dim).to(self.device)

        print("Generating GNN embeddings...")
        train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float).to(self.device)
        train_edge_index = train_edge_index.to(self.device)

        gnn.eval()
        with torch.no_grad():
            train_gnn_embeddings = gnn(train_embeddings_tensor, train_edge_index).cpu().numpy()

        train_targets = (train_df['rating'] > 3).astype(int).values
        train_data = TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float),
            torch.tensor(train_gnn_embeddings, dtype=torch.float),
            torch.tensor(train_features['category_features'].values, dtype=torch.float),
            torch.tensor(train_features['numerical_features'].values, dtype=torch.float),
            torch.tensor(train_features['aspects_matrix'], dtype=torch.float),
            torch.tensor(train_targets, dtype=torch.float)
        )

        test_edge_index, test_edge_attr = graph_builder.build_product_graph(
            test_embeddings,
            test_features['category_features'].values
        )

        test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float).to(self.device)
        test_edge_index = test_edge_index.to(self.device)

        with torch.no_grad():
            test_gnn_embeddings = gnn(test_embeddings_tensor, test_edge_index).cpu().numpy()

        test_targets = (test_df['rating'] > 3).astype(int).values
        test_data = TensorDataset(
            torch.tensor(test_embeddings, dtype=torch.float),
            torch.tensor(test_gnn_embeddings, dtype=torch.float),
            torch.tensor(test_features['category_features'].values, dtype=torch.float),
            torch.tensor(test_features['numerical_features'].values, dtype=torch.float),
            torch.tensor(test_features['aspects_matrix'], dtype=torch.float),
            torch.tensor(test_targets, dtype=torch.float)
        )

        return train_data, test_data, gnn
    def train_model(self, model, train_dataloader, val_dataloader, epochs):
        model.to(self.device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            feature_weights_sum = None
            num_batches = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(train_dataloader):
                embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat, targets = [
                    t.to(self.device) for t in batch
                ]

                optimizer.zero_grad()

                predictions, attention_weights, _ = model(
                    embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat
                )

                loss = criterion(predictions, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # ✅ Sum feature weights across batches
                if feature_weights_sum is None:
                    feature_weights_sum = attention_weights.mean(dim=1).sum(dim=0)
                else:
                    feature_weights_sum += attention_weights.mean(dim=1).sum(dim=0)

            avg_train_loss = train_loss / num_batches
            avg_feature_weights = feature_weights_sum / num_batches

            model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat, targets = [
                        t.to(self.device) for t in batch
                    ]

                    predictions, _, _ = model(
                        embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat
                    )
                    loss = criterion(predictions, targets.unsqueeze(1))
                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            #  Store one feature weight vector per epoch
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['feature_weights'].append(avg_feature_weights.detach().cpu().numpy().tolist())
            self.history['epochs'] = epoch + 1

        return model

    # def train_model(self, model, train_dataloader, val_dataloader, epochs):
    #     model.to(self.device)
    #     criterion = nn.BCELoss()

    #     print("\nChecking model parameters inside train_model:")
    #     total_params_in_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print(f"Total trainable parameters: {total_params_in_train}")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"- {name}")
    #     print("--- End of parameter check ---")

    #     if total_params_in_train == 0:
    #         raise ValueError("Model has no trainable parameters.")
    #     optimizer = optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))
    #     print("Optimizer initialized.")

    #     for epoch in range(epochs):
    #         model.train()
    #         train_loss = 0
    #         feature_weights_sum = 0
    #         num_batches = 0
    #         start_time = time.time()

    #         for batch_idx, batch in enumerate(train_dataloader):
    #             # Unpack batch
    #             # Assuming the order matches your TensorDataset:
    #             # embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat, targets
    #             embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat, targets = [
    #                 t.to(self.device) for t in batch
    #             ]

    #             optimizer.zero_grad()

    #             # Forward pass
    #             predictions, attention_weights, _ = model(
    #                 embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat
    #             )

    #             # --- Add this print statement immediately after the model call ---
    #             print(f"  Epoch {epoch+1}, Batch {batch_idx}: Shape of predictions tensor after model call: {predictions.shape}")
    #             # --- End of added print statement ---

    #             # --- Keep the existing print statements for shapes before loss ---
    #             print(f"  Epoch {epoch+1}, Batch {batch_idx}:")
    #             print(f"    Shape of predictions tensor before loss: {predictions.shape}")
    #             print(f"    Shape of targets tensor before unsqueeze: {targets.shape}")
    #             print(f"    Shape of targets tensor after unsqueeze(1): {targets.unsqueeze(1).shape}")
    #             # --- End of existing print statements ---


    #             # Calculate loss
    #             loss = criterion(predictions, targets.unsqueeze(1))
    #             loss.backward()
    #             optimizer.step()

    #             train_loss += loss.item()
    #             feature_weights_sum += attention_weights.mean(dim=0).squeeze()
    #             num_batches += 1

    #         avg_train_loss = train_loss / num_batches
    #         avg_feature_weights = feature_weights_sum / num_batches

    #         model.eval()
    #         val_loss = 0
    #         num_val_batches = 0

    #         with torch.no_grad():
    #             for batch in val_dataloader:
    #                 embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat, targets = [
    #                     t.to(self.device) for t in batch
    #                 ]

    #                 predictions, _, _ = model(
    #                     embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat
    #                 )
    #                 loss = criterion(predictions, targets.unsqueeze(1))
    #                 val_loss += loss.item()
    #                 num_val_batches += 1

    #         avg_val_loss = val_loss / num_val_batches

    #         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    #         print(f"Feature Weights: {avg_feature_weights.detach().cpu().numpy()}")

    #         self.history['train_loss'].append(avg_train_loss)
    #         self.history['val_loss'].append(avg_val_loss)
    #         self.history['feature_weights'].append(avg_feature_weights.detach().cpu().numpy().tolist())
    #         self.history['epochs'] = epoch + 1

    #     return model

    def save_model(self, model, gnn, feature_engineer, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(self.model_dir, f"hybrid_recommender_{timestamp}")
        os.makedirs(model_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_path, "hybrid_model.pt"))
        torch.save(gnn.state_dict(), os.path.join(model_path, "gnn_model.pt"))

        if hasattr(feature_engineer, 'mlb') and feature_engineer.mlb is not None:
            import pickle
            with open(os.path.join(model_path, "mlb.pkl"), 'wb') as f:
                pickle.dump(feature_engineer.mlb, f)

        with open(os.path.join(model_path, "config.json"), 'w') as f:
            json.dump(self.config, f, indent=4)

        with open(os.path.join(model_path, "training_history.json"), 'w') as f:
            json.dump(self.history, f, indent=4)

        with open(os.path.join(model_path, "feature_engineer.pkl"), 'wb') as f:
            pickle.dump(feature_engineer, f)

        print(f"Model saved to {model_path}")
        return model_path

    def plot_training_history(self, output_path=None):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        epochs = list(range(1, self.history['epochs'] + 1))

        axs[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axs[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        feature_names = ['Text', 'Graph', 'Category', 'Numerical', 'Aspects']
        feature_weights = np.array(self.history['feature_weights'])

        for i, feature in enumerate(feature_names):
            axs[1].plot(epochs, feature_weights[:, i], label=feature)

        axs[1].set_title('Feature Importance Weights Over Time')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Weight')
        axs[1].legend()

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Training history plot saved to {output_path}")
        plt.show()

