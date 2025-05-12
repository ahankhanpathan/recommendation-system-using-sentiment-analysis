# models.py
import torch.nn as nn

# Moved SimilarityScorer class definition outside HybridRecommender
class SimilarityScorer(nn.Module):
   def __init__(self, input_dim):
       super(SimilarityScorer, self).__init__()
       self.model = nn.Sequential(
           nn.Linear(input_dim * 2, 256),
           nn.ReLU(),
           nn.Linear(256, 64),
           nn.ReLU(),
           nn.Linear(64, 1),
           nn.Sigmoid()
           )

   def forward(self, x1, x2):
       pair = torch.cat([x1, x2], dim=1)
       return self.model(pair)


# # Hybrid Fusion Module

# %%


import os
import json
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
class AttentionFusion(nn.Module):
    def __init__(self, input_dims, fusion_dim):
        super(AttentionFusion, self).__init__() # Ensure this is the first line

        print(f"  Inside AttentionFusion __init__ START - input_dims: {input_dims}, fusion_dim: {fusion_dim}")
        # Check parameters immediately after super().__init__()
        print(f"    Parameters immediately after super().__init__(): {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.num_features = len(input_dims)

        # Linear layers to project each feature to the fusion dimension
        self.projection_layers = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in input_dims
        ])
        print(f"    AttentionFusion - projection_layers created. Number of layers: {len(self.projection_layers)}")
        proj_params = sum(p.numel() for layer in self.projection_layers for p in layer.parameters() if p.requires_grad)
        print(f"    AttentionFusion - Projection layer trainable parameters: {proj_params}")
        print(f"    Parameters after projection_layers: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        # Attention mechanism - Query, Key, Value concept on the projected features
        self.query = nn.Linear(fusion_dim, fusion_dim)
        self.key = nn.Linear(fusion_dim, fusion_dim)
        self.value = nn.Linear(fusion_dim, fusion_dim)
        self.scale = fusion_dim ** 0.5 # Scaling factor

        print("    AttentionFusion - Query, Key, Value layers created.")
        attn_params = sum(p.numel() for p in self.query.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in self.key.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in self.value.parameters() if p.requires_grad)
        print(f"    AttentionFusion - QKV trainable parameters: {attn_params}")
        print(f"    Parameters after QKV layers: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        # Output layer after fusion
        self.output_layer = nn.Linear(fusion_dim, fusion_dim) # Example: Project fused output back to fusion_dim
        print("    AttentionFusion - output_layer created.")
        out_params = sum(p.numel() for p in self.output_layer.parameters() if p.requires_grad)
        print(f"    AttentionFusion - Output layer trainable parameters: {out_params}")
        print(f"    Parameters after output_layer: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        total_attn_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Inside AttentionFusion __init__ END - Total trainable parameters in AttentionFusion instance: {total_attn_params}")


    def forward(self, features):
        """
        Args:
            features (list of tensors): List of feature tensors [text_emb, gnn_emb, category_feat, ...]
                                        Each tensor shape should be (batch_size, feature_dim).
        Returns:
            fused_output (tensor): Combined feature representation (batch_size, fusion_dim)
            attention_weights (tensor): Weights assigned to each feature type (batch_size, num_features, num_features)
        """
        # --- Added input validation ---
        if not isinstance(features, list) or len(features) != self.num_features:
            raise ValueError(f"AttentionFusion expected a list of {self.num_features} tensors, but got {type(features)} with length {len(features) if isinstance(features, list) else 'N/A'}")

        # --- Process each feature through its projection layer ---
        projected_features = [layer(feat) for layer, feat in zip(self.projection_layers, features)]
        print(f"  Inside AttentionFusion forward - Projected features list length: {len(projected_features)}")
        for i, proj_feat in enumerate(projected_features):
             print(f"    Projected feature {i} shape: {proj_feat.shape}, device: {proj_feat.device}")


        # Stack projected features along a new dimension (batch_size, num_features, fusion_dim)
        stacked_features = torch.stack(projected_features, dim=1)
        print(f"  Inside AttentionFusion forward - Stacked features shape: {stacked_features.shape}, device: {stacked_features.device}")


        # --- Attention Calculation ---
        # Apply Q, K, V linear layers
        queries = self.query(stacked_features) # (batch_size, num_features, fusion_dim)
        keys = self.key(stacked_features)     # (batch_size, num_features, fusion_dim)
        values = self.value(stacked_features)   # (batch_size, num_features, fusion_dim)
        print(f"  Inside AttentionFusion forward - Queries shape: {queries.shape}, Keys shape: {keys.shape}, Values shape: {values.shape}")


        # Calculate attention scores: (Q @ K.transpose(-2, -1)) / scale
        # (batch_size, num_features, fusion_dim) @ (batch_size, fusion_dim, num_features)
        keys_transposed = keys.transpose(-2, -1)
        print(f"  Inside AttentionFusion forward - Keys transposed shape: {keys_transposed.shape}")

        # --- Add these print statements right before the failing matmul ---
        print(f"  Inside AttentionFusion forward - Matmul shapes for attention_scores: {queries.shape} @ {keys_transposed.shape}")
        # --- End of added print ---

        attention_scores = torch.matmul(queries, keys_transposed) / self.scale # (batch_size, num_features, num_features)
        print(f"  Inside AttentionFusion forward - Attention scores shape: {attention_scores.shape}")


        # Apply softmax to get attention weights across feature types for each item in the batch
        attention_weights = F.softmax(attention_scores, dim=-1) # (batch_size, num_features, num_features)
        print(f"  Inside AttentionFusion forward - Attention weights shape: {attention_weights.shape}")


        # Apply attention weights to values
        # (batch_size, num_features, num_features) @ (batch_size, num_features, fusion_dim)
        # This matrix multiplication seems incorrect if you want to weight each feature type's
        # *projected vector* by a single weight for that feature type.
        # It looks like you are doing something more complex, potentially weighted sum based on feature interaction scores.
        # If the goal is a weighted sum of projected features:
        # weighted_features = (attention_weights.unsqueeze(-1) * values).sum(dim=1) # Requires different attention_weights shape
        # Let's assume your original multiplication and sum is intentional based on your code.
        # --- Add these print statements right before the second matmul ---
        print(f"  Inside AttentionFusion forward - Matmul shapes for weighted_sum: {attention_weights.shape} @ {values.shape}")
        # --- End of added print ---
        weighted_sum = torch.matmul(attention_weights, values) # (batch_size, num_features, fusion_dim)
        print(f"  Inside AttentionFusion forward - Weighted sum shape: {weighted_sum.shape}")

        # Sum across the feature dimension to get a single fused vector for each item
        fused_output = weighted_sum.sum(dim=1) # (batch_size, fusion_dim)
        print(f"  Inside AttentionFusion forward - Fused output shape after sum: {fused_output.shape}")


        # Optional: Pass through an output layer
        fused_output = self.output_layer(fused_output)
        print(f"  Inside AttentionFusion forward - Fused output shape after output layer: {fused_output.shape}")


        # Return fused output and attention weights
        return fused_output, attention_weights


class HybridRecommender(nn.Module):
    def __init__(self, config, embedding_dim, gnn_dim, category_dim, numerical_dim, aspects_dim):
        super(HybridRecommender, self).__init__() # Ensure this is the first line

        print(f"Inside HybridRecommender __init__ START - embedding_dim: {embedding_dim}, gnn_dim: {gnn_dim}, category_dim: {category_dim}, numerical_dim: {numerical_dim}, aspects_dim: {aspects_dim}")
        # Check parameters immediately after super().__init__()
        print(f"  Parameters immediately after super().__init__(): {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        # Define input dimensions as a list for AttentionFusion
        # input_dims_list = [embedding_dim, gnn_dim, category_dim, numerical_dim, aspects_dim]


        # Get fusion_dim from config
        fusion_dim = config['hybrid']['fusion_output_dim']
        hidden_dim = config['hybrid'].get('hidden_dim', fusion_dim)
        input_dims_list = [fusion_dim] * 5
        

        print(f"  HybridRecommender __init__ - input_dims_list for AttentionFusion: {input_dims_list}, fusion_dim: {fusion_dim}")


        # Linear layers for each feature type - These layers *should* have parameters
        self.embedding_layer = nn.Linear(embedding_dim, fusion_dim)
        self.gnn_layer = nn.Linear(gnn_dim, fusion_dim)
        self.category_layer = nn.Linear(category_dim, fusion_dim)
        self.numerical_layer = nn.Linear(numerical_dim, fusion_dim)
        self.aspects_layer = nn.Linear(aspects_dim, fusion_dim)

        print("  HybridRecommender __init__ - Initial linear layers created.")
        initial_layer_params = sum(p.numel() for p in self.embedding_layer.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in self.gnn_layer.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in self.category_layer.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in self.numerical_layer.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in self.aspects_layer.parameters() if p.requires_grad)
        print(f"  HybridRecommender __init__ - Initial linear layer trainable parameters (sum of individual): {initial_layer_params}")
        # Check parameters of HybridRecommender itself after creating these layers
        print(f"  Parameters after initial linear layers: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        # Fusion layer - This should have parameters from its submodules
        self.fusion_layer = AttentionFusion(input_dims_list, fusion_dim)
        print("  HybridRecommender __init__ - AttentionFusion layer created.")
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters() if p.requires_grad)
        print(f"  HybridRecommender __init__ - AttentionFusion trainable parameters: {fusion_params}")
        print(f"  Parameters after fusion_layer: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        # Predictor layer - This nn.Sequential should have parameters from its linear layers
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config['gnn'].get('dropout', 0.2)),
            nn.Linear(hidden_dim, 1), # Output dimension is 1 for binary prediction
            nn.Sigmoid()
        )
        print("  HybridRecommender __init__ - Predictor layer created.")
        predictor_params = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        print(f"  HybridRecommender __init__ - Predictor trainable parameters: {predictor_params}")
        print(f"  Parameters after predictor: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


        total_hybrid_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Inside HybridRecommender __init__ END - Total trainable parameters in HybridRecommender instance: {total_hybrid_params}")

    def forward(self, embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat):
        # --- Add print statements for input tensor shapes to forward ---
        print(f"  Inside HybridRecommender forward - embedding_feat shape: {embedding_feat.shape}, device: {embedding_feat.device}")
        print(f"  Inside HybridRecommender forward - gnn_feat shape: {gnn_feat.shape}, device: {gnn_feat.device}")
        print(f"  Inside HybridRecommender forward - category_feat shape: {category_feat.shape}, device: {category_feat.device}")
        print(f"  Inside HybridRecommender forward - numerical_feat shape: {numerical_feat.shape}, device: {numerical_feat.device}")
        print(f"  Inside HybridRecommender forward - aspects_feat shape: {aspects_feat.shape}, device: {aspects_feat.device}")
    
        # Initialize outputs with default values in case of early error
        prediction = torch.tensor([float('nan')], device=embedding_feat.device)
        attention_weights = None
        fused_output = torch.tensor([float('nan')], device=embedding_feat.device)
    
        # --- Process each feature through its linear layer with more granular checks ---
        try:
            print("  Inside HybridRecommender forward - Calling embedding_layer...")
            embedded = self.embedding_layer(embedding_feat)
            print(f"    embedding_layer output shape: {embedded.shape}")
    
            print("  Inside HybridRecommender forward - Calling gnn_layer...")
            gnn_out = self.gnn_layer(gnn_feat)
            print(f"    gnn_layer output shape: {gnn_out.shape}")
    
            print("  Inside HybridRecommender forward - Calling category_layer...")
            category_out = self.category_layer(category_feat)
            print(f"    category_layer output shape: {category_out.shape}")
    
            print("  Inside HybridRecommender forward - Calling numerical_layer...")
            numerical_out = self.numerical_layer(numerical_feat)
            print(f"    numerical_layer output shape: {numerical_out.shape}")
    
            print("  Inside HybridRecommender forward - Calling aspects_layer...")
            aspects_out = self.aspects_layer(aspects_feat)
            print(f"    aspects_layer output shape: {aspects_out.shape}")
    
            print("  Inside HybridRecommender forward - All initial linear layers processed successfully.")
    
            # --- Prepare features for fusion layer ---
            features_for_fusion = [embedded, gnn_out, category_out, numerical_out, aspects_out]
            print(f"  Inside HybridRecommender forward - Prepared features for fusion. List length: {len(features_for_fusion)}")
            for i, feat in enumerate(features_for_fusion):
                print(f"    Fusion input {i} shape: {feat.shape}, device: {feat.device}")
    
            # Pass through fusion layer (AttentionFusion)
            print("  Inside HybridRecommender forward - Calling fusion_layer...")
            fused_output, attention_weights = self.fusion_layer(features_for_fusion)
            print("  Inside HybridRecommender forward - Fusion layer executed successfully.")
            print(f"  Inside HybridRecommender forward - fused_output shape: {fused_output.shape}, device: {fused_output.device}")
            print(f"  Inside HybridRecommender forward - attention_weights shape: {attention_weights.shape}, device: {attention_weights.device}")
    
            # Pass fused features through predictor
            print("  Inside HybridRecommender forward - Calling predictor...")
            prediction = self.predictor(fused_output)
            print(f"  Inside HybridRecommender forward - Prediction shape after predictor: {prediction.shape}")
    
            # Ensure prediction has shape [batch_size, 1]
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(1)
                print(f"  Inside HybridRecommender forward - Unsqueezed prediction to shape: {prediction.shape}")
    
        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during processing: {e}")
            # We'll still return our default values if there's an error
    
        print(f"  Inside HybridRecommender forward - Returning prediction with shape: {prediction.shape}")
        return prediction, attention_weights, fused_output

        # --- Prepare features for fusion layer (defined AFTER the first try block) ---
        # The AttentionFusion layer expects a list of tensors in a specific order
        # Ensure this order matches your AttentionFusion implementation:
        # [text_emb, gnn_emb, category_feat, numerical_feat, aspects_feat]
        features_for_fusion = [embedded, gnn_out, category_out, numerical_out, aspects_out]
        print(f"  Inside HybridRecommender forward - Prepared features for fusion. List length: {len(features_for_fusion)}")
        for i, feat in enumerate(features_for_fusion):
             print(f"    Fusion input {i} shape: {feat.shape}, device: {feat.device}")


        # Pass through fusion layer (AttentionFusion)
        try:
            print("  Inside HybridRecommender forward - Calling fusion_layer...")
            # Ensure the fusion layer returns two values and they are unpacked correctly
            fused_output, attention_weights = self.fusion_layer(features_for_fusion)
            print("  Inside HybridRecommender forward - Fusion layer executed successfully.")
            print(f"  Inside HybridRecommender forward - fused_output type: {type(fused_output)}, shape: {fused_output.shape}, device: {fused_output.device}")
            # Check type before accessing shape for attention_weights
            print(f"  Inside HybridRecommender forward - attention_weights type: {type(attention_weights)}, shape: {attention_weights.shape if isinstance(attention_weights, torch.Tensor) else 'N/A'}, device: {attention_weights.device if isinstance(attention_weights, torch.Tensor) else 'N/A'}")


        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during fusion layer: {e}")
            # Return default or error indicator if fusion fails
            return prediction, attention_weights, fused_output


        # Pass fused features through predictor
        try:
            print("  Inside HybridRecommender forward - Calling predictor...")
            prediction = self.predictor(fused_output)
            # --- Add this print statement immediately after the predictor call ---
            print(f"  Inside HybridRecommender forward - Prediction shape after predictor: {prediction.shape}")
            # --- End of added print statement ---


        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during predictor layer: {e}")
            # Return error indicators if predictor fails
            return prediction, attention_weights, fused_output


        # --- Final return statement ---
        # Ensure prediction has shape [batch_size, 1] before returning
        # Reshape prediction to [batch_size, 1] regardless of its current dimensions
        # This is a more robust way to ensure the correct shape for loss calculation
        if prediction.dim() == 1:
             prediction = prediction.unsqueeze(1)
             print(f"  Inside HybridRecommender forward - INFO: Prediction shape was [batch_size], unsqueezing to shape: {prediction.shape}")
        elif prediction.dim() > 2 or (prediction.dim() == 2 and prediction.shape[1] != 1):
             # Handle unexpected shapes if necessary, e.g., flatten and then unsqueeze
             print(f"  Inside HybridRecommender forward - WARNING: Unexpected prediction shape {prediction.shape}. Attempting to reshape.")
             prediction = prediction.view(-1, 1) # Flatten to [N, 1]
             print(f"  Inside HybridRecommender forward - INFO: Reshaped prediction to shape: {prediction.shape}")


        print(f"  Inside HybridRecommender forward - Returning prediction with shape: {prediction.shape}")
        return prediction, attention_weights, fused_output


        # --- Prepare features for fusion layer (defined AFTER the first try block) ---
        # The AttentionFusion layer expects a list of tensors in a specific order
        # Ensure this order matches your AttentionFusion implementation:
        # [text_emb, gnn_emb, category_feat, numerical_feat, aspects_feat]
        features_for_fusion = [embedded, gnn_out, category_out, numerical_out, aspects_out]
        print(f"  Inside HybridRecommender forward - Prepared features for fusion. List length: {len(features_for_fusion)}")
        for i, feat in enumerate(features_for_fusion):
             print(f"    Fusion input {i} shape: {feat.shape}, device: {feat.device}")


        # Pass through fusion layer (AttentionFusion)
        try:
            print("  Inside HybridRecommender forward - Calling fusion_layer...")
            # Ensure the fusion layer returns two values and they are unpacked correctly
            fused_output, attention_weights = self.fusion_layer(features_for_fusion)
            print("  Inside HybridRecommender forward - Fusion layer executed successfully.")
            print(f"  Inside HybridRecommender forward - fused_output type: {type(fused_output)}, shape: {fused_output.shape}, device: {fused_output.device}")
            # Check type before accessing shape for attention_weights
            print(f"  Inside HybridRecommender forward - attention_weights type: {type(attention_weights)}, shape: {attention_weights.shape if isinstance(attention_weights, torch.Tensor) else 'N/A'}, device: {attention_weights.device if isinstance(attention_weights, torch.Tensor) else 'N/A'}")


        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during fusion layer: {e}")
            # Return default or error indicator if fusion fails
            return prediction, attention_weights, fused_output


        # Pass fused features through predictor
        try:
            print("  Inside HybridRecommender forward - Calling predictor...")
            prediction = self.predictor(fused_output)
            print(f"  Inside HybridRecommender forward - Predictor executed, prediction type: {type(prediction)}, shape: {prediction.shape}, device: {prediction.device}")

        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during predictor layer: {e}")
            # Return error indicators if predictor fails
            return prediction, attention_weights, fused_output


        # --- Final return statement ---
        # Add unsqueeze(1) here if prediction is unexpectedly squeezed
        # This is a temporary workaround to ensure shape is [batch_size, 1] for loss calculation
        if prediction.dim() == 1:
             prediction = prediction.unsqueeze(1)
             print(f"  Inside HybridRecommender forward - WARNING: Prediction was squeezed, unsqueezing to shape: {prediction.shape}")


        print(f"  Inside HybridRecommender forward - Returning prediction with shape: {prediction.shape}")
        return prediction, attention_weights, fused_output


        # --- Prepare features for fusion layer (defined AFTER the first try block) ---
        # The AttentionFusion layer expects a list of tensors in a specific order
        # Ensure this order matches your AttentionFusion implementation:
        # [text_emb, gnn_emb, category_feat, numerical_feat, aspects_feat]
        features_for_fusion = [embedded, gnn_out, category_out, numerical_out, aspects_out]
        print(f"  Inside HybridRecommender forward - Prepared features for fusion. List length: {len(features_for_fusion)}")
        for i, feat in enumerate(features_for_fusion):
             print(f"    Fusion input {i} shape: {feat.shape}, device: {feat.device}")


        # Pass through fusion layer (AttentionFusion)
        try:
            print("  Inside HybridRecommender forward - Calling fusion_layer...")
            # Ensure the fusion layer returns two values and they are unpacked correctly
            fused_output, attention_weights = self.fusion_layer(features_for_fusion)
            print("  Inside HybridRecommender forward - Fusion layer executed successfully.")
            print(f"  Inside HybridRecommender forward - fused_output type: {type(fused_output)}, shape: {fused_output.shape}, device: {fused_output.device}")
            # Check type before accessing shape for attention_weights
            print(f"  Inside HybridRecommender forward - attention_weights type: {type(attention_weights)}, shape: {attention_weights.shape if isinstance(attention_weights, torch.Tensor) else 'N/A'}, device: {attention_weights.device if isinstance(attention_weights, torch.Tensor) else 'N/A'}")


        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during fusion layer: {e}")
            # Return default or error indicator if fusion fails
            return prediction, attention_weights, fused_output


        # Pass fused features through predictor
        try:
            print("  Inside HybridRecommender forward - Calling predictor...")
            prediction = self.predictor(fused_output)
            print(f"  Inside HybridRecommender forward - Predictor executed, prediction type: {type(prediction)}, shape: {prediction.shape}, device: {prediction.device}") # <-- Added print for shape

        except Exception as e:
            print(f"  Inside HybridRecommender forward - Error during predictor layer: {e}")
            # Return error indicators if predictor fails
            return prediction, attention_weights, fused_output


        # --- Final return statement ---
        print(f"  Inside HybridRecommender forward - Returning prediction with shape: {prediction.shape}") # <-- Added print just before return
        return prediction, attention_weights, fused_output


def get_feature_importance(self):
        # Dummy pass to extract average attention over features
        with torch.no_grad():
            B = 1
            dims = self.fusion.embedding_transform.in_features
            dummy = lambda dim: torch.zeros(B, dim)
            _, weights, _ = self.forward(
                dummy(self.fusion.embedding_transform.in_features),
                dummy(self.fusion.gnn_transform.in_features),
                dummy(self.fusion.category_transform.in_features),
                dummy(self.fusion.numerical_transform.in_features),
                dummy(self.fusion.aspects_transform.in_features),
            )
            avg = weights.mean(dim=0)  # [5]
            names = ['text','graph','category','numerical','aspects']
            return {n: float(w) for n,w in zip(names, avg)}





# # Graph Construction & GNN Module

# %%


import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


class GNNModel(torch.nn.Module):
    def __init__(self, config, input_dim):
        super(GNNModel, self).__init__()
        self.config = config
        self.gnn_type = config.get('gnn_type', 'gcn')
        hidden_channels = config.get('gnn_hidden_channels', 128)
        num_layers = config.get('gnn_num_layers', 2)
        output_dim = config.get('gnn_output_dim', 64)
        dropout = config.get('gnn_dropout', 0.2)
        
        # Define GNN layers
        self.layers = torch.nn.ModuleList()
        
        if self.gnn_type == 'gcn':
            self.layers.append(GCNConv(input_dim, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, output_dim))
        elif self.gnn_type == 'gat':
            self.layers.append(GATConv(input_dim, hidden_channels, heads=4, dropout=dropout))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout))
            self.layers.append(GATConv(hidden_channels * 4, output_dim, heads=1, concat=False, dropout=dropout))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, layer in enumerate(self.layers[:-1]):
            if self.gnn_type == 'gcn':
                x = layer(x, edge_index)
            else:  # gat
                x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Last layer
        if self.gnn_type == 'gcn':
            x = self.layers[-1](x, edge_index)
        else:  # gat
            x = self.layers[-1](x, edge_index)
        
        return x



