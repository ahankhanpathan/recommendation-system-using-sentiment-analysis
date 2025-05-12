# graph_builder.py


import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class GraphBuilder:
    def __init__(self, config):
        self.config = config
        self.similarity_threshold = config.get('similarity_threshold', 0.5)
        
    def build_product_graph(self, embeddings, category_features=None, k=10):
        """
        Build a product similarity graph based on embeddings and optional category features
        
        Args:
            embeddings: Numpy array of product embeddings
            category_features: Optional categorical features to enforce category constraints
            k: Number of nearest neighbors to connect in the graph
            
        Returns:
            edge_index: PyTorch Geometric edge index tensor
            edge_attr: Edge weights based on similarity
        """
        num_products = embeddings.shape[0]
        
        # Calculate cosine similarity matrix
        similarity = cosine_similarity(embeddings)
        
        # Apply category constraints if provided
        # if category_features is not None:
        #     category_mask = category_features @ category_features.T
        #     similarity = similarity * (category_mask > 0).astype(float)

        # #--mask for category
        # if category_features is not None:
        #    # only connect products with identical main_category
        #     same_cat = (category_features[:, None] == category_features[None, :]).all(axis=2)
        #     similarity = similarity * same_cat.astype(float)

        #New cateogry for category 
        if category_features is not None:
    # category_features is one–hot → turn it into a single id
            cat_id = category_features.argmax(1)          # (N,)
            same_cat = cat_id[:, None] == cat_id[None, :]
            similarity *= same_cat.astype(float)

        
        # Convert to sparse representation for efficiency
        similarity_sparse = self._get_k_nearest_neighbors(similarity, k)
        
        # Create edge index and attributes for PyTorch Geometric
        rows, cols = similarity_sparse.nonzero()
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(similarity_sparse[rows, cols], dtype=torch.float)
        
        return edge_index, edge_attr
    
    def _get_k_nearest_neighbors(self, similarity_matrix, k):
        """Get k nearest neighbors for each node from similarity matrix"""
        n = similarity_matrix.shape[0]
        
        # Create masks for top-k values
        similarity_sparse = np.zeros_like(similarity_matrix)
        
        # For each row, get the top k indices
        for i in range(n):
            # Sort similarities in descending order
            top_indices = np.argsort(-similarity_matrix[i, :])[:k+1]  # +1 to include self
            
            # Set the corresponding values in the sparse matrix
            for j in top_indices:
                if i != j and similarity_matrix[i, j] > self.similarity_threshold:  # Exclude self-loops
                    similarity_sparse[i, j] = similarity_matrix[i, j]
        
        return similarity_sparse

