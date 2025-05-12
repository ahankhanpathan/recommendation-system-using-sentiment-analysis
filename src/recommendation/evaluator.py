# evaluator.py


# # Evaluation Framework

# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import json

class RecommendationEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metrics = {}
        self.k_values = [5, 10, 20]

    # Keep only ONE definition of evaluate_model
    def evaluate_model(self, model, dataloader, num_items):
        model.eval()
        all_predictions = []
        all_targets = []
        all_recommendations = [] # This list needs to be populated for coverage

        with torch.no_grad():
            for batch in dataloader:
                # Unpack all items if your dataset has more than 2 elements:
                embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat, targets = [
                    t.to(self.device) for t in batch
                ]
                predictions, _, _ = model(
                    embedding_feat, gnn_feat, category_feat, numerical_feat, aspects_feat
                )
                predictions = predictions.cpu().numpy().flatten()
                targets = targets.cpu().numpy().flatten()

                all_predictions.append(predictions)
                all_targets.append(targets)

                # Capture top_k_indices for coverage calculation
                # Get top indices up to the maximum k value used for evaluation
                top_indices = np.argsort(-predictions)[:max(self.k_values)]
                # Change .extend to .append here
                all_recommendations.append(top_indices.tolist()) # <--- Changed to append


        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        metrics = {}
        for k in self.k_values:
            precision, recall, f1 = self.calculate_precision_recall_at_k(all_targets, all_predictions, k)
            ndcg = self.calculate_ndcg_at_k(all_targets, all_predictions, k)
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
            metrics[f'ndcg@{k}'] = ndcg

        # Calculate coverage using the populated all_recommendations list
        coverage = self.calculate_coverage(all_recommendations, num_items)
        metrics['coverage'] = coverage

        self.metrics = metrics
        return metrics

    def prepare_evaluation_data(self, df):
        """Prepare data for evaluation"""
        df['positive_interaction'] = (df['rating'] > 3).astype(int)
        user_item_matrix = df.pivot_table(
            index='user_id', 
            columns='parent_asin', 
            values='positive_interaction',
            fill_value=0
        )
        return user_item_matrix

    def calculate_precision_recall_at_k(self, y_true, y_pred, k=10):
        """Calculate precision and recall at k"""
        top_k_indices = np.argsort(-y_pred)[:k]
        top_k_pred = np.zeros_like(y_pred)
        top_k_pred[top_k_indices] = 1
        precision = precision_score(y_true, top_k_pred, zero_division=0)
        recall = recall_score(y_true, top_k_pred, zero_division=0)
        f1 = f1_score(y_true, top_k_pred, zero_division=0)
        return precision, recall, f1

    def calculate_ndcg_at_k(self, y_true, y_pred, k=10):
        """Calculate NDCG at k"""
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
        ndcg = ndcg_score(y_true, y_pred, k=k)
        return ndcg

    def calculate_coverage(self, all_recommendations, total_items):
        """Calculate catalog coverage"""
        unique_items = set()
        for rec_list in all_recommendations:
            unique_items.update(rec_list)
        coverage = len(unique_items) / total_items
        return coverage

    def plot_metrics(self, output_path=None):
        """Plot evaluation metrics"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        k_values = self.k_values
        precision_values = [self.metrics[f'precision@{k}'] for k in k_values]
        recall_values = [self.metrics[f'recall@{k}'] for k in k_values]
        f1_values = [self.metrics[f'f1@{k}'] for k in k_values]

        axs[0, 0].plot(k_values, precision_values, 'o-', label='Precision')
        axs[0, 0].plot(k_values, recall_values, 'o-', label='Recall')
        axs[0, 0].plot(k_values, f1_values, 'o-', label='F1')
        axs[0, 0].set_xlabel('k')
        axs[0, 0].set_title('Precision, Recall, and F1 at k')
        axs[0, 0].legend()

        ndcg_values = [self.metrics[f'ndcg@{k}'] for k in k_values]
        axs[0, 1].plot(k_values, ndcg_values, 'o-', color='green')
        axs[0, 1].set_xlabel('k')
        axs[0, 1].set_title('NDCG at k')

        axs[1, 0].bar(['Coverage'], [self.metrics['coverage']], color='purple')
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_title('Catalog Coverage')

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.01, f'Evaluation Time: {timestamp}', ha='center')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Evaluation metrics plot saved to {output_path}")
        plt.show()

    def evaluate_similarity_scorer(self, recommendation_engine, df):
        print("✅ Running SimilarityScorer Evaluation...")
        from sklearn.metrics import precision_score, recall_score

        product_reps = recommendation_engine.get_product_representations(df)
        # REMOVE: scorer = SimilarityScorer(product_reps.shape[1]).to(self.device)
        scorer = recommendation_engine.similarity_scorer # Use the scorer from RecommendationEngine
        # REMOVE: scorer.eval() # eval mode should be set in RecommendationEngine __init__

        k_values = self.k_values
        precisions = {k: [] for k in k_values}
        recalls = {k: [] for k in k_values}

        ground_truth = (df['rating'].values > 3).astype(int)
        gt_idx = np.where(ground_truth == 1)[0]

        for idx in range(len(df)):
            target = product_reps[idx]
            target_tensor = torch.tensor(target).unsqueeze(0).repeat(len(df), 1).to(self.device)
            reps_tensor = torch.tensor(product_reps).to(self.device)

            with torch.no_grad():
                scores = scorer(target_tensor, reps_tensor).squeeze().cpu().numpy() # Use the passed instance's scorer

            top_k_idx = np.argsort(-scores)

            for k in k_values:
                top_k = top_k_idx[:k]
                hits = np.intersect1d(top_k, gt_idx)
                precisions[k].append(len(hits) / k)
                recalls[k].append(len(hits) / len(gt_idx) if len(gt_idx) > 0 else 0)

        self.metrics = {
            f'precision@{k}': float(np.mean(precisions[k])) for k in k_values
        }
        self.metrics.update({
            f'recall@{k}': float(np.mean(recalls[k])) for k in k_values
        })

        print("\n📊 Similarity Scorer Evaluation:")
        for k in k_values:
            print(f"precision@{k}: {self.metrics[f'precision@{k}']:.4f}, recall@{k}: {self.metrics[f'recall@{k}']:.4f}")

        return self.metrics


    def save_metrics(self, output_path):
        """Save metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            print(f"Evaluation metrics saved to {output_path}")


# # Explainability Module

# %%


import requests
import json

class ExplainabilityModule:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('groq_api_key', '')
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = config.get('groq_model', 'llama3-70b-8192')
    
    def generate_explanation(self, product_data, recommendation_data, feature_weights):
        """
        Generate natural language explanation for recommendation
        
        Args:
            product_data: Original product data
            recommendation_data: Recommended product data
            feature_weights: Weights from attention mechanism
        
        Returns:
            str: Natural language explanation
        """
        # Format product information
        orig_product = {
            'title': product_data['title'],
            'category': product_data['main_category'],
            'price': product_data['price'],
            'rating': product_data['rating'],
            'description': product_data['description'][:200] + "..." if len(product_data['description']) > 200 else product_data['description']
        }
        
        rec_product = {
            'title': recommendation_data['title'],
            'category': recommendation_data['main_category'],
            'price': recommendation_data['price'],
            'rating': recommendation_data['rating'],
            'similarity_score': recommendation_data['similarity_score'],
            'description': recommendation_data['description'][:200] + "..." if len(recommendation_data['description']) > 200 else recommendation_data['description']
        }
        
        # Feature weights
        weights = {
            'text_content': float(feature_weights[0]),
            'product_relationships': float(feature_weights[1]),
            'category': float(feature_weights[2]),
            'numerical_attributes': float(feature_weights[3]),
            'product_aspects': float(feature_weights[4])
        }
        
        # Create prompt
        prompt = f"""
        As a recommendation system, I need to explain why Product A is being recommended to a user who showed interest in Product B.
        
        Product B (User's Interest):
        {json.dumps(orig_product, indent=2)}
        
        Product A (Recommendation):
        {json.dumps(rec_product, indent=2)}
        
        Feature importance weights in making this recommendation:
        {json.dumps(weights, indent=2)}
        
        Based on this information, provide a concise, natural-sounding explanation (3-5 sentences) of why Product A is being recommended to someone interested in Product B. Focus on the most important factors that led to this recommendation.
        """
        
        # Call LLM API
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 256
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response_data = response.json()
            
            explanation = response_data['choices'][0]['message']['content'].strip()
            return explanation
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "We recommended this product based on its similarity to your interests."

    def generate_batch_explanations(self, original_product, recommendations_df, feature_weights_list):
        """Generate explanations for a batch of recommendations"""
        explanations = []
        
        for i, row in recommendations_df.iterrows():
            explanation = self.generate_explanation(
                original_product,
                row,
                feature_weights_list[i]
            )
            explanations.append(explanation)
        
        recommendations_df['explanation'] = explanations
        return recommendations_df




