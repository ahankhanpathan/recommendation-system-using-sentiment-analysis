 # run_recommendation.py

# scripts/run_recommendation.py

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import os
import json
import sys
from datetime import datetime
from torch.utils.data import DataLoader

from src.recommendation.data_processor import DataProcessor
from src.recommendation.feature_engineer import FeatureEngineer
from src.recommendation.graph_builder import GraphBuilder
from src.recommendation.transformer_embedder import TransformerEmbedder
from src.recommendation.training_pipeline import TrainingPipeline
from src.recommendation.models import HybridRecommender
from src.recommendation.models import GNNModel
from src.recommendation.engine import RecommendationEngine
from src.recommendation.evaluator import RecommendationEvaluator, ExplainabilityModule

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_components(model_path, config):
    import pickle

    embedding_dim = 384  # Assume MiniLM
    gnn_dim = config['gnn']['output_dim']

    try:
        with open(os.path.join(model_path, "feature_engineer.pkl"), 'rb') as f:
            feature_engineer = pickle.load(f)
    except FileNotFoundError:
        print("feature_engineer.pkl not found. Initializing new FeatureEngineer.")
        feature_engineer = FeatureEngineer(config)

    category_dim = len(feature_engineer.category_columns) if hasattr(feature_engineer, 'category_columns') else config.get('category_dim')
    numerical_dim = 3  # Assuming score, rating, price
    aspects_dim = len(feature_engineer.mlb.classes_) if hasattr(feature_engineer, 'mlb') else config.get('aspects_dim')

    if None in [category_dim, numerical_dim, aspects_dim]:
        raise ValueError("Category/Numerical/Aspect dimensions could not be determined.")

    hybrid_model = HybridRecommender(
        config,
        embedding_dim=embedding_dim,
        gnn_dim=gnn_dim,
        category_dim=category_dim,
        numerical_dim=numerical_dim,
        aspects_dim=aspects_dim,
    )
    hybrid_model.load_state_dict(torch.load(os.path.join(model_path, "hybrid_model.pt")))
    hybrid_model.eval()

    gnn_model = GNNModel(config['gnn'], input_dim=embedding_dim)
    gnn_model.load_state_dict(torch.load(os.path.join(model_path, "gnn_model.pt")))
    gnn_model.eval()

    return hybrid_model, gnn_model, feature_engineer

def run_pipeline(config_path, data_path, mode='train', output_dir='output'):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    data_processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)
    transformer_embedder = TransformerEmbedder(config)
    graph_builder = GraphBuilder(config)

    print("Loading and preprocessing data...")
    df = data_processor.load_data(data_path)

    fused_feature_dim = config.get('fusion_dim', 128)

    if mode == 'train':
        train_df, test_df = data_processor.create_train_test_split(df)
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        train_indices, test_indices = train_df.index, test_df.index

        training_pipeline = TrainingPipeline(config)
        train_data, test_data, gnn = training_pipeline.prepare_training_data(
            df, train_indices, test_indices, feature_engineer, transformer_embedder, graph_builder
        )

        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        embedding_dim = train_data[0][0].shape[0]
        gnn_dim = train_data[0][1].shape[0]
        category_dim = train_data[0][2].shape[0]
        numerical_dim = train_data[0][3].shape[0]
        aspects_dim = train_data[0][4].shape[0]

        hybrid_model = HybridRecommender(
            config,
            embedding_dim=embedding_dim,
            gnn_dim=gnn_dim,
            category_dim=category_dim,
            numerical_dim=numerical_dim,
            aspects_dim=aspects_dim,
        )

        print(f"Training model with {sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)} trainable parameters.")

        epochs = config.get('epochs', 10)
        hybrid_model = training_pipeline.train_model(hybrid_model, train_loader, test_loader, epochs=epochs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_pipeline.save_model(hybrid_model, gnn, feature_engineer, timestamp=timestamp)

        history_plot = os.path.join(output_dir, f"training_history_{timestamp}.png")
        training_pipeline.plot_training_history(output_path=history_plot)

        evaluator = RecommendationEvaluator(config)
        metrics = evaluator.evaluate_model(hybrid_model, test_loader, len(test_df))

        metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
        evaluator.save_metrics(metrics_path)

    elif mode == 'evaluate':
        model_path = config['paths']['model_path']
        if not model_path:
            raise ValueError("model_path missing in config.")

        hybrid_model, gnn, feature_engineer = load_model_components(model_path, config)
        recommendation_engine = RecommendationEngine(
            hybrid_model, gnn, data_processor, feature_engineer, transformer_embedder, graph_builder, fused_feature_dim
        )

        evaluator = RecommendationEvaluator(config)
        similarity_metrics = evaluator.evaluate_similarity_scorer(recommendation_engine, df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_metrics_path = os.path.join(output_dir, f"similarity_metrics_{timestamp}.json")
        evaluator.save_metrics(sim_metrics_path)

    elif mode == 'recommend':
        model_path = config.get('model_path')
        if not model_path:
            raise ValueError("model_path missing in config.")

        hybrid_model, gnn, feature_engineer = load_model_components(model_path, config)
        recommendation_engine = RecommendationEngine(
            hybrid_model, gnn, data_processor, feature_engineer, transformer_embedder, graph_builder, fused_feature_dim
        )
        explainability = ExplainabilityModule(config)

        if 'product_id' in config:
            product_idx = int(config['product_id'])
            recommendations = recommendation_engine.recommend_similar_products(product_idx, df, top_n=config.get('top_n', 10))

            product_data = df.iloc[product_idx]
            feature_weights_list = [hybrid_model.get_feature_importance()] * len(recommendations)

            recommendations_with_explanations = explainability.generate_batch_explanations(
                product_data, recommendations, feature_weights_list
            )

            output_csv = os.path.join(output_dir, f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            recommendations_with_explanations.to_csv(output_csv, index=False)
            print(f"Recommendations saved to {output_csv}")

        elif 'category' in config:
            category = config['category']
            recommendations = recommendation_engine.recommend_by_category(category, df, top_n=config.get('top_n', 10))
            output_csv = os.path.join(output_dir, f"category_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            recommendations.to_csv(output_csv, index=False)
            print(f"Category recommendations saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Recommendation System Runner')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML')
    parser.add_argument('--data', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'recommend'], default='train', help='Pipeline mode')
    parser.add_argument('--output', type=str, default='output', help='Directory for outputs')
    args = parser.parse_args()

    run_pipeline(args.config, args.data, args.mode, args.output)

if __name__ == "__main__":
    main()
