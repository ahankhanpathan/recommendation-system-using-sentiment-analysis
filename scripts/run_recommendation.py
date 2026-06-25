"""CLI entry-point for training, evaluating, and running recommendations."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.recommendation.data_processor import DataProcessor
from src.recommendation.engine import RecommendationEngine, build_engine_from_trained_folder
from src.recommendation.evaluator import ExplainabilityModule, RecommendationEvaluator
from src.recommendation.feature_engineer import FeatureEngineer
from src.recommendation.graph_builder import GraphBuilder
from src.recommendation.models import GNNModel, HybridRecommender
from src.recommendation.training_pipeline import TrainingPipeline
from src.recommendation.transformer_embedder import TransformerEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str, data_path: str, mode: str = "train", output_dir: str = "output") -> None:
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    data_processor = DataProcessor(config)
    logger.info("Loading and preprocessing data …")
    df = data_processor.load_data(data_path)

    if mode == "train":
        feature_engineer = FeatureEngineer(config)
        transformer_embedder = TransformerEmbedder(config["transformer"])
        graph_builder = GraphBuilder(config["gnn"])

        train_df, test_df = data_processor.create_train_test_split(df)

        training_pipeline = TrainingPipeline(config)
        train_data, test_data, gnn = training_pipeline.prepare_training_data(
            pd.concat([train_df, test_df]).reset_index(drop=True),
            train_df.index,
            test_df.index,
            feature_engineer,
            transformer_embedder,
            graph_builder,
        )

        hybrid_cfg = config.get("hybrid", {})
        batch_size: int = hybrid_cfg.get("batch_size", config.get("batch_size", 32))
        epochs: int = hybrid_cfg.get("epochs", config.get("epochs", 10))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        # Infer dims from the first batch
        sample = train_data[0]
        embedding_dim, gnn_dim, category_dim, numerical_dim, aspects_dim = (
            sample[0].shape[0],
            sample[1].shape[0],
            sample[2].shape[0],
            sample[3].shape[0],
            sample[4].shape[0],
        )

        hybrid_model = HybridRecommender(
            config,
            embedding_dim=embedding_dim,
            gnn_dim=gnn_dim,
            category_dim=category_dim,
            numerical_dim=numerical_dim,
            aspects_dim=aspects_dim,
        )

        logger.info(
            "Training model — %d trainable parameters",
            sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad),
        )
        hybrid_model = training_pipeline.train_model(hybrid_model, train_loader, test_loader, epochs=epochs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_pipeline.save_model(hybrid_model, gnn, feature_engineer, timestamp=timestamp)
        training_pipeline.plot_training_history(
            output_path=os.path.join(output_dir, f"training_history_{timestamp}.png")
        )

        evaluator = RecommendationEvaluator(config)
        metrics = evaluator.evaluate_model(hybrid_model, test_loader, len(test_df))
        evaluator.save_metrics(os.path.join(output_dir, f"metrics_{timestamp}.json"))

    elif mode == "evaluate":
        model_path = config.get("paths", {}).get("model_path")
        if not model_path:
            raise ValueError("'paths.model_path' is missing in config.")

        engine = build_engine_from_trained_folder(model_path)
        evaluator = RecommendationEvaluator(config)
        evaluator.evaluate_similarity_scorer(engine, df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluator.save_metrics(os.path.join(output_dir, f"similarity_metrics_{timestamp}.json"))

    elif mode == "recommend":
        model_path = config.get("paths", {}).get("model_path")
        if not model_path:
            raise ValueError("'paths.model_path' is missing in config.")

        engine = build_engine_from_trained_folder(model_path)
        explainability = ExplainabilityModule(config.get("explainability", {}))

        rec_cfg = config.get("recommendation", {})
        top_n: int = rec_cfg.get("top_n", 10)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if rec_cfg.get("product_id") is not None:
            product_idx = int(rec_cfg["product_id"])
            recommendations = engine.recommend_similar_products(product_idx, df, top_n=top_n)

            product_data = df.iloc[product_idx]
            feature_weights_list = [engine.hybrid.get_feature_importance()] * len(recommendations)
            recommendations = explainability.generate_batch_explanations(
                product_data, recommendations, feature_weights_list
            )

            out_csv = os.path.join(output_dir, f"recommendations_{timestamp}.csv")
            recommendations.to_csv(out_csv, index=False)
            logger.info("Recommendations saved to %s", out_csv)

        elif rec_cfg.get("category"):
            category = rec_cfg["category"]
            recommendations = engine.recommend_by_category(category, df, top_n=top_n)
            out_csv = os.path.join(output_dir, f"category_recommendations_{timestamp}.csv")
            recommendations.to_csv(out_csv, index=False)
            logger.info("Category recommendations saved to %s", out_csv)

        else:
            logger.warning(
                "No 'product_id' or 'category' specified in config under 'recommendation'."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Recommendation System")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "recommend"],
        default="train",
    )
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.config, args.data, args.mode, args.output)


if __name__ == "__main__":
    main()
