"""Data loading and preprocessing utilities."""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataProcessor:
    """Loads a product CSV, applies preprocessing, and provides train/test splits."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess the product dataset.

        - Parses timestamps.
        - Negates sentiment scores for negative reviews.
        - Fills common missing values.
        - Creates the ``all_text`` column used by the embedder.
        """
        logger.info("Loading data from %s", filepath)
        df = pd.read_csv(filepath)

        # Timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Sentiment-weighted score
        if "score" in df.columns and "sentiment" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            neg_mask = df["sentiment"].str.lower() == "negative"
            df.loc[neg_mask, "score"] = -1 * df.loc[neg_mask, "score"].abs()

        # Fill common missing values
        df["main_category"] = df.get("main_category", pd.Series("unknown")).fillna("unknown")
        df["title"] = df.get("title", pd.Series("unknown")).fillna("unknown")
        df["description"] = df.get("description", pd.Series("")).fillna("")

        # Build the combined text field used for embedding generation
        text_parts = [df["title"], df["description"]]
        if "text" in df.columns:
            text_parts.append(df["text"].fillna(""))
        df["all_text"] = pd.concat(text_parts, axis=1).apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )

        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """Return (train_df, test_df) with reset indices."""
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the processed dataframe to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Processed data saved to %s", output_path)
