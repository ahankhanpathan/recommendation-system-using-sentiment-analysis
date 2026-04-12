"""Streamlit UI for interactive product search and recommendation."""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# Workaround for a Streamlit × PyTorch file-watcher conflict
torch._classes = {}  # type: ignore[attr-defined]

from src.utils.io import load_trained_model
from src.recommendation.engine import RecommendationEngine

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent

# Model directory: prefer the MODEL_DIR environment variable, then config,
# then fall back to the most recently modified folder in models/.
def _resolve_model_dir() -> Path:
    env_val = os.environ.get("MODEL_DIR")
    if env_val:
        return Path(env_val)

    models_root = BASE_DIR / "models"
    if models_root.exists():
        subdirs = sorted(models_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = [d for d in subdirs if d.is_dir() and (d / "hybrid_model.pt").exists()]
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        "No trained model found. Set the MODEL_DIR environment variable or train a model first."
    )


MODEL_DIR = _resolve_model_dir()
DATA_FILE = BASE_DIR / "data" / "processed" / "recommendation_example1.csv"

# --------------------------------------------------------------------------- #
# Data and model loading                                                       #
# --------------------------------------------------------------------------- #

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_engine() -> RecommendationEngine:
    model, gnn, feature_engineer, embedder, graph_builder = load_trained_model(str(MODEL_DIR))
    return RecommendationEngine(
        model,
        gnn,
        feature_engineer,
        embedder,
        graph_builder,
        fused_feature_dim=128,
    )


df = load_data()
engine = load_engine()

# --------------------------------------------------------------------------- #
# Streamlit UI                                                                 #
# --------------------------------------------------------------------------- #

st.title("Product Recommender System")

query = st.text_input("Search products by title or keyword")

if query:
    title_col = "title" if "title" in df.columns else df.columns[0]
    filtered_df = df[df[title_col].str.contains(query, case=False, na=False)]

    if not filtered_df.empty:
        selected_idx = st.selectbox(
            "Select a product:",
            options=filtered_df.index.tolist(),
            format_func=lambda idx: f"{df.at[idx, title_col]}  ({df.at[idx, 'main_category']})",
        )

        if st.button("Recommend Similar Products"):
            st.success(f"Recommending products similar to: {df.at[selected_idx, title_col]}")

            st.subheader("Selected Product Details")
            st.markdown(f"**Title:** {df.at[selected_idx, title_col]}")
            st.markdown(f"**Category:** {df.at[selected_idx, 'main_category']}")
            st.markdown(f"**Rating:** {df.at[selected_idx, 'rating']}")
            st.markdown(f"**Price:** {df.at[selected_idx, 'price']}")
            desc = str(df.at[selected_idx, "description"])
            st.markdown(f"**Description:** {desc[:300]}{'...' if len(desc) > 300 else ''}")
            st.markdown("---")

            with st.spinner("Generating recommendations …"):
                recommendations = engine.recommend_similar_products(selected_idx, df, top_n=5)

            st.subheader("Top Recommendations")
            for _, row in recommendations.iterrows():
                st.markdown(f"**{row[title_col]}**")
                st.markdown(f"Category: {row['main_category']}")
                st.markdown(f"Rating: {row['rating']}")
                st.markdown(f"Price: {row['price']}")
                row_desc = str(row.get("description", ""))
                st.markdown(f"Description: {row_desc[:200]}{'...' if len(row_desc) > 200 else ''}")
                st.markdown(f"Similarity Score: {row['similarity_score']:.3f}")
                st.markdown("---")
    else:
        st.warning("No products found for that keyword. Please try something else.")

else:
    st.info("Enter a search keyword to begin.")
