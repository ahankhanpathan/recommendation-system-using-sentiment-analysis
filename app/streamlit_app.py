"""Streamlit UI for the Hybrid Product Recommendation System."""

import ast
import os
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Workaround for Streamlit × PyTorch file-watcher conflict
torch._classes = {}  # type: ignore[attr-defined]

from src.recommendation.engine import RecommendationEngine
from src.utils.io import load_trained_model

# --------------------------------------------------------------------------- #
# Page config (must be first Streamlit call)                                   #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "recommendation_example1.csv"


def _resolve_model_dir() -> Path:
    env_val = os.environ.get("MODEL_DIR")
    if env_val:
        return Path(env_val)
    models_root = BASE_DIR / "models"
    if models_root.exists():
        candidates = sorted(
            [d for d in models_root.iterdir() if d.is_dir() and (d / "hybrid_model.pt").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        "No trained model found. Set the MODEL_DIR environment variable or train first."
    )


def _stars(rating: float) -> str:
    """Return a star string like ★★★★☆ for a 1-5 rating."""
    try:
        r = round(float(rating))
    except (ValueError, TypeError):
        return "N/A"
    return "★" * r + "☆" * (5 - r)


def _sentiment_badge(sentiment: str) -> str:
    colours = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
    label = str(sentiment).lower()
    return f"{colours.get(label, '⚪')} {label.capitalize()}"


def _parse_aspects(aspects_val) -> list:
    if not aspects_val or (isinstance(aspects_val, float) and np.isnan(aspects_val)):
        return []
    try:
        parsed = ast.literal_eval(str(aspects_val))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _format_price(price) -> str:
    try:
        return f"${float(price):.2f}"
    except (ValueError, TypeError):
        return str(price) if price else "N/A"


# --------------------------------------------------------------------------- #
# Cached resources                                                              #
# --------------------------------------------------------------------------- #

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Normalise column name
    if "main_title" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"main_title": "title"})
    return df


@st.cache_resource
def load_engine() -> RecommendationEngine:
    model_dir = _resolve_model_dir()
    model, gnn, fe, embedder, graph_builder = load_trained_model(str(model_dir))
    return RecommendationEngine(model, gnn, fe, embedder, graph_builder, fused_feature_dim=128)


# --------------------------------------------------------------------------- #
# UI Components                                                                #
# --------------------------------------------------------------------------- #

def render_product_card(row: pd.Series, title_col: str, show_score: bool = False) -> None:
    """Render a single product as a styled expander card."""
    title = row.get(title_col, "Unknown")
    rating = row.get("rating", None)
    price = row.get("price", None)
    category = row.get("main_category", "")
    description = str(row.get("description", ""))
    sentiment = row.get("sentiment", "")
    aspects = _parse_aspects(row.get("aspects", ""))
    sim_score = row.get("similarity_score", None)
    weighted_score = row.get("weighted_score", None)

    with st.expander(f"**{title}**  —  {_stars(rating)}  {category}", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            if description:
                st.markdown(description[:300] + ("..." if len(description) > 300 else ""))
            if aspects:
                tags = "  ".join(f"`{a}`" for a in aspects[:10])
                st.markdown(f"**Aspects:** {tags}")
        with col2:
            st.metric("Price", _format_price(price))
            st.metric("Rating", f"{rating} / 5" if rating else "N/A")
            if sentiment:
                st.markdown(f"**Sentiment:** {_sentiment_badge(sentiment)}")
            if show_score and sim_score is not None:
                st.metric("Similarity", f"{sim_score:.3f}")
            if show_score and weighted_score is not None:
                st.metric("Weighted Score", f"{weighted_score:.3f}")


def render_feature_importance(engine: RecommendationEngine) -> None:
    """Show a horizontal bar chart of model feature attention weights."""
    try:
        importance = engine.hybrid.get_feature_importance()
        names = list(importance.keys())
        values = list(importance.values())

        fig, ax = plt.subplots(figsize=(5, 2.5))
        colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
        bars = ax.barh(names, values, color=colours)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_xlabel("Average attention weight")
        ax.set_title("Feature Importance", fontsize=10)
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.caption(f"Feature importance unavailable: {e}")


# --------------------------------------------------------------------------- #
# Main app                                                                     #
# --------------------------------------------------------------------------- #

try:
    df = load_data()
    engine = load_engine()
    model_loaded = True
except FileNotFoundError as exc:
    model_loaded = False
    st.error(str(exc))

title_col = "title" if "title" in df.columns else df.columns[0]
categories = sorted(df["main_category"].dropna().unique().tolist()) if model_loaded else []

# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("🛍️ Product Recommender")
    st.caption("Powered by GNN + Transformer + Attention Fusion")
    st.markdown("---")

    st.subheader("Filters")
    selected_categories = st.multiselect(
        "Category",
        options=categories,
        default=[],
        help="Leave empty to search all categories",
    )
    min_rating = st.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
    price_vals = df["price"].dropna() if model_loaded else pd.Series([0, 1000])
    try:
        price_min, price_max = float(price_vals.min()), float(price_vals.max())
    except Exception:
        price_min, price_max = 0.0, 1000.0
    price_range = st.slider(
        "Price Range ($)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=1.0,
    )
    top_n = st.slider("Number of Recommendations", min_value=3, max_value=20, value=5)

    st.markdown("---")
    st.subheader("Model Info")
    st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.caption(f"Products in catalogue: {len(df):,}")
    st.caption(f"Categories: {len(categories)}")

    if model_loaded:
        st.markdown("**Feature Importance**")
        render_feature_importance(engine)

# --------------------------------------------------------------------------- #
# Apply sidebar filters to the dataframe                                       #
# --------------------------------------------------------------------------- #

if model_loaded:
    filtered_df = df.copy()
    if selected_categories:
        filtered_df = filtered_df[filtered_df["main_category"].isin(selected_categories)]
    filtered_df = filtered_df[filtered_df["rating"].fillna(0) >= min_rating]
    try:
        filtered_df = filtered_df[
            filtered_df["price"].fillna(0).between(price_range[0], price_range[1])
        ]
    except Exception:
        pass

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} products after filters")

# --------------------------------------------------------------------------- #
# Main content — two tabs                                                      #
# --------------------------------------------------------------------------- #

if model_loaded:
    tab_similar, tab_category = st.tabs(["🔍 Find Similar Products", "🗂️ Browse by Category"])

    # ------------------------------------------------------------------ #
    # Tab 1 — similar products by keyword search                          #
    # ------------------------------------------------------------------ #
    with tab_similar:
        st.header("Find Similar Products")
        query = st.text_input("Search by product title or keyword", placeholder="e.g. wireless headphones")

        if query:
            results = filtered_df[
                filtered_df[title_col].str.contains(query, case=False, na=False)
            ]
            if results.empty:
                st.warning("No products match your query. Try a different keyword or adjust filters.")
            else:
                selected_idx = st.selectbox(
                    f"Found {len(results)} products — select one:",
                    options=results.index.tolist(),
                    format_func=lambda i: (
                        f"{df.at[i, title_col]}  |  "
                        f"{df.at[i, 'main_category']}  |  "
                        f"{_stars(df.at[i, 'rating'])}"
                    ),
                )

                st.markdown("### Selected Product")
                render_product_card(df.loc[selected_idx], title_col)

                if st.button("🚀 Get Recommendations", type="primary"):
                    with st.spinner("Generating recommendations …"):
                        try:
                            recs = engine.recommend_similar_products(
                                selected_idx, df, top_n=top_n
                            )
                            # Apply price/rating filters to results
                            recs = recs[recs["rating"].fillna(0) >= min_rating]

                            st.markdown(f"### Top {len(recs)} Similar Products")
                            for _, row in recs.iterrows():
                                render_product_card(row, title_col, show_score=True)

                        except Exception as exc:
                            st.error(f"Recommendation failed: {exc}")
        else:
            st.info("Enter a keyword above to search the product catalogue.")

    # ------------------------------------------------------------------ #
    # Tab 2 — browse entire category                                       #
    # ------------------------------------------------------------------ #
    with tab_category:
        st.header("Browse by Category")

        if not categories:
            st.warning("No categories found in the data.")
        else:
            chosen_cat = st.selectbox("Choose a category:", options=categories)

            col_info, col_btn = st.columns([3, 1])
            with col_info:
                cat_count = (df["main_category"] == chosen_cat).sum()
                st.caption(f"{cat_count:,} products in **{chosen_cat}**")
            with col_btn:
                run_cat = st.button("🔎 Show Top Products", type="primary")

            if run_cat:
                with st.spinner(f"Finding top products in '{chosen_cat}' …"):
                    try:
                        cat_recs = engine.recommend_by_category(chosen_cat, df, top_n=top_n)
                        if cat_recs.empty:
                            st.warning(f"No recommendations found for '{chosen_cat}'.")
                        else:
                            st.markdown(f"### Top {len(cat_recs)} in {chosen_cat}")

                            # Quick summary bar chart of sentiment distribution
                            if "sentiment" in cat_recs.columns:
                                sent_counts = cat_recs["sentiment"].value_counts()
                                fig, ax = plt.subplots(figsize=(4, 1.8))
                                colours_map = {
                                    "positive": "#59a14f",
                                    "neutral": "#f28e2b",
                                    "negative": "#e15759",
                                }
                                ax.bar(
                                    sent_counts.index,
                                    sent_counts.values,
                                    color=[colours_map.get(s, "#aaa") for s in sent_counts.index],
                                )
                                ax.set_title("Sentiment in Results", fontsize=9)
                                ax.set_ylabel("Count")
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                            for _, row in cat_recs.iterrows():
                                render_product_card(row, title_col, show_score=True)
                    except Exception as exc:
                        st.error(f"Category recommendation failed: {exc}")
