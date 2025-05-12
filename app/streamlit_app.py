import os
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
from functools import lru_cache
# 🔧 Fix for Streamlit x PyTorch watcher bug
torch._classes = {}

# 🔧 Import your modules
from src.utils.io import load_trained_model
from src.recommendation.engine import RecommendationEngine

# ---------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "hybrid_recommender_20250428_103456"
DATA_FILE = BASE_DIR / "data" / "processed" / "recommendation_example1.csv"
CFG_JSON = MODEL_DIR / "config.json"

# Load data
df = pd.read_csv(DATA_FILE)

# Load model and pipeline components
model, gnn, feature_engineer, embedder, graph_builder = load_trained_model(str(MODEL_DIR))

# Create recommendation engine
engine = RecommendationEngine(
    model,
    gnn,
    feature_engineer,
    embedder,
    graph_builder,
    fused_feature_dim=128  #  keyword-only argument
)

# ---------------------------------------------------------------------
# One-shot catalog embedding caching
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_all_reps():
    return engine.get_product_representations(df)

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.title("Product Recommender System")

query = st.text_input("Search products by title or keyword")

if query:
    filtered_df = df[df['main_title'].str.contains(query, case=False, na=False)]
    if not filtered_df.empty:
        selected_idx = st.selectbox(
            "Select a product:",
            options=filtered_df.index,
            format_func=lambda idx: f"{df.at[idx, 'main_title']} ({df.at[idx, 'main_category']})"
        )

        if st.button("Recommend Similar Products"):
            st.success(f"Recommending products similar to: {df.at[selected_idx, 'main_title']}")

            # Show Selected Product Information
            st.subheader("Selected Product Details")
            st.markdown(f"**Title:** {df.at[selected_idx, 'main_title']}")
            st.markdown(f"**Category:** {df.at[selected_idx, 'main_category']}")
            st.markdown(f"**Rating:** {df.at[selected_idx, 'rating']}")
            st.markdown(f"**Price:** {df.at[selected_idx, 'price']}")
            st.markdown(f"**Description:** {df.at[selected_idx, 'description'][:300]}...")
            st.markdown("---")
            # # Display aspects (parsed from string to list)
            # try:
            #     import ast
            #     aspects_list = ast.literal_eval(df.at[selected_idx, 'aspects'])
            # except:
            #     aspects_list = []

            # # Display aspects cleanly
            # if aspects_list:
            #     st.markdown("**Aspects:**")
            #     for aspect in aspects_list:
            #         st.markdown(f"- {aspect}")
            # st.markdown(f"**Aspects:** {', '.join(aspects_list)}")
            # st.markdown("---")

            # Generate Recommendations
            reps = get_all_reps()
            recommendations = engine.recommend_similar_products(selected_idx, df, top_n=5)

            st.subheader("Top Recommendations:")
            for _, row in recommendations.iterrows():
                st.markdown(f"**{row['main_title']}**")
                st.markdown(f" Category: {row['main_category']}")
                st.markdown(f" Rating: {row['rating']}")
                st.markdown(f"Price: {row['price']}")
                desc = str(df.at[selected_idx, 'description']) if pd.notna(df.at[selected_idx, 'description']) else "No description available."
                st.markdown(f" Description: {row['description'][:200]}...")
                # try:
                #     aspects_list = ast.literal_eval(row['aspects'])
                # except:
                #     aspects_list = []
                # st.markdown(f"**Aspects:** {', '.join(aspects_list)}")
                # st.markdown("---")



            



    else:
        st.warning("No products found for your query. Please try a different keyword.")

else:
    st.info("Please enter a search keyword to begin!")
