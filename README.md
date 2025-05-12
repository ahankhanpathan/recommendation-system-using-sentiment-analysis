# recommendation-system-using-sentiment-analysis

Here’s a suggested `README.md` description for your repo based on the codebase:

---

# 🧠 Hybrid Product Recommendation System with Sentiment & GNN

This repository contains a **hybrid product recommendation engine** that integrates **transformer-based embeddings**, **graph neural networks (GNNs)**, **aspect-based sentiment analysis**, and an **attention fusion model** to deliver personalized and explainable product recommendations.

## 🚀 Features

* **Aspect-Based Sentiment Analysis** using LLMs (Groq LLaMA 3.0).
* **Transformer embeddings** (e.g., MiniLM) for product descriptions.
* **Graph Neural Network (GNN)** encoding product relationships based on cosine similarity.
* **Attention-based fusion model** combining text, graph, category, numerical, and aspect features.
* **Streamlit UI** for interactive product search and recommendation.
* **Explainable recommendations** with natural language rationale for suggestions.

## 📦 Project Structure

```
.
├── streamlit_app.py               # Streamlit frontend
├── run_recommendation.py         # CLI entry-point for train/evaluate/recommend
├── sentiment_batch.py            # Batch sentiment analysis script
├── config.yaml                   # Config for models and paths
├── requirements.txt              # Dependencies
├── models/                       # Trained models (hybrid + GNN)
├── data/                         # Raw and processed datasets
└── src/
    ├── recommendation/           # Core recommendation engine
    ├── sentiment/                # Sentiment analysis pipeline
    └── utils/                    # Utilities (e.g., model loading)
```

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python run_recommendation.py --config config.yaml --data data/processed/input.csv --mode train
```

### 3. Run sentiment analysis

```bash
python sentiment_batch.py --csv_path data/raw/reviews.csv --save_path data/processed/sentiment.csv
```

### 4. Launch Streamlit app

```bash
streamlit run streamlit_app.py
```

## 📊 Evaluation

Supports `precision@k`, `recall@k`, `NDCG`, `F1`, and **catalog coverage** for both collaborative and content-based metrics. Visual plots included via `matplotlib`.

## 📚 Tech Stack

* **Transformers**: Sentence-BERT
* **GNN**: GCN/GAT via PyTorch Geometric
* **Fusion Model**: Attention-based module
* **Sentiment LLM**: Groq API (LLaMA3)
* **UI**: Streamlit
* **Eval**: scikit-learn, matplotlib
