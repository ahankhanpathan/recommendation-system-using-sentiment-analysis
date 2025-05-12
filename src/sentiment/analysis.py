#Sentiment_analysis 
# src/sentiment/analysis.py

import pandas as pd
import logging
import json
from tqdm import tqdm
from langchain import PromptTemplate, LLMChain
from src.sentiment.llm import GroqLLM, parse_llm_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_sentiments(df: pd.DataFrame,
                        col_sentiment: str = "sentiment",
                        col_score: str = "score") -> dict:
    """Quickly evaluate sentiment distribution."""
    dist = df[col_sentiment].value_counts(normalize=True).to_dict()
    logging.info("Sentiment distribution:")
    for k, v in dist.items():
        logging.info(f"  {k:<8}: {v*100:5.1f}%")
    low_conf = (df[col_score] < 0.5).sum()
    logging.info(f"Low-confidence (<0.5): {low_conf}/{len(df)}")
    return dist

def analyze_sentiment_and_update_csv(csv_path: str,
                                     start_row: int,
                                     end_row: int,
                                     model_name: str,
                                     save_path: str,
                                     batch_size: int = 50,
                                     checkpoint_interval: int = 100):
    """
    Analyze sentiments over a CSV file between start_row and end_row
    and update the file with sentiment fields.
    """

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df.fillna("", inplace=True)
    work = df.copy()

    # Ensure required sentiment columns exist
    for col, default in (("sentiment", ""),
                         ("score", 0.0),
                         ("aspects", ""),
                         ("raw_response", "")):
        if col not in work.columns:
            work[col] = default

    # Set up the PromptTemplate and LLM
    prompt_tmpl = """
Analyze the sentiment of the following review. Return JSON with:
- sentiment: one of ["positive","neutral","negative"]
- score: float 0-1
- aspects: list of keywords

Review: \"\"\"{text}\"\"\"
"""
    prompt = PromptTemplate(template=prompt_tmpl, input_variables=["text"])
    llm    = GroqLLM(model=model_name)
    chain  = LLMChain(prompt=prompt, llm=llm)

    logging.info(f"Starting sentiment analysis from row {start_row} to {end_row}...")

    with tqdm(total=end_row - start_row, desc="reviews") as bar:
        for i in range(start_row, end_row):
            if i >= len(work):
                break

            review = (
                work.at[i, "combined_text"]
                if "combined_text" in work.columns
                else f"{work.at[i,'title']} {work.at[i,'text']}"
            ).strip()

            if not review:
                work.loc[i, ["sentiment", "score", "aspects"]] = ["unknown", 0.0, json.dumps(["empty"])]
                bar.update(1)
                continue

            try:
                response = chain.invoke({"text": review})
                content  = response["text"] if isinstance(response, dict) else response
                parsed   = parse_llm_response(content)

                work.loc[i, "raw_response"] = content
                work.loc[i, "sentiment"]    = parsed["sentiment"]
                work.loc[i, "score"]        = parsed["score"]
                work.loc[i, "aspects"]      = json.dumps(parsed["aspects"])

            except Exception as e:
                logging.warning(f"Row {i} failed: {e}")
                work.loc[i, ["sentiment", "score", "aspects"]] = ["unknown", 0.0, json.dumps(["error"])]

            bar.update(1)

            if (i + 1) % checkpoint_interval == 0:
                work.to_csv(save_path, index=False)
                logging.info(f"Checkpoint saved at row {i}")

    work.to_csv(save_path, index=False)
    logging.info(f"✅ Sentiment analysis completed. Final file saved to {save_path}")

    return evaluate_sentiments(work.iloc[start_row:end_row])
