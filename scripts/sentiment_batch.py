# sentiment_batch.py

# scripts/sentiment_batch.py

import argparse
import os
from src.sentiment.analysis import analyze_sentiment_and_update_csv

def main():
    parser = argparse.ArgumentParser(description='Batch Sentiment Analysis Runner')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--start_row', type=int, default=0, help='Row to start sentiment analysis from')
    parser.add_argument('--end_row', type=int, default=None, help='Row to end sentiment analysis at (None for full)')
    parser.add_argument('--model_name', type=str, default='groq-llama3-8b-8192', help='Model to use for sentiment')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the sentiment updated CSV')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for API calls')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save progress every N rows')

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"Starting sentiment analysis from {args.csv_path}")
    analyze_sentiment_and_update_csv(
        csv_path=args.csv_path,
        start_row=args.start_row,
        end_row=args.end_row,
        model_name=args.model_name,
        save_path=args.save_path,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval
    )
    print(f" Sentiment analysis completed and saved to {args.save_path}")

if __name__ == "__main__":
    main()
