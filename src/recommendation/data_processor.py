# # Data Processing Pipeline



import pandas as pd
import numpy as np
import torch
import os
import json
from datetime import datetime

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_data(self, filepath):
        """Load and preprocess the dataset"""
        df = pd.read_csv(filepath)
        
        # Apply existing preprocessing steps from your code
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.loc[df['sentiment'].str.lower() == 'negative', 'score'] = -1 * df.loc[df['sentiment'].str.lower() == 'negative', 'score']
        
        # Fill missing values
        df['main_category'] = df['main_category'].fillna("unknown")
        df['title'] = df['title'].fillna("unknown")
        df['description'] = df['description'].fillna("")
        
        # Create combined text fields for embedding generation
        df['all_text'] = df['title'] + " " + df['description'] + " " + df['text']
        
        return df
    
    def create_train_test_split(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df
    
    def save_processed_data(self, df, output_path):
        """Save processed dataframe"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

