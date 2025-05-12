# feature_engineer.py


# Feature Engineering Module

# %%


from sklearn.preprocessing import MultiLabelBinarizer
import ast
import pandas as pd

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.mlb = None
        self.category_columns = None 

    def parse_aspects(self, aspects_str):
        try:
            aspects = ast.literal_eval(aspects_str)
            return aspects if isinstance(aspects, list) else []
        except Exception:
            return []
            
    def engineer_features(self, df):
        # Process aspects
        df['aspects'] = df['aspects'].apply(self.parse_aspects)
        
        # Create category features and cast to float
        category_dummies = pd.get_dummies(df['main_category'], prefix='cat').astype(float)
        self.category_columns = category_dummies.columns
        
        # Process aspects with MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        aspects_matrix = self.mlb.fit_transform(df['aspects'])
        
        # Extract numerical features
        numerical_features = df[['score', 'rating', 'price']].fillna(0)
        
        return {
            'category_features': category_dummies,
            'aspects_matrix': aspects_matrix,
            'numerical_features': numerical_features
        }
    
    def transform_features(self, df):
        category_dummies = pd.get_dummies(df['main_category'], prefix='cat')
        if self.category_columns is not None:
            category_dummies = category_dummies.reindex(columns=self.category_columns, fill_value=0)
        # Cast to float
        category_dummies = category_dummies.astype(float)
        
        aspects_matrix = self.mlb.transform(df['aspects'].apply(self.parse_aspects))
        numerical_features = df[['score', 'rating', 'price']].fillna(0)
        
        return {
            'category_features': category_dummies,
            'aspects_matrix': aspects_matrix,
            'numerical_features': numerical_features
        }



