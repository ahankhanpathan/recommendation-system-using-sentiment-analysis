# transformer_embedder.py



# # Text Embedding and Simlairty score

# %%


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel # Ensure these are imported here or earlier

# Moved TextDataset class definition outside TransformerEmbedder
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }




#  # Embedding Generation Module

# %%


from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader


class TransformerEmbedder:
    def __init__(self, config):
        self.config = config
        self.model_name = config.get('transformer_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.max_length = config.get('max_length', 128)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_column = config.get('text_column', 'description')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, df):
        """Get embeddings for texts in the dataframe"""
        print(f"Generating text embeddings from column: {self.text_column}")
        
        # Check if text column exists
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataframe. Available columns: {df.columns}")
        
        texts = df[self.text_column].tolist()
        return self.generate_embeddings(texts)
    
    def generate_embeddings(self, texts, batch_size=16):
        """Generate embeddings for a list of texts"""
        print(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
        
        # Create a simple dataset
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, max_length=128):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text, 
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
        
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_embeddings = self.mean_pooling(outputs, attention_mask)
                embeddings.append(batch_embeddings.cpu())
        
        all_embeddings = torch.cat(embeddings, dim=0).numpy()
        print(f"Generated embeddings with shape: {all_embeddings.shape}")
        return all_embeddings
    
    def save_embeddings(self, embeddings, output_path):
        """Save embeddings to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Embeddings saved to {output_path}")

