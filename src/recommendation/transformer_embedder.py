"""Sentence-transformer wrapper for generating product text embeddings."""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class _TextDataset(Dataset):
    """Internal dataset that tokenises a list of strings on the fly."""

    def __init__(self, texts: list, tokenizer, max_length: int = 128) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


class TransformerEmbedder:
    """
    Generates dense sentence embeddings using a Hugging Face transformer.

    Defaults to ``sentence-transformers/all-MiniLM-L6-v2`` which produces
    384-dimensional embeddings via mean-pooling over token outputs.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_name: str = config.get(
            "model_name",
            config.get("transformer_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        self.max_length: int = config.get("max_length", 128)
        self.batch_size: int = config.get("batch_size", 16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading tokeniser and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings weighted by the attention mask."""
        token_emb = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, texts: list, batch_size: int | None = None) -> np.ndarray:
        """
        Generate embeddings for a list of strings.

        Args:
            texts: List of raw text strings.
            batch_size: Override the instance-level batch size if provided.

        Returns:
            NumPy array of shape (N, embedding_dim).
        """
        bs = batch_size or self.batch_size
        logger.info("Generating embeddings for %d texts (batch_size=%d) …", len(texts), bs)

        dataset = _TextDataset(texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=bs)

        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_emb = self._mean_pooling(outputs, attention_mask)
                embeddings.append(batch_emb.cpu())

        result = torch.cat(embeddings, dim=0).numpy()
        logger.info("Embeddings shape: %s", result.shape)
        return result
