# llm.py

# src/sentiment/llm.py

import requests
import logging
import time
import random
import json
import re
from langchain.llms.base import LLM
import os

def fix_json_string(text: str) -> str:
    """Fix bad JSON from LLM."""
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    text = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', text)
    return text.replace("'", '"')

def rate_limited_api_call(url, headers, data, max_retries=5):
    """API call with exponential backoff on rate-limit."""
    retry = 0
    while retry < max_retries:
        try:
            r = requests.post(url, headers=headers, json=data, timeout=60)
            if r.status_code == 200:
                return r.json()

            if r.status_code == 429 or "rate_limit_exceeded" in r.text.lower():
                retry += 1
                wait = (2 ** retry) + random.uniform(1.0, 3.0)
                logging.info(f"Rate-limit hit; sleeping {wait:.1f}s (try {retry}/{max_retries})")
                time.sleep(wait)
                continue

            retry += 1
            wait = (2 ** retry) + random.uniform(1.0, 3.0)
            logging.warning(f"API error {r.status_code}: {r.text[:80]}… retrying in {wait:.1f}s")
            time.sleep(wait)

        except Exception as e:
            retry += 1
            wait = (2 ** retry) + random.uniform(2.0, 5.0)
            logging.warning(f"Request failed: {e}  retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries")

class GroqLLM(LLM):
    """Simple Groq LLM wrapper."""
    api_key: str = os.getenv("GROQ_API_KEY")
    model: str = "groq-llama3-8b-8192"
    max_tokens: int = 512
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "groq_llm"

    def _call(self, prompt: str, stop=None) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = rate_limited_api_call(url, headers, body)
        return response["choices"][0]["message"]["content"]

def parse_llm_response(text: str) -> dict:
    """Best-effort JSON parser."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is None:
        text_fixed = fix_json_string(text)
        try:
            data = json.loads(text_fixed)
        except Exception:
            logging.warning(f"Unable to parse: {text[:100]}")
            return {"sentiment": "unknown", "score": 0.0, "aspects": []}

    if data.get("sentiment") not in ("positive", "neutral", "negative"):
        data["sentiment"] = "unknown"

    data["score"] = float(data.get("score", 0.0))
    data["aspects"] = data["aspects"] if isinstance(data.get("aspects"), list) else []

    return data
