"""
embeddings.py
=============

Sentence-embedding utility supporting three back-ends:

* **FastText**  (平均埋め込みのみ・指定パスでロード)
* **GPT-2**      (好きな層 & last_token / average)
* **meta-llama/Meta-Llama-3-8B** (同上)

依存:
    pip install torch transformers accelerate fasttext numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import fasttext


# -------------------------------------------------------------------------
# 型エイリアス
# -------------------------------------------------------------------------
_METHOD = Literal["last_token", "average"]
_MODEL = Literal["gpt2", "meta-llama/Meta-Llama-3-8B", "fasttext"]


# -------------------------------------------------------------------------
# 設定データクラス
# -------------------------------------------------------------------------
@dataclass(slots=True)
class EmbeddingConfig:
    """設定オブジェクト"""

    model_type: _MODEL = "gpt2"
    layer: int = -1
    method: _METHOD = "average"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: Optional[str] = None  # HuggingFace リポ ID または FastText パス

    # FastText 用デフォルト
    fasttext_path: str = (
        "/work03/masaki/model/fastText_vec/wiki-news-300d-1M-subword.bin"
    )

    def __post_init__(self) -> None:  # noqa: D401
        """FastText は平均固定に強制."""
        if self.model_type == "fasttext":
            object.__setattr__(self, "method", "average")


# -------------------------------------------------------------------------
# 本体
# -------------------------------------------------------------------------
class EmbeddingModel:
    """文埋め込み取得をシンプルに行うラッパ."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

        if cfg.model_type == "fasttext":
            path = cfg.model_name or cfg.fasttext_path
            self._ft_model = fasttext.load_model(path)
            self._dim = self._ft_model.get_dimension()
            return

        # ---------------- Transformer 系 -----------------
        repo_id = cfg.model_name or cfg.model_type  # 文字列そのまま渡す
        self._tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        added_tokens = 0
        if self._tokenizer.pad_token is None:
            eos = self._tokenizer.eos_token or self._tokenizer.sep_token or "</s>"
            self._tokenizer.add_special_tokens({"pad_token": eos})
            added_tokens = 1  
                
        hf_cfg = AutoConfig.from_pretrained(repo_id, output_hidden_states=True)

        self._model = AutoModel.from_pretrained(
            repo_id,
            config=hf_cfg,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(cfg.device)

        if added_tokens:
            self._model.resize_token_embeddings(len(self._tokenizer))

        self._model.eval()
        self._dim = hf_cfg.hidden_size

    # -----------------------------------------------------------------
    # 公開 API
    # -----------------------------------------------------------------
    @torch.no_grad()
    def encode(self, sentences: List[str]) -> np.ndarray:
        """複数文を埋め込んで ndarray (N, dim) を返す."""
        if self.cfg.model_type == "fasttext":
            return np.stack([self._encode_fasttext(s) for s in sentences])

        return self._encode_transformer(sentences)

    @property
    def dim(self) -> int:
        """埋め込み次元."""
        return self._dim

    # -----------------------------------------------------------------
    # 内部実装 (FastText)
    # -----------------------------------------------------------------
    def _encode_fasttext(self, sentence: str) -> np.ndarray:
        words = sentence.split()
        if not words:
            return np.zeros(self._dim, dtype=np.float32)
        vecs = [self._ft_model.get_word_vector(w) for w in words]
        return np.mean(vecs, axis=0).astype(np.float32)

    # -----------------------------------------------------------------
    # 内部実装 (Transformer)
    # -----------------------------------------------------------------
    def _encode_transformer(self, sentences: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.cfg.device)

        out = self._model(**toks)
        hiddens = out.hidden_states  # tuple(layer) -> (B,T,D)

        layer_idx = self.cfg.layer
        hidden = hiddens[layer_idx]  # (B,T,D)
        mask = toks.attention_mask.bool()  # (B,T)

        if self.cfg.method == "average":
            summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            sent_emb = summed / counts
        else:  # last_token
            idx = mask.sum(dim=1) - 1
            sent_emb = hidden[torch.arange(hidden.size(0)), idx]

        return sent_emb.cpu().float().numpy()


# -------------------------------------------------------------------------
# エクスポート
# -------------------------------------------------------------------------
__all__ = ["EmbeddingConfig", "EmbeddingModel"]
