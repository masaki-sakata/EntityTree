# embeddings.py
"""
Sentence-embedding utility supporting four back‑ends and multi‑layer output:

* **FastText**  – average pooling only
* **GPT‑2**      – choose any hidden layer or **"all"** for every layer
* **meta‑llama/Meta‑Llama‑3‑8B** – same as GPT‑2
* **random_emb** – random Gaussian embeddings for baseline comparison

依存:
    pip install torch transformers accelerate fasttext numpy

Template Support:
    This module now works seamlessly with template-generated text.
    For template text like "What is the occupation of [entity]?", the embedding
    extraction will focus on the last token or use average pooling as specified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union
import re

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import fasttext

# -------------------------------------------------------------------------
# 型エイリアス
# -------------------------------------------------------------------------
_METHOD = Literal["last_token", "average", "entity_last_token"]
_MODEL = Literal["gpt2", "meta-llama/Meta-Llama-3-8B", "fasttext", "random_emb"]


# -------------------------------------------------------------------------
# 設定データクラス
# -------------------------------------------------------------------------
@dataclass(slots=True)
class EmbeddingConfig:
    """設定オブジェクト

    ``layer`` に整数を渡すとその層のみ、文字列 **"all"** を渡すと全層を対象に
    埋め込みを返します。
    
    Template Support:
    - method="last_token": 文全体の最後のトークン
    - method="average": 文全体の平均
    - method="entity_last_token": エンティティ名の最後のトークンのみ（テンプレート対応）
    """

    model_type: _MODEL = "gpt2"
    layer: Union[int, Literal["all"]] = "all"
    method: _METHOD = "average"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: Optional[str] = None  # HuggingFace リポ ID または FastText パス

    # FastText 用デフォルト
    fasttext_path: str = (
        "/work03/masaki/model/fastText_vec/wiki-news-300d-1M-subword.bin"
    )

    # Random embedding 用パラメータ
    random_dim: int = 768
    random_std: float = 1.0
    random_seed: int = 42

    def __post_init__(self) -> None:  # noqa: D401
        """FastText は平均のみ。entity_last_tokenは他のモデルでのみサポート。"""
        if self.model_type in ["fasttext"]:
            object.__setattr__(self, "method", "average")
        elif self.model_type == "random_emb" and self.method == "entity_last_token":
            # Random embeddingsではentity_last_tokenをaverage に変更
            object.__setattr__(self, "method", "average")


# -------------------------------------------------------------------------
# 本体
# -------------------------------------------------------------------------
class EmbeddingModel:
    """文埋め込み取得をシンプルに行うラッパ. テンプレート対応."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

        # ---------------- FastText -----------------
        if cfg.model_type == "fasttext":
            path = cfg.model_name or cfg.fasttext_path
            self._ft_model = fasttext.load_model(path)
            self._dim = self._ft_model.get_dimension()
            return

        # ---------------- Random Embeddings --------
        if cfg.model_type == "random_emb":
            self._dim = cfg.random_dim
            return

        # ---------------- Transformer 系 -----------------
        repo_id = cfg.model_name or cfg.model_type  # 文字列そのまま渡す

        self._tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        added_tokens = 0
        if self._tokenizer.pad_token is None:
            eos = self._tokenizer.eos_token or self._tokenizer.sep_token or ""
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
        """文リストを埋め込んで返す。

        戻り値の形状は次のとおり:

        * ``layer`` が "all" の場合 … ``(L, N, D)``
        * それ以外の場合           … ``(N, D)``
        """

        if self.cfg.model_type == "fasttext":
            return np.stack([self._encode_fasttext(s) for s in sentences])

        if self.cfg.model_type == "random_emb":
            return self._encode_random(sentences)

        return self._encode_transformer(sentences)

    @property
    def dim(self) -> int:
        """埋め込み次元."""
        return self._dim

    # -------------------------------------------------------------
    # その他ユーティリティ
    # -------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        """利用可能な Transformer 隠れ層数"""
        if self.cfg.model_type == "fasttext":
            return 1
        if self.cfg.model_type == "random_emb":
            return 1  # Random embeddings are single layer
        return self._model.config.num_hidden_layers

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
    # 内部実装 (Random Embeddings)
    # -----------------------------------------------------------------
    def _encode_random(self, sentences: List[str]) -> np.ndarray:
        """Generate random embeddings for sentences with specified parameters."""
        # Set seed for reproducibility
        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        
        num_sentences = len(sentences)
        
        # Generate random embeddings (single layer only)
        embeddings = torch.randn(num_sentences, self.cfg.random_dim) * self.cfg.random_std
        
        # Move to specified device if needed
        if self.cfg.device.startswith('cuda') and torch.cuda.is_available():
            embeddings = embeddings.cuda()
        
        print(f"Generated random embeddings: shape={embeddings.shape}, "
              f"std={self.cfg.random_std}, seed={self.cfg.random_seed}")
        print(f"Actual mean: {embeddings.mean().item():.4f}, "
              f"actual std: {embeddings.std().item():.4f}")
        
        return embeddings.cpu().numpy()

    # -----------------------------------------------------------------
    # 内部実装 (Transformer) - Template Support Added
    # -----------------------------------------------------------------
    def _encode_transformer(self, sentences: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.cfg.device)

        out = self._model(**toks)
        hiddens = out.hidden_states  # Tuple[L](B,T,D)
        mask = toks.attention_mask.bool()  # (B,T)

        # ---------------- 内部ヘルパ ----------------
        def _pool(hidden):
            if self.cfg.method == "average":
                summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
                counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                emb = summed / counts
            elif self.cfg.method == "last_token":
                idx = mask.sum(dim=1) - 1
                emb = hidden[torch.arange(hidden.size(0)), idx]
            elif self.cfg.method == "entity_last_token":
                # テンプレート内のエンティティ名の最後のトークンを抽出
                emb = self._extract_entity_embeddings(hidden, mask, sentences)
            else:
                raise ValueError(f"Unknown method: {self.cfg.method}")
            return emb.cpu().float().numpy()

        # 全層まとめて
        if self.cfg.layer == "all":
            pooled = [_pool(h) for h in hiddens]
            return np.stack(pooled)  # (L, B, D)

        # 単層のみ
        hidden = hiddens[self.cfg.layer]
        return _pool(hidden)

    def _extract_entity_embeddings(self, hidden, mask, sentences):
        """テンプレート文からエンティティ名の最後のトークンの埋め込みを抽出"""
        batch_size = hidden.size(0)
        embeddings = []
        
        for i in range(batch_size):
            sentence = sentences[i]
            
            # エンティティ名を抽出（簡単な正規表現ベース）
            entity_name = self._extract_entity_name_from_template(sentence)
            
            if entity_name:
                # エンティティ名の最後のトークンの位置を特定
                entity_last_token_idx = self._find_entity_last_token_position(
                    sentence, entity_name, i
                )
                if entity_last_token_idx is not None and entity_last_token_idx < mask[i].sum():
                    embeddings.append(hidden[i, entity_last_token_idx])
                else:
                    # フォールバック: 文全体の最後のトークン
                    last_idx = mask[i].sum() - 1
                    embeddings.append(hidden[i, last_idx])
            else:
                # エンティティが見つからない場合は最後のトークンを使用
                last_idx = mask[i].sum() - 1
                embeddings.append(hidden[i, last_idx])
        
        return torch.stack(embeddings)

    def _extract_entity_name_from_template(self, template_text: str) -> Optional[str]:
        """テンプレート文からエンティティ名を抽出
        
        テンプレートの例:
        - "Barack Obama" -> "Barack Obama"
        - "What is the occupation of Barack Obama?" -> "Barack Obama"
        - "Who is Marie Curie?" -> "Marie Curie"
        """
        
        # よくあるテンプレートパターンを定義
        patterns = [
            # "What is the occupation of [entity]?"
            r"What is the occupation of (.+?)\?",
            # "Who is [entity]?"
            r"Who is (.+?)\?",
            # "Tell me about [entity]."
            r"Tell me about (.+?)\.",
            # "Describe [entity]."
            r"Describe (.+?)\.",
            # "[entity] is a person who"
            r"^(.+?) is a person who",
            # "Classify [entity] by profession:"
            r"Classify (.+?) by profession:",
            # その他のパターンを必要に応じて追加
        ]
        
        for pattern in patterns:
            match = re.search(pattern, template_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # パターンが見つからない場合、文全体がエンティティ名の可能性
        # （"Barack Obama"のような単純なケース）
        if len(template_text.split()) <= 4 and not template_text.endswith(('?', '.', '!')):
            return template_text.strip()
        
        return None

    def _find_entity_last_token_position(self, sentence: str, entity_name: str, batch_idx: int) -> Optional[int]:
        """エンティティ名の最後のトークンの位置を特定"""
        
        # エンティティ名をトークン化
        entity_tokens = self._tokenizer.tokenize(entity_name)
        if not entity_tokens:
            return None
        
        # 文全体をトークン化
        sentence_tokens = self._tokenizer.tokenize(sentence)
        
        # エンティティトークンが文中のどこに現れるかを検索
        entity_start_idx = self._find_token_sequence(sentence_tokens, entity_tokens)
        
        if entity_start_idx is not None:
            # +1 for [CLS] token (if present)
            has_special_tokens = self._tokenizer.cls_token is not None
            offset = 1 if has_special_tokens else 0
            return entity_start_idx + len(entity_tokens) - 1 + offset
        
        return None

    def _find_token_sequence(self, haystack: List[str], needle: List[str]) -> Optional[int]:
        """トークンシーケンス内で部分シーケンスを検索"""
        if len(needle) > len(haystack):
            return None
        
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        
        return None


# -------------------------------------------------------------------------
# エクスポート
# -------------------------------------------------------------------------
__all__ = ["EmbeddingConfig", "EmbeddingModel"]