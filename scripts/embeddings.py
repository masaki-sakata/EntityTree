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
    extraction will focus on the entity name tokens only.
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
_METHOD = Literal["last_token", "average"]
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
    - method="last_token": エンティティ名の最後のトークン
    - method="average": エンティティ名のトークンの平均
    """

    model_type: _MODEL = "gpt2"
    layer: Union[int, Literal["all"]] = "all"
    method: _METHOD = "average"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: Optional[str] = None  # HuggingFace リポ ID または FastText パス
    verbose: bool = False  # デバッグ用：使用トークンを出力

    # FastText 用デフォルト
    fasttext_path: str = (
        "/work03/masaki/model/fastText_vec/wiki-news-300d-1M-subword.bin"
    )

    # Random embedding 用パラメータ
    random_dim: int = 768
    random_std: float = 1.0
    random_seed: int = 42

    def __post_init__(self) -> None:  # noqa: D401
        """FastText は平均のみ。"""
        if self.model_type in ["fasttext"]:
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
        # FastTextの場合、エンティティ名を抽出してその単語の平均を取る
        entity_name = self._extract_entity_name_from_template(sentence)
        if not entity_name:
            raise ValueError(f"FastText: Entity not found in template sentence: '{sentence}'")
        
        words = entity_name.split()
        
        if self.cfg.verbose:
            print(f"[FastText Debug] Sentence: '{sentence}'")
            print(f"[FastText Debug] Entity: '{entity_name}'")
            print(f"[FastText Debug] Words used: {words[:5]}")  # 最初の5個
        
        if not words:
            raise ValueError(f"FastText: No words found in entity name: '{entity_name}'")
        
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
        
        # Verbose debug for random embeddings
        if self.cfg.verbose:
            print(f"[Random Embeddings Debug] Processing {num_sentences} sentences")
            for i, sentence in enumerate(sentences[:5]):  # Show first 5
                entity_name = self._extract_entity_name_from_template(sentence)
                print(f"[Random Debug {i+1}] Sentence: '{sentence}'")
                print(f"[Random Debug {i+1}] Entity: '{entity_name}'")
                print(f"[Random Debug {i+1}] Note: Random embeddings don't use actual tokens")
                print("-" * 50)
        
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
    # 内部実装 (Transformer) - Entity-focused Methods
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
            batch_size = hidden.size(0)
            embeddings = []
            
            for i in range(batch_size):
                sentence = sentences[i]
                entity_name = self._extract_entity_name_from_template(sentence)
                
                if not entity_name:
                    raise ValueError(f"Transformer: Entity not found in template sentence: '{sentence}'")
                
                # エンティティ名のトークン範囲を特定
                entity_start_idx, entity_end_idx = self._find_entity_token_range(
                    sentence, entity_name, i
                )
                
                if entity_start_idx is None or entity_end_idx is None:
                    if self.cfg.verbose and i < 5:
                        entity_tokens = self._tokenizer.tokenize(entity_name)
                        sentence_tokens = self._tokenizer.tokenize(sentence)
                        print(f"[Debug {i+1}] ERROR: Entity tokens not found!")
                        print(f"[Debug {i+1}] Sentence: '{sentence}'")
                        print(f"[Debug {i+1}] Entity: '{entity_name}'")
                        print(f"[Debug {i+1}] Entity tokens: {entity_tokens}")
                        print(f"[Debug {i+1}] Sentence tokens: {sentence_tokens}")
                        print("-" * 50)
                    raise ValueError(f"Transformer: Entity tokens '{entity_name}' not found in sentence tokens for: '{sentence}'")
                
                # エンティティ名のトークン範囲内で処理
                entity_mask = mask[i, entity_start_idx:entity_end_idx+1]
                entity_hidden = hidden[i, entity_start_idx:entity_end_idx+1]
                
                # デバッグ出力
                if self.cfg.verbose and i < 5:  # 最初の5個のみ出力
                    entity_tokens = self._tokenizer.tokenize(entity_name)
                    sentence_tokens = self._tokenizer.tokenize(sentence)
                    
                    # 実際に使用されるトークンIDを取得して表示
                    full_tokenized = self._tokenizer(sentence, return_tensors="pt")
                    token_ids = full_tokenized.input_ids[0]
                    all_tokens_with_ids = [(self._tokenizer.decode([tid]), tid.item()) for tid in token_ids]
                    entity_tokens_with_ids = all_tokens_with_ids[entity_start_idx:entity_end_idx+1]
                    
                    print(f"[Debug {i+1}] Sentence: '{sentence}'")
                    print(f"[Debug {i+1}] Entity: '{entity_name}'")
                    print(f"[Debug {i+1}] Entity tokens: {entity_tokens}")
                    print(f"[Debug {i+1}] All sentence tokens: {sentence_tokens}")
                    print(f"[Debug {i+1}] Token range: {entity_start_idx}-{entity_end_idx}")
                    print(f"[Debug {i+1}] Actually used tokens+IDs: {entity_tokens_with_ids}")
                    print(f"[Debug {i+1}] Method: {self.cfg.method}")
                    print(f"[Debug {i+1}] IMPORTANT: Using CONTEXTUALIZED embeddings!")
                    print(f"[Debug {i+1}] -> The model processes the ENTIRE sentence first")
                    print(f"[Debug {i+1}] -> Then extracts embeddings from entity token range")
                    if entity_tokens:
                        context_tokens = sentence_tokens[:max(0, entity_start_idx)]
                        print(f"[Debug {i+1}] -> '{entity_tokens[-1]}' is contextualized by: {context_tokens}")
                    print("-" * 50)
                
                if self.cfg.method == "average":
                    # エンティティ名トークンの平均
                    if entity_mask.sum() > 0:
                        summed = (entity_hidden * entity_mask.unsqueeze(-1)).sum(dim=0)
                        count = entity_mask.sum().clamp(min=1)
                        emb = summed / count
                    else:
                        emb = entity_hidden.mean(dim=0)
                else:  # last_token
                    # エンティティ名の最後のトークン
                    if entity_mask.sum() > 0:
                        valid_indices = entity_mask.nonzero(as_tuple=True)[0]
                        last_valid_idx = valid_indices[-1]
                        emb = entity_hidden[last_valid_idx]
                    else:
                        emb = entity_hidden[-1]
                
                embeddings.append(emb)
            
            return torch.stack(embeddings).cpu().float().numpy()

        # 全層まとめて
        if self.cfg.layer == "all":
            pooled = [_pool(h) for h in hiddens]
            return np.stack(pooled)  # (L, B, D)

        # 単層のみ
        hidden = hiddens[self.cfg.layer]
        return _pool(hidden)

    def _extract_entity_name_from_template(self, template_text: str) -> Optional[str]:
        """テンプレート文からエンティティ名を抽出
        
        テンプレートの例:
        - "Barack Obama" -> "Barack Obama"
        - "What is the occupation of Barack Obama?" -> "Barack Obama"
        - "Who is Marie Curie?" -> "Marie Curie"
        """
        
        # よくあるテンプレートパターンを定義（順序重要：より具体的なものを先に）
        patterns = [
            # "What is the occupation of [entity]?"
            r"What is the occupation of (.+?)",
            # "What would be a good gift for [entity]?"
            r"What would be a good gift for (.+?)",
            # "Who is [entity]?"
            r"Who is (.+?)",
            # "Tell me about [entity]."
            r"Tell me about (.+?)\.",
            # "Describe [entity]."
            r"Describe (.+?)\.",
            # "[entity] is a person who"
            r"^(.+?) is a person who",
            # "Classify [entity] by profession:"
            r"Classify (.+?) by profession:",
            # "The profession of [entity] is"
            r"The profession of (.+?) is",
            # "[entity]'s occupation"
            r"^(.+?)'s occupation",
            # その他のパターンを必要に応じて追加
        ]
        
        for pattern in patterns:
            match = re.search(pattern, template_text, re.IGNORECASE)
            if match:
                entity = match.group(1).strip()
                if self.cfg.verbose:
                    print(f"[Entity Extraction] Pattern matched: '{pattern}' -> '{entity}'")
                return entity
        
        # パターンが見つからない場合、文全体がエンティティ名の可能性
        # （"Barack Obama"のような単純なケース）
        if len(template_text.split()) <= 4 and not template_text.endswith(('?', '.', '!')):
            if self.cfg.verbose:
                print(f"[Entity Extraction] Using whole text as entity: '{template_text}'")
            return template_text.strip()
        
        # デバッグ情報を出力
        if self.cfg.verbose:
            print(f"[Entity Extraction] WARNING: No pattern matched for: '{template_text}'")
            print(f"[Entity Extraction] Available patterns: {len(patterns)}")
        
        return None

    def _find_entity_token_range(self, sentence: str, entity_name: str, batch_idx: int) -> tuple[Optional[int], Optional[int]]:
        """エンティティ名のトークン範囲（開始と終了インデックス）を特定"""
        
        # エンティティ名をトークン化
        entity_tokens = self._tokenizer.tokenize(entity_name)
        if not entity_tokens:
            if self.cfg.verbose:
                print(f"[Token Range] ERROR: No tokens for entity '{entity_name}'")
            return None, None
        
        # 文全体をトークン化
        sentence_tokens = self._tokenizer.tokenize(sentence)
        
        if self.cfg.verbose and batch_idx < 5:
            print(f"[Token Range] Entity '{entity_name}' -> tokens: {entity_tokens}")
            print(f"[Token Range] Sentence tokens: {sentence_tokens}")
        
        # エンティティトークンが文中のどこに現れるかを検索
        entity_start_idx = self._find_token_sequence(sentence_tokens, entity_tokens)
        
        if entity_start_idx is not None:
            # +1 for [CLS] token (if present)
            has_special_tokens = self._tokenizer.cls_token is not None
            offset = 1 if has_special_tokens else 0
            start_idx = entity_start_idx + offset
            end_idx = start_idx + len(entity_tokens) - 1
            
            if self.cfg.verbose and batch_idx < 5:
                print(f"[Token Range] Found at sentence position: {entity_start_idx}")
                print(f"[Token Range] Final range (with offset {offset}): {start_idx}-{end_idx}")
            
            return start_idx, end_idx
        
        # 完全一致が失敗した場合、部分マッチを試行
        partial_match = self._find_partial_token_sequence(sentence_tokens, entity_tokens)
        if partial_match is not None:
            has_special_tokens = self._tokenizer.cls_token is not None
            offset = 1 if has_special_tokens else 0
            start_idx = partial_match[0] + offset
            end_idx = partial_match[1] + offset
            
            if self.cfg.verbose and batch_idx < 5:
                print(f"[Token Range] Partial match found at: {partial_match}")
                print(f"[Token Range] Final range (with offset {offset}): {start_idx}-{end_idx}")
            
            return start_idx, end_idx
        
        if self.cfg.verbose and batch_idx < 5:
            print(f"[Token Range] ERROR: No match found for entity tokens")
        
        return None, None

    def _find_partial_token_sequence(self, haystack: List[str], needle: List[str]) -> Optional[tuple[int, int]]:
        """部分的なトークンシーケンスマッチを試行"""
        if not needle:
            return None
        
        # 各トークンを個別に検索して連続する範囲を見つける
        for i in range(len(haystack)):
            matched_count = 0
            for j, token in enumerate(needle):
                if i + j < len(haystack):
                    # トークンの部分一致も許可（サブワードトークン対応）
                    if (token in haystack[i + j] or 
                        haystack[i + j] in token or
                        token.lower() == haystack[i + j].lower()):
                        matched_count += 1
                    else:
                        break
                else:
                    break
            
            # 過半数以上マッチした場合を有効とする
            if matched_count >= len(needle) * 0.6:  # 60%以上マッチ
                return (i, i + matched_count - 1)
        
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