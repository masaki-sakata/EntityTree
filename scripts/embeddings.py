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
    def encode(self, sentences: List[str], entity_names: List[str]) -> np.ndarray:
        """文リストを埋め込んで返す。

        Args:
            sentences: テンプレート化されたテキストのリスト
            entity_names: エンティティ名のリスト（必須）

        戻り値の形状は次のとおり:

        * ``layer`` が "all" の場合 … ``(L, N, D)``
        * それ以外の場合           … ``(N, D)``
        """
        if len(sentences) != len(entity_names):
            raise ValueError(f"sentences と entity_names の長さが一致しません: {len(sentences)} != {len(entity_names)}")

        if self.cfg.model_type == "fasttext":
            return np.stack([self._encode_fasttext(s, entity_names[i]) for i, s in enumerate(sentences)])

        if self.cfg.model_type == "random_emb":
            return self._encode_random(sentences, entity_names)

        return self._encode_transformer(sentences, entity_names)

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
    def _encode_fasttext(self, sentence: str, entity_name: str) -> np.ndarray:
        # FastTextの場合、エンティティ名を使用
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
    def _encode_random(self, sentences: List[str], entity_names: List[str]) -> np.ndarray:
        """Generate random embeddings for sentences with specified parameters."""
        # Set seed for reproducibility
        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        
        num_sentences = len(sentences)
        
        # Verbose debug for random embeddings
        if self.cfg.verbose:
            print(f"[Random Embeddings Debug] Processing {num_sentences} sentences")
            for i, sentence in enumerate(sentences[:5]):  # Show first 5
                entity_name = entity_names[i]
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
    def _encode_transformer(self, sentences: List[str], entity_names: List[str]) -> np.ndarray:
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
                entity_name = entity_names[i]
                
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
                    raise ValueError(f"Transformer: Entity tokens '{entity_name}' not found in sentence tokens for: '{sentence}'")
                
                # エンティティ名のトークン範囲内で処理
                entity_mask = mask[i, entity_start_idx:entity_end_idx+1]
                entity_hidden = hidden[i, entity_start_idx:entity_end_idx+1]
                
                # トークンIDを取得（デバッグ出力用）
                full_tokenized = self._tokenizer(sentence, return_tensors="pt")
                token_ids = full_tokenized.input_ids[0]
                
                # デバッグ出力
                if self.cfg.verbose and i < 5:  # 最初の5個のみ出力
                    entity_tokens = self._tokenizer.tokenize(entity_name)
                    sentence_tokens = self._tokenizer.tokenize(sentence)
                    
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
                
                if self.cfg.method == "average":
                    # エンティティ名トークンの平均
                    if entity_mask.sum() > 0:
                        summed = (entity_hidden * entity_mask.unsqueeze(-1)).sum(dim=0)
                        count = entity_mask.sum().clamp(min=1)
                        emb = summed / count
                        
                        # Verbose: 平均に使用されたトークンを表示
                        if self.cfg.verbose and i < 5:
                            valid_indices = entity_mask.nonzero(as_tuple=True)[0]
                            used_tokens = [self._tokenizer.decode([token_ids[entity_start_idx + idx]]) 
                                         for idx in valid_indices]
                            used_token_ids = [token_ids[entity_start_idx + idx].item() 
                                            for idx in valid_indices]
                            print(f"[Debug {i+1}] AVERAGE method: Using {len(used_tokens)} tokens")
                            print(f"[Debug {i+1}] -> Tokens used for average: {list(zip(used_tokens, used_token_ids))}")
                    else:
                        emb = entity_hidden.mean(dim=0)
                        if self.cfg.verbose and i < 5:
                            print(f"[Debug {i+1}] AVERAGE method: No valid tokens, using mean of all entity tokens")
                else:  # last_token
                    # エンティティ名の最後のトークン
                    if entity_mask.sum() > 0:
                        valid_indices = entity_mask.nonzero(as_tuple=True)[0]
                        last_valid_idx = valid_indices[-1]
                        emb = entity_hidden[last_valid_idx]
                        
                        # Verbose: 最後のトークンを表示
                        if self.cfg.verbose and i < 5:
                            last_token = self._tokenizer.decode([token_ids[entity_start_idx + last_valid_idx]])
                            last_token_id = token_ids[entity_start_idx + last_valid_idx].item()
                            print(f"[Debug {i+1}] LAST_TOKEN method: Using the last token")
                            print(f"[Debug {i+1}] -> Last token used: ('{last_token}', {last_token_id})")
                    else:
                        emb = entity_hidden[-1]
                        if self.cfg.verbose and i < 5:
                            print(f"[Debug {i+1}] LAST_TOKEN method: No valid tokens, using last entity token")
                
                embeddings.append(emb)
            
            return torch.stack(embeddings).cpu().float().numpy()

        # 全層まとめて
        if self.cfg.layer == "all":
            pooled = [_pool(h) for h in hiddens]
            return np.stack(pooled)  # (L, B, D)

        # 単層のみ
        hidden = hiddens[self.cfg.layer]
        return _pool(hidden)

    def _find_entity_token_range(self, sentence: str, entity_name: str, batch_idx: int) -> tuple[Optional[int], Optional[int]]:
        """エンティティ名のトークン範囲（開始と終了インデックス）を特定"""
        
        # 文字レベルでエンティティの位置を特定
        char_start = sentence.lower().find(entity_name.lower())
        if char_start == -1:
            if self.cfg.verbose and batch_idx < 5:
                print(f"[Token Range] ERROR: Entity '{entity_name}' not found in sentence '{sentence}'")
            return None, None
        
        char_end = char_start + len(entity_name)
        
        if self.cfg.verbose and batch_idx < 5:
            print(f"[Token Range] Entity '{entity_name}' found at character range: {char_start}-{char_end-1}")
            print(f"[Token Range] Entity text: '{sentence[char_start:char_end]}'")
        
        # Fast tokenizerのoffset mappingを試行
        try:
            tokenized = self._tokenizer(
                sentence, 
                return_tensors="pt", 
                return_offsets_mapping=True
            )
            offset_mapping = tokenized.offset_mapping[0]  # (num_tokens, 2)
            
            # エンティティの文字範囲に重なるトークンを見つける
            entity_token_indices = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < char_end and token_end > char_start:
                    entity_token_indices.append(token_idx)
            
            if entity_token_indices:
                start_idx = entity_token_indices[0]
                end_idx = entity_token_indices[-1]
                
                if self.cfg.verbose and batch_idx < 5:
                    token_ids = tokenized.input_ids[0]
                    entity_tokens_with_ids = [(self._tokenizer.decode([tid]), tid.item()) for tid in token_ids[start_idx:end_idx+1]]
                    print(f"[Token Range] Found entity at token range: {start_idx}-{end_idx}")
                    print(f"[Token Range] Entity tokens+IDs: {entity_tokens_with_ids}")
                
                return start_idx, end_idx
                
        except Exception as e:
            if self.cfg.verbose and batch_idx < 5:
                print(f"[Token Range] Offset mapping failed: {e}, falling back to manual method")
        
        # フォールバック: 手動でトークン位置を特定
        tokenized = self._tokenizer(sentence, return_tensors="pt")
        token_ids = tokenized.input_ids[0]
        
        # より確実な方法：各トークンをデコードして順次照合
        reconstructed_text = ""
        entity_start_idx = None
        entity_end_idx = None
        
        for i, token_id in enumerate(token_ids):
            token_str = self._tokenizer.decode([token_id], skip_special_tokens=True)
            prev_length = len(reconstructed_text)
            reconstructed_text += token_str
            current_length = len(reconstructed_text)
            
            # エンティティがこのトークンの範囲に重なるかチェック
            if (prev_length < char_end and current_length > char_start):
                if entity_start_idx is None:
                    entity_start_idx = i
                entity_end_idx = i
        
        if entity_start_idx is not None and entity_end_idx is not None:
            if self.cfg.verbose and batch_idx < 5:
                entity_tokens_with_ids = [(self._tokenizer.decode([token_ids[i]]), token_ids[i].item()) for i in range(entity_start_idx, entity_end_idx + 1)]
                print(f"[Token Range] Found entity at token range (fallback): {entity_start_idx}-{entity_end_idx}")
                print(f"[Token Range] Entity tokens+IDs: {entity_tokens_with_ids}\n\n")
            
            return entity_start_idx, entity_end_idx
        
        if self.cfg.verbose and batch_idx < 5:
            print(f"[Token Range] ERROR: Could not find token range for entity")
        
        return None, None

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