#!/usr/bin/env python3
"""
Extract last–token hidden states of every node-title that appears in an
“in-context learning (ICL)” tree description.

2025-06-16 版
  • char_end を SentencePiece 正規化後の文字列で計算
  • --is_verbose でタイトル＋ token を表示
  • 位置が合わなければ即 RuntimeError で停止
"""

import argparse, json, os, sys
from collections import defaultdict

import torch
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',  required=True)
    p.add_argument('--output_dir', required=True)

    p.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B')
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--device', default='auto')

    p.add_argument('--text_mode', default='title')
    p.add_argument('--num_repeats', type=int, default=10,
                   help='How many times the same tree is repeated (k)')
    p.add_argument('--is_verbose', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            if not (line := line.strip()):
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f'[ERROR] line {ln}: {e}')
                sys.exit(1)
    groups = defaultdict(list)
    for d in data:
        groups[d['tree_id']].append(d)
    print(f'Loaded {len(data)} nodes in {len(groups)} trees')
    return groups


# ---------------------------------------------------------------------------
def build_tree_structure(tree_data):
    nodes, child_map = {}, defaultdict(list)
    all_children = set()

    for item in tree_data:
        qid = item['qid']
        nodes[qid] = {'title': item['wiki_title']}
        for e in item['edges']:
            if e['property'] == 'P527':
                child_map[qid].append(e['target_qid'])
                all_children.add(e['target_qid'])

    roots = [q for q in nodes if q not in all_children]
    if len(roots) != 1:
        raise ValueError(f'Cannot determine unique root: {roots}')
    return nodes, child_map, roots[0]


# ---------------------------------------------------------------------------
def _sanitize(title: str) -> str:
    return title.replace(' ', '_')


def make_icl_text_and_char_end(nodes, child_map, root_qid, normalizer,
                               indent='    '):
    """
    1. raw ICL 文字列を作る
    2. 正規化器で prefix を正規化しながら char_end を測定
    戻り値:
        icl_text_norm : str   (正規化済み文字列)
        char_end_norm : {qid: int}
    """
    raw_parts = []
    raw_cursor = 0
    char_end_norm = {}
    norm = normalizer

    # helper: 正規化した prefix 長を取る
    def norm_len(s):
        return len(norm(s))

    def rec(qid: str, depth: int):
        nonlocal raw_cursor
        ind = indent * depth
        title = _sanitize(nodes[qid]['title'])

        raw_parts.append(ind);   raw_cursor += len(ind)
        raw_parts.append(title); raw_cursor += len(title)

        # 正規化後の prefix 長を char_end として記録
        char_end_norm[qid] = norm_len(''.join(raw_parts))

        children = child_map.get(qid, [])
        if children:
            raw_parts.append('(\n'); raw_cursor += 2
            for i, ch in enumerate(children):
                rec(ch, depth + 1)
                if i != len(children) - 1:
                    raw_parts.append(',\n'); raw_cursor += 2
                else:
                    raw_parts.append('\n');  raw_cursor += 1
            raw_parts.append(ind + ')'); raw_cursor += len(ind) + 1

    rec(root_qid, 0)
    icl_raw = ''.join(raw_parts)
    icl_norm = norm(icl_raw)
    return icl_norm, char_end_norm


# ---------------------------------------------------------------------------
def prepare_prompt_and_lookup(icl_norm, char_end_norm, repeats, normalizer):
    sep_norm = normalizer('\n')
    unit = icl_norm + sep_norm
    unit_len = len(unit)

    lookup = {}
    for r in range(1, repeats + 1):
        offset = (r - 1) * unit_len
        for qid, end in char_end_norm.items():
            lookup[(r, qid)] = offset + end

    prompt_norm = unit * repeats
    return prompt_norm, lookup


# ---------------------------------------------------------------------------
def load_model(model_name, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(model_name,
                                    torch_dtype=torch.bfloat16,
                                    output_hidden_states=True)
    mdl.to(device).eval()
    return tok, mdl


# ---------------------------------------------------------------------------
def extract_embeddings(prompt: str,
                       lookup: dict,
                       tokenizer, model,
                       device, max_length,
                       qid2title,
                       verbose=False):
    """
    prompt 全体を 1 回 forward し、
    lookup[(rep, qid)] が指す “最後のサブトークン” の隠れ状態を取得。
    戻り値:
        rep_embeddings : {rep_idx: {qid: tensor[L,D]}}
        num_layers     : int
    """
    enc = tokenizer(prompt,
                    return_tensors='pt',
                    return_offsets_mapping=True,
                    padding=False,
                    truncation=False)
    if enc['input_ids'].shape[1] > max_length:
        raise ValueError(f'Prompt too long: {enc["input_ids"].shape[1]} tokens')

    offsets = enc.pop('offset_mapping')[0].tolist()            # [(s,e), ...]
    char2tok = {e: i for i, (s, e) in enumerate(offsets) if e != 0}

    input_ids = enc['input_ids'][0].tolist()                   # verbose 用
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
    hidden = out.hidden_states                                 # tuple[L+1]
    num_layers = len(hidden)

    rep_embeddings = defaultdict(dict)
    missing = []                                               # align 失敗
    buf = []                                                   # verbose

    for (rep, qid), char_end in lookup.items():
        tok_idx = char2tok.get(char_end)

        # ---------- 追加: 最大 BACK_STEP 文字だけ後ろへ戻って探す ----------
        if tok_idx is None:
            BACK_STEP = 8
            for off in range(1, BACK_STEP + 1):
                tok_idx = char2tok.get(char_end - off)
                if tok_idx is not None:
                    break
        # --------------------------------------------------------------------

        if tok_idx is None:        # それでも見つからない → missing
            missing.append((rep, qid))
            continue

        vec = torch.stack([h[0, tok_idx].cpu() for h in hidden])   # [L,D]
        rep_embeddings[rep][qid] = vec

        if verbose and len(buf) < 30:
            tok_str = tokenizer.convert_ids_to_tokens(input_ids[tok_idx],
                                                      skip_special_tokens=False)
            buf.append(f'repeat={rep:2d}  title="{qid2title[qid]}"  '
                       f'token_idx={tok_idx:4d}  token="{tok_str}"')

    if verbose and buf:
        print('  [VERBOSE] 例として抽出したトークン:')
        for l in buf:
            print('   ', l)

    # missing が残っていればエラーで止める
    if missing:
        sample = ', '.join(qid2title[q] for _, q in missing[:5])
        raise RuntimeError(f'{len(missing)} nodes could not be aligned '
                           f'(e.g., {sample})')

    return rep_embeddings, num_layers


# ---------------------------------------------------------------------------
def save_embeddings(rep_embeddings, num_layers,
                    tree_id, nodes_order,
                    output_root, model_name,
                    text_mode, repeat_idx):
    outdir = os.path.join(output_root,
                          text_mode,
                          f'tree_{tree_id}',
                          f'repeat_{repeat_idx}')
    os.makedirs(outdir, exist_ok=True)

    embeds = [rep_embeddings[qid] for qid in nodes_order]  # 全ノード必須
    tensor = torch.stack(embeds)                           # [N,L,D]

    meta = {
        'tree_id': tree_id,
        'repeat_idx': repeat_idx,
        'nodes': nodes_order,
        'num_layers': num_layers,
        'model_name': model_name,
        'text_mode': text_mode
    }
    path = os.path.join(outdir, f'tree_{tree_id}_embeddings.pt')
    torch.save({'embeddings': tensor, 'metadata': meta}, path)
    print(f'  saved → {path}  shape={tuple(tensor.shape)}')

def get_normalizer(tokenizer):
    """
    Return a function norm(text) -> str

    • fast 版 SentencePiece tokenizer  : internal normalizer を呼び出す
    • slow 版 (backend_tokenizer が None) : 恒等写像を返す
    """
    btok = getattr(tokenizer, "backend_tokenizer", None)
    if btok is not None and getattr(btok, "normalizer", None) is not None:
        return btok.normalizer.normalize_str
    else:
        print("  [INFO] slow tokenizer detected → identity normalizer")
        return lambda x: x

# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    device = (torch.device('cuda') if (args.device == 'auto' and
                                       torch.cuda.is_available())
              else torch.device(args.device if args.device != 'auto'
                                else 'cpu'))
    print(f'Using device : {device}')

    trees = load_jsonl(args.data_path)

    tokenizer, model = load_model(args.model_name, device)
    normalizer = get_normalizer(tokenizer)          
    output_root = args.output_dir

    for tree_id, tree_data in trees.items():
        print(f'\n======= Tree {tree_id} =======')
        nodes, child_map, root_qid = build_tree_structure(tree_data)

        # 1 木ぶんを正規化文字列で構築
        icl_norm, char_end_norm = make_icl_text_and_char_end(
            nodes, child_map, root_qid, normalizer
        )

        # prompt 全体と lookup を作成
        prompt, lookup = prepare_prompt_and_lookup(
            icl_norm, char_end_norm, args.num_repeats, normalizer
        )

        qid2title = {qid: info['title'] for qid, info in nodes.items()}

        rep_embeds, num_layers = extract_embeddings(
            prompt, lookup, tokenizer, model,
            device, args.max_length,
            qid2title=qid2title,
            verbose=args.is_verbose
        )

        nodes_order = list(nodes.keys())
        for r in range(1, args.num_repeats + 1):
            save_embeddings(rep_embeds[r], num_layers,
                            tree_id, nodes_order,
                            output_root, args.model_name,
                            args.text_mode, r)

    print('\n✓ finished')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()