import json
from collections import defaultdict, deque

def filter_tree_to_worker_root(input_file, output_file):
    """
    JSONLファイルの木構造をフィルタリングして、Workerのみを根ノードとして残す
    
    Args:
        input_file: 入力JSONLファイルのパス
        output_file: 出力JSONLファイルのパス
    """
    
    # データを読み込み
    nodes = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            node = json.loads(line.strip())
            nodes[node['qid']] = node
    
    print(f"総ノード数: {len(nodes)}")
    
    # エッジ関係を分析
    parent_to_children = defaultdict(set)  # P527関係 (has part)
    child_to_parents = defaultdict(set)    # P361関係 (part of)
    
    for qid, node in nodes.items():
        for edge in node.get('edges', []):
            if edge['property'] == 'P527':  # has part
                parent_to_children[qid].add(edge['target_qid'])
                child_to_parents[edge['target_qid']].add(qid)
            elif edge['property'] == 'P361':  # part of
                child_to_parents[qid].add(edge['target_qid'])
                parent_to_children[edge['target_qid']].add(qid)
    
    # 根ノード（親を持たないノード）を特定
    root_nodes = set()
    for qid in nodes.keys():
        if qid not in child_to_parents or len(child_to_parents[qid]) == 0:
            root_nodes.add(qid)
    
    print(f"根ノード: {[nodes[qid]['wiki_title'] for qid in root_nodes if qid in nodes]}")
    
    # Workerノードを見つける
    worker_qid = None
    for qid, node in nodes.items():
        if node['wiki_title'] == 'Worker':
            worker_qid = qid
            break
    
    if worker_qid is None:
        raise ValueError("Workerノードが見つかりません")
    
    print(f"Workerノード: {worker_qid}")
    
    # Worker以外の根ノードを特定
    other_root_nodes = root_nodes - {worker_qid}
    print(f"削除対象の根ノード: {[nodes[qid]['wiki_title'] for qid in other_root_nodes if qid in nodes]}")
    
    # Worker以外の根ノードとその子孫を削除
    nodes_to_remove = set()
    
    def mark_subtree_for_removal(node_qid):
        """指定されたノードとその子孫を削除対象としてマーク"""
        if node_qid in nodes_to_remove:
            return
        nodes_to_remove.add(node_qid)
        # 子ノードも再帰的に削除対象とする
        for child_qid in parent_to_children.get(node_qid, []):
            mark_subtree_for_removal(child_qid)
    
    # Worker以外の根ノードとその子孫を削除対象とする
    for root_qid in other_root_nodes:
        mark_subtree_for_removal(root_qid)
    
    print(f"初期削除対象ノード数: {len(nodes_to_remove)}")
    
    # 削除されたノードにのみ接続していたノードも順次削除
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        print(f"削除イテレーション {iteration}")
        
        # 各ノードについて、親がすべて削除されたかチェック
        for qid in list(nodes.keys()):
            if qid in nodes_to_remove:
                continue
            
            # このノードの親をチェック
            parents = child_to_parents.get(qid, set())
            if parents:  # 親がある場合
                # すべての親が削除対象かチェック
                all_parents_removed = all(parent_qid in nodes_to_remove for parent_qid in parents)
                if all_parents_removed:
                    mark_subtree_for_removal(qid)
                    changed = True
                    print(f"  削除: {nodes[qid]['wiki_title']} (すべての親が削除済み)")
    
    print(f"最終削除対象ノード数: {len(nodes_to_remove)}")
    
    # 残るノードを特定
    remaining_nodes = {qid: node for qid, node in nodes.items() if qid not in nodes_to_remove}
    print(f"残存ノード数: {len(remaining_nodes)}")
    
    # 残るノードのエッジを更新（削除されたノードへの参照を除去）
    updated_nodes = {}
    for qid, node in remaining_nodes.items():
        updated_node = node.copy()
        updated_edges = []
        
        for edge in node.get('edges', []):
            target_qid = edge['target_qid']
            if target_qid in remaining_nodes:  # 対象ノードが残存している場合のみエッジを保持
                updated_edges.append(edge)
        
        updated_node['edges'] = updated_edges
        updated_node['num_edges'] = len(updated_edges)
        updated_nodes[qid] = updated_node
    
    # 結果を出力
    with open(output_file, 'w', encoding='utf-8') as f:
        for node in updated_nodes.values():
            f.write(json.dumps(node, ensure_ascii=False) + '\n')
    
    print(f"フィルタリング完了: {output_file}")
    
    # 統計情報を表示
    print("\n=== 統計情報 ===")
    print(f"元のノード数: {len(nodes)}")
    print(f"削除されたノード数: {len(nodes_to_remove)}")
    print(f"残存ノード数: {len(remaining_nodes)}")
    
    # 削除されたノードの例
    if nodes_to_remove:
        print("\n削除されたノードの例:")
        for i, qid in enumerate(list(nodes_to_remove)[:10]):
            if qid in nodes:
                print(f"  - {nodes[qid]['wiki_title']}")
    
    # 残存ノードの根ノード
    remaining_roots = []
    for qid in remaining_nodes.keys():
        parents = child_to_parents.get(qid, set())
        # 親がないか、すべての親が削除されている場合
        if not parents or all(p in nodes_to_remove for p in parents):
            remaining_roots.append(qid)
    
    print("\n残存する根ノード:")
    for qid in remaining_roots:
        if qid in remaining_nodes:
            print(f"  - {remaining_nodes[qid]['wiki_title']}")

# 使用例
if __name__ == "__main__":
    # ファイルパスを指定して実行
    input_file = "/work03/masaki/data/taxonomy/taxonomy_from_popQA_merged_worker_top0_05.jsonl"  # 入力ファイル
    output_file = "/work03/masaki/data/taxonomy/taxonomy_from_popQA_Only_worker_top0_05.jsonl"  # 出力ファイル
    
    try:
        filter_tree_to_worker_root(input_file, output_file)
    except Exception as e:
        print(f"エラー: {e}")