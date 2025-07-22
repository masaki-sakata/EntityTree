#!/usr/bin/env python3
"""
Prune a JSON-lines tree so that only the top Nth-percentile by `pop` remain,
while keeping the hierarchy intact (based on P527 edges).
"""
import argparse
import json
from collections import defaultdict
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prune a JSONL tree by pop percentile, preserving hierarchy based on P527 edges.'
    )
    parser.add_argument(
        '--input',
        help='Path to the input JSONL file containing the tree.'
    )
    parser.add_argument(
        '--output',
        help='Path to the output JSONL file for the pruned tree.'
    )
    parser.add_argument(
        '--percentile', '-p',
        type=float,
        default=99.0,
        help='Percentile threshold (0-100). Leaves without pop or with pop below this will be pruned.'
    )
    return parser.parse_args()


def load_nodes(path):
    nodes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes.append(json.loads(line))
    return nodes


def compute_threshold(nodes, percentile):
    pops = [n['pop'] for n in nodes if 'pop' in n]
    if not pops:
        print('No pop values found; nothing to prune.', file=sys.stderr)
        sys.exit(1)
    pops_sorted = sorted(pops)
    k = (len(pops_sorted) - 1) * percentile / 100.0
    f = int(k)
    c = f + 1
    if c >= len(pops_sorted):
        return pops_sorted[-1]
    d0 = pops_sorted[f] * (c - k)
    d1 = pops_sorted[c] * (k - f)
    return d0 + d1


def build_graph(nodes):
    # Build hierarchy only from P527 edges (hasPart)
    children = defaultdict(list)
    parents = defaultdict(list)
    for n in nodes:
        q = n['qid']
        for edge in n.get('edges', []):
            if edge.get('property') == 'P527':
                tgt = edge['target_qid']
                children[q].append(tgt)
                parents[tgt].append(q)
    return children, parents


def prune(nodes_by_qid, children, parents, threshold):
    to_remove = set()
    # Seed: leaves (no P527 children) without pop or pop < threshold
    for qid, node in list(nodes_by_qid.items()):
        if not children.get(qid):
            if 'pop' not in node or node['pop'] < threshold:
                to_remove.add(qid)

    while to_remove:
        next_round = set()
        for qid in to_remove:
            for p in parents.get(qid, []):
                if qid in children.get(p, []):
                    children[p].remove(qid)
                # If parent becomes leaf and should be pruned
                pnode = nodes_by_qid.get(p)
                if pnode and not children[p]:
                    if 'pop' not in pnode or pnode['pop'] < threshold:
                        next_round.add(p)
            nodes_by_qid.pop(qid, None)
        to_remove = next_round


def write_nodes(nodes_by_qid, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for node in nodes_by_qid.values():
            f.write(json.dumps(node, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    nodes = load_nodes(args.input)
    threshold = compute_threshold(nodes, args.percentile)
    print(f'Pruning nodes with pop below percentile {args.percentile} --> threshold {threshold}', file=sys.stderr)
    nodes_by_qid = {n['qid']: n for n in nodes}
    children, parents = build_graph(nodes)
    prune(nodes_by_qid, children, parents, threshold)
    write_nodes(nodes_by_qid, args.output)
    print(f'Pruned tree written to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
