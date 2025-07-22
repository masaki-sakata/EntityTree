#!/usr/bin/env python3
"""merge_forest_worker_root.py

Merge multiple taxonomy trees (provided as JSON Lines) into **one** tree whose sole
root is the node with ``qid == "Worker"``.

Steps performed
---------------
1. **Parse** the source JSON Lines file (each line is a node dict).
2. **Create an undirected graph** from every edge (both directions) so that we can
   find the connected component that contains *Worker*.
3. **Collect** every node in that component.
4. **Filter** to keep only the nodes whose ``pop`` value is in the top 10 % of that
   component (and *always* keep the Worker node itself, even if it has no ``pop``).
5. **Drop dangling edges** so that every edge in the result points to a node that
   survived the pop‑filter.
6. **Write** the cleaned‐up tree back to disk in JSON Lines format, preserving
   the original record structure as much as possible (but updating ``num_edges``
   and ``source_props``).

Usage
-----
```bash
poetry run python3 merge_forest_worker_root.py \
    --input  /work03/masaki/data/taxonomy/taxonomy_from_popQA.jsonl \
    --output /work03/masaki/data/taxonomy/taxonomy_from_popQA_merged_worker.jsonl
```

"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT_QID = "Worker"
POP_PERCENTILE = 0.10  # keep the *top* 10 % of pop values

################################################################################
# Utility helpers
################################################################################

def read_jsonl(path: Path) -> List[dict]:
    """Load a JSON Lines file into a list of dicts."""
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def build_index(nodes: List[dict]) -> Dict[str, dict]:
    """Return mapping from qid to the *first* occurrence of that qid."""
    index: Dict[str, dict] = {}
    for node in nodes:
        index.setdefault(node["qid"], node)
    return index


def build_undirected_graph(nodes: List[dict]) -> Dict[str, Set[str]]:
    """Treat every edge as bidirectional to form an undirected adjacency dict."""
    g: Dict[str, Set[str]] = defaultdict(set)
    for node in nodes:
        src = node["qid"]
        for edge in node.get("edges", []):
            tgt = edge["target_qid"]
            g[src].add(tgt)
            g[tgt].add(src)
    return g


def bfs_component(graph: Dict[str, Set[str]], start: str) -> Set[str]:
    """Return the set of qids reachable from *start* using BFS."""
    visited: Set[str] = set()
    queue: List[str] = [start]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(graph.get(node, []))
    return visited


def compute_keep_set(
    component: Set[str],
    nodes_by_qid: Dict[str, dict],
    root_qid: str = ROOT_QID,
) -> Set[str]:
    """Return the qids that survive the pop‑filter + root retention."""
    # Gather pop values (skip nodes without a numeric pop)
    pops = [nodes_by_qid[qid]["pop"] for qid in component if "pop" in nodes_by_qid[qid]]
    if not pops:
        # Nothing to rank; keep the whole connected component.
        return set(component)

    pops.sort(reverse=True)
    k = max(1, int(len(pops) * POP_PERCENTILE))
    threshold = pops[k - 1]  # value at the 90th‑percentile frontier

    keep: Set[str] = {root_qid}
    for qid in component:
        node = nodes_by_qid[qid]
        if qid == root_qid:
            continue  # already kept
        if "pop" in node and node["pop"] >= threshold:
            keep.add(qid)
    return keep


def trim_and_rewire(
    keep: Set[str],
    nodes_by_qid: Dict[str, dict],
) -> List[dict]:
    """Return a list of nodes restricted to *keep* and with edges rewired."""
    trimmed: List[dict] = []
    for qid in keep:
        node = dict(nodes_by_qid[qid])  # shallow copy is fine; we’ll mutate local copy

        # Retain only edges that target another kept node
        new_edges = [e for e in node.get("edges", []) if e["target_qid"] in keep]
        node["edges"] = new_edges
        node["num_edges"] = len(new_edges)
        # Recompute source_props based on surviving edges
        node["source_props"] = sorted({e["property"] for e in new_edges})
        trimmed.append(node)
    return trimmed


def write_jsonl(nodes: List[dict], path: Path) -> None:
    """Write *nodes* to *path* in JSON Lines format (UTF‑8)."""
    with path.open("w", encoding="utf-8") as fp:
        for node in nodes:
            json.dump(node, fp, ensure_ascii=False)
            fp.write("\n")


################################################################################
# Main entry point
################################################################################

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge forest into a Worker‑rooted tree")
    parser.add_argument("--input", required=True, type=Path, help="Path to source JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Path for merged JSONL")
    ns = parser.parse_args(argv)

    # 1. Read & index
    nodes = read_jsonl(ns.input)
    by_qid = build_index(nodes)

    if ROOT_QID not in by_qid:
        sys.exit(f"Error: root qid '{ROOT_QID}' not found in input {ns.input}")

    # 2. Build undirected graph & find component containing Worker
    graph = build_undirected_graph(nodes)
    component = bfs_component(graph, ROOT_QID)

    # 3. Pop‑filter (top 10 %) + always keep the root
    keep = compute_keep_set(component, by_qid, ROOT_QID)

    # 4. Trim nodes & drop dangling edges
    merged_nodes = trim_and_rewire(keep, by_qid)

    # 5. Write result
    write_jsonl(merged_nodes, ns.output)

    print(
        f"Merged {len(nodes)} nodes → {len(merged_nodes)} nodes. "
        f"Output written to {ns.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
