# html_tree_encoding.py (Simple Fixed Version)
# -------------------------------------------------
# StoryTree 可視化 (PyVis + HTML 出力)
#
# Features:
# * Simple circular nodes with profession-based colors
# * Node labels shown as small text below nodes
# * Fixed duplicate title issue
# -------------------------------------------------

from pathlib import Path
from collections import deque
import textwrap
import json
from typing import Dict, Optional, Set

from pyvis.network import Network


class HTMLTreeEncoding:
    def __init__(
        self,
        adjacency: dict[int, tuple[int, int]],
        births=None,
        n_leaves=None,
        n_nodes=None,
        highlights: set[int] | None = None,
        labels: dict[int, str] | None = None,
        node_colors: dict[int, str] | None = None,
        title: str = "Hierarchical Tree Visualization",
        height_px: int = 1000,
        width_pct: int = 100,
        font_size: int = 14,
    ):
        self.adj: dict[int, tuple[int, int]] = adjacency
        self.highlights = highlights or set()
        self.labels = labels or {}
        self.node_colors = node_colors or {}
        self.title = title
        self.font_size = font_size

        # PyVis ネットワーク - headingは使わない（重複の原因）
        self.net = Network(
            height=f"{height_px}px",
            width=f"{width_pct}%",
            directed=True,
            bgcolor="#ffffff",
            font_color="black"
        )

        # 各ノードの深さ（hierarchical layout 用）
        self.levels = self._compute_levels()

    # -------------------- 内部ユーティリティ --------------------
    def _compute_levels(self) -> dict[int, int]:
        """root からの深さを BFS で求める"""
        children = {c for pair in self.adj.values() for c in pair}
        roots = [n for n in self.adj.keys() if n not in children]
        if not roots and self.adj:
            roots = [next(iter(self.adj))]  # フォールバック

        levels: dict[int, int] = {}
        dq = deque([(r, 0) for r in roots])
        while dq:
            node, lv = dq.popleft()
            if node in levels:
                continue
            levels[node] = lv
            for ch in self.adj.get(node, ()):
                dq.append((ch, lv + 1))
        return levels

    def _get_node_label(self, node_id: int) -> str:
        """ノード表示用ラベル（短縮版）"""
        text = self.labels.get(node_id, str(node_id))
        if node_id in self.adj:
            return ""  # 内部ノードは空
        # return textwrap.shorten(text, width=15, placeholder="...")
        return text

    def _get_node_title(self, node_id: int) -> str:
        """ホバー時ツールチップ（完全版）"""
        return self.labels.get(node_id, str(node_id))

    def _get_node_color(self, node_id: int) -> str:
        """ノードの色を決定"""
        if node_id in self.highlights:
            return "#FF0000"  # 強調ノードは赤
        
        # 職業ベースの色分け
        if node_id in self.node_colors:
            return self.node_colors[node_id]
        
        # デフォルト色（内部ノード用）
        if node_id in self.adj:
            return "#666666"  # ダークグレー（内部ノード）
        else:
            return "#CCCCCC"  # ライトグレー（未分類の葉ノード）

    def _get_node_size(self, node_id: int) -> int:
        """ノードサイズを決定"""
        if node_id in self.highlights:
            return 25  # 強調ノードは大きく
        elif node_id in self.adj:
            return 12  # 内部ノード（小さく）
        else:
            return 18  # 葉ノード

    def _add_nodes_edges(self) -> None:
        """ノード→エッジ順に登録"""
        for parent, (right, left) in self.adj.items():
            # 親ノード（内部ノード）
            self.net.add_node(
                parent,
                label=self._get_node_label(parent),
                title=self._get_node_title(parent),
                color=self._get_node_color(parent),
                shape="dot",  # 丸
                size=self._get_node_size(parent),
                level=self.levels.get(parent, 0),
                physics=False,
                font={
                    "size": self.font_size - 2,
                    "color": "black"
                }
            )

            # 子ノード + エッジ
            for child in (right, left):
                self.net.add_node(
                    child,
                    label=self._get_node_label(child),
                    title=self._get_node_title(child),
                    color=self._get_node_color(child),
                    shape="dot",  # 丸
                    size=self._get_node_size(child),
                    level=self.levels.get(child, 0),
                    physics=False,
                    font={
                        "size": self.font_size,
                        "color": "black",
                        "strokeWidth": 0,
                        "strokeColor": "white"
                    }
                )
                
                # エッジを追加
                self.net.add_edge(
                    parent, 
                    child,
                    color="gray",
                    width=2,
                    arrows="to"
                )

    # -------------------- 公開 API --------------------
    def draw(self, path: str = "tree.html", open_browser: bool = False) -> None:
        """HTML を生成。"""
        self._add_nodes_edges()

        # ツリー表示用オプション
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "levelSeparation": 120,
                    "nodeSpacing": 200,
                    "treeSpacing": 200,
                    "sortMethod": "directed"
                }
            },
            "physics": {
                "enabled": False
            },
            "interaction": {
                "dragNodes": True,
                "dragView": True,
                "zoomView": True,
                "hover": True,
                "tooltipDelay": 300
            },
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "chosen": True,
                "font": {
                    "size": self.font_size,
                    "color": "black",
                    "face": "Arial, sans-serif",
                    "background": "rgba(255,255,255,0.8)",
                    "strokeWidth": 1,
                    "strokeColor": "white"
                }
            },
            "edges": {
                "color": {
                    "color": "#848484",
                    "highlight": "#333333"
                },
                "width": 2,
                "smooth": {
                    "enabled": True,
                    "type": "continuous"
                }
            }
        }

        self.net.set_options(json.dumps(options))

        # HTMLファイルを生成
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # PyVisでHTMLを生成
        self.net.write_html(str(out_path), open_browser=False, notebook=False)
        
        # 生成されたHTMLを読み込んで修正
        with open(out_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # タイトルを追加/修正
        title_replacement = f"<title>{self.title}</title>"
        if "<title>" in html_content:
            # 既存のtitleタグを置換
            import re
            html_content = re.sub(r'<title>.*?</title>', title_replacement, html_content)
        else:
            # titleタグを追加
            html_content = html_content.replace('<head>', f'<head>\n{title_replacement}')
        
        # ページヘッダーを追加
        header_html = f"""
        <div style="text-align: center; padding: 20px; background: #f5f5f5; border-bottom: 2px solid #ddd;">
            <h1 style="margin: 0; color: #333; font-size: 24px;">{self.title}</h1>
        </div>
        """
        
        # bodyタグの直後にヘッダーを挿入
        html_content = html_content.replace('<body>', f'<body>\n{header_html}')
        
        # 修正されたHTMLを保存
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if open_browser:
            import webbrowser
            webbrowser.open(f'file://{out_path.absolute()}')


# For backward compatibility
TreeEncoding = HTMLTreeEncoding