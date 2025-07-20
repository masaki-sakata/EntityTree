# html_tree_encoding.py
# -------------------------------------------------
# StoryTree 可視化 (PyVis + HTML 出力)
#
# * hierarchical layout で“木”らしいレイアウト
# * 葉ノードはセンテンス本文をラベル表示（50 文字で省略）
# * ホバー時に全文ツールチップ
# * highlights で強調ノードを指定可（赤）
# * write_html() を使い notebook=False でテンプレート問題を回避
# -------------------------------------------------

from pathlib import Path
from collections import deque
import textwrap

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
        height_px: int = 900,
        width_pct: int = 100,
    ):
        self.adj: dict[int, tuple[int, int]] = adjacency
        self.highlights = highlights or set()
        self.labels = labels or {}

        # PyVis ネットワーク
        self.net = Network(
            height=f"{height_px}px",
            width=f"{width_pct}%",
            directed=True,
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

    def _label(self, node_id: int) -> str:
        """ノード表示用ラベル（50 文字超は省略）"""
        text = self.labels.get(node_id, str(node_id))
        return textwrap.shorten(text, width=50, placeholder="…")

    def _title(self, node_id: int) -> str:
        """ホバー時ツールチップ"""
        return self.labels.get(node_id, str(node_id))

    def _add_nodes_edges(self) -> None:
        """ノード→エッジ順に登録（親ノードを先に追加）"""
        for parent, (right, left) in self.adj.items():
            # 親ノード
            self.net.add_node(
                parent,
                label=self._label(parent),
                title=self._title(parent),
                color="grey",
                level=self.levels.get(parent, 0),
                physics=False,
            )

            # 子ノード + エッジ
            for child in (right, left):
                color = (
                    "red"
                    if child in self.highlights
                    else ("lightblue" if child in self.adj else "orange")
                )
                self.net.add_node(
                    child,
                    label=self._label(child),
                    title=self._title(child),
                    color=color,
                    level=self.levels.get(child, 0),
                    physics=False,
                )
                self.net.add_edge(parent, child)

    # -------------------- 公開 API --------------------
    def draw(self, path: str = "tree.html", open_browser: bool = False) -> None:
        """HTML を生成。open_browser=True で自動表示。"""
        self._add_nodes_edges()

        # ツリー表示用オプション（純粋な JSON 形式）
        self.net.set_options(
            """
{
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "levelSeparation": 120,
      "nodeSpacing": 200,
      "treeSpacing": 200,
      "sortMethod": "directed"
    }
  },
  "physics": {
    "enabled": false
  }
}
"""
        )

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.net.write_html(str(out_path), open_browser=open_browser, notebook=False)
