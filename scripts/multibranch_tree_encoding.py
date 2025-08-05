# multibranch_tree_encoding.py
# -------------------------------------------------
# Multi-branch Tree Visualization using PyVis
# -------------------------------------------------

from pathlib import Path
from collections import deque
import textwrap
import json
from typing import Dict, Optional, Set, List

from pyvis.network import Network


class MultiBranchTreeEncoding:
    def __init__(
        self,
        adjacency: dict[int, List[int]],  # Multi-branch adjacency
        births=None,
        n_leaves=None,
        n_nodes=None,
        highlights: set[int] | None = None,
        labels: dict[int, str] | None = None,
        node_colors: dict[int, str] | None = None,
        title: str = "Multi-branch Tree Visualization",
        height_px: int = 1000,
        width_pct: int = 100,
        font_size: int = 14,
        group_spacing_multiplier: float = 10.0,
        sibling_spacing_multiplier: float = 0.8,
        base_spacing_min: int = 100,
        base_spacing_max: int = 300,
        base_spacing_divisor: int = 6000,
    ):
        self.adj: dict[int, List[int]] = adjacency
        self.highlights = highlights or set()
        self.labels = labels or {}
        self.node_colors = node_colors or {}
        self.title = title
        self.font_size = font_size
        self.n_leaves = n_leaves or 0
        
        # Spacing control parameters
        self.group_spacing_multiplier = group_spacing_multiplier
        self.sibling_spacing_multiplier = sibling_spacing_multiplier  
        self.base_spacing_min = base_spacing_min
        self.base_spacing_max = base_spacing_max
        self.base_spacing_divisor = base_spacing_divisor

        # PyVis network
        self.net = Network(
            height=f"{height_px}px",
            width=f"{width_pct}%",
            directed=True,
            bgcolor="#ffffff",
            font_color="black"
        )
        
        # Calculate levels for hierarchical layout
        self.levels = self._calculate_levels()

    def _calculate_levels(self) -> Dict[int, int]:
        """Calculate hierarchical levels for nodes."""
        levels = {}
        
        # Find root nodes (nodes that are not children of any other node)
        all_children = set()
        for children in self.adj.values():
            all_children.update(children)
        
        # Find root nodes
        roots = []
        for node_id in self.adj:
            if node_id not in all_children:
                roots.append(node_id)
        
        # Assign levels using BFS from roots
        queue = deque()
        for root in roots:
            levels[root] = 0  # Root at top (level 0)
            queue.append((root, 0))
            
        while queue:
            node_id, level = queue.popleft()
            
            if node_id in self.adj:
                for child in self.adj[node_id]:
                    if child not in levels:
                        child_level = level + 1
                        levels[child] = child_level
                        queue.append((child, child_level))
        
        # Ensure all leaf nodes (entities) are at the deepest level
        max_internal_level = max((level for node_id, level in levels.items() if node_id >= self.n_leaves), default=-1)
        leaf_level = max_internal_level + 1
        
        for i in range(self.n_leaves):
            levels[i] = leaf_level  # All entities at bottom level
        return levels

    def _get_node_label(self, node_id: int) -> str:
        """Get display label for node."""
        if node_id in self.labels:
            label = self.labels[node_id]
            # Wrap long labels for better readability
            if len(label) > 15:
                words = label.split()
                if len(words) > 1:
                    mid = len(words) // 2
                    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
            return label
        return f"Node {node_id}"

    def _get_node_title(self, node_id: int) -> str:
        """Get hover title for node."""
        if node_id in self.labels:
            return f"ID: {node_id}\nLabel: {self.labels[node_id]}"
        return f"Node ID: {node_id}"

    def _get_node_color(self, node_id: int) -> str:
        """Get color for node."""
        if node_id < self.n_leaves:
            # Leaf nodes (entities) use profession-based colors if available
            if node_id in self.node_colors:
                return self.node_colors[node_id]
            else:
                return "#87CEEB"  # Light blue for leaf nodes without specific color
        else:
            # Internal nodes (categories) are all grey
            return "#808080"  # Grey for all category nodes

    def _get_node_size(self, node_id: int) -> int:
        """Determine node size."""
        if node_id in self.highlights:
            return 30  # Highlighted nodes are large
        elif node_id < self.n_leaves:
            return 15  # Leaf nodes (entities) are smaller
        elif node_id in self.levels:
            # Size based on hierarchy level (higher level = larger)
            level = self.levels[node_id]
            return max(15, 25 - level * 2)  # Root largest, decreasing by level
        else:
            return 20  # Default for internal nodes

    def _add_nodes_edges(self) -> None:
        """Add nodes and edges to the network."""
        # First, add all leaf nodes
        for i in range(self.n_leaves):
            self.net.add_node(
                i,
                label=self._get_node_label(i),
                title=self._get_node_title(i),
                color=self._get_node_color(i),
                shape="dot",
                size=self._get_node_size(i),
                level=self.levels.get(i, 0),
                physics=False,
                font={
                    "size": self.font_size,
                    "color": "black",
                    "strokeWidth": 0,
                    "strokeColor": "white"
                }
            )
        
        # Then add internal nodes and edges
        for parent, children in self.adj.items():
            # Add parent node (internal node)
            self.net.add_node(
                parent,
                label=self._get_node_label(parent),
                title=self._get_node_title(parent),
                color=self._get_node_color(parent),
                shape="dot",
                size=self._get_node_size(parent),
                level=self.levels.get(parent, 0),
                physics=False,
                font={
                    "size": self.font_size - 2,
                    "color": "black"
                }
            )

            # Add edges to all children
            for child in children:
                # Add child node if it's an internal node
                if child >= self.n_leaves:
                    self.net.add_node(
                        child,
                        label=self._get_node_label(child),
                        title=self._get_node_title(child),
                        color=self._get_node_color(child),
                        shape="dot",
                        size=self._get_node_size(child),
                        level=self.levels.get(child, 0),
                        physics=False,
                        font={
                            "size": self.font_size - 2,
                            "color": "black"
                        }
                    )
                
                # Add edge from parent to child
                self.net.add_edge(
                    parent,
                    child,
                    arrows="to",
                    color="#666666",
                    width=2
                )

    def _add_nodes_edges_with_positions(self, positions: Dict[int, Dict[str, float]]) -> None:
        """Add nodes and edges to the network with explicit positions."""
        # First, add all leaf nodes with positions
        for i in range(self.n_leaves):
            pos = positions.get(i, {"x": 0, "y": 0})
            self.net.add_node(
                i,
                label=self._get_node_label(i),
                title=self._get_node_title(i),
                color=self._get_node_color(i),
                shape="dot",
                size=self._get_node_size(i),
                x=pos["x"],
                y=pos["y"],
                physics=False,
                font={
                    "size": self.font_size,
                    "color": "black",
                    "strokeWidth": 0,
                    "strokeColor": "white"
                }
            )
        
        # Then add internal nodes and edges with positions
        for parent, children in self.adj.items():
            # Add parent node (internal node) with position
            pos = positions.get(parent, {"x": 0, "y": 0})
            self.net.add_node(
                parent,
                label=self._get_node_label(parent),
                title=self._get_node_title(parent),
                color=self._get_node_color(parent),
                shape="dot",
                size=self._get_node_size(parent),
                x=pos["x"],
                y=pos["y"],
                physics=False,
                font={
                    "size": self.font_size - 2,
                    "color": "black"
                }
            )

            # Add edges to all children
            for child in children:
                # Add child node if it's an internal node with position
                if child >= self.n_leaves:
                    child_pos = positions.get(child, {"x": 0, "y": 0})
                    self.net.add_node(
                        child,
                        label=self._get_node_label(child),
                        title=self._get_node_title(child),
                        color=self._get_node_color(child),
                        shape="dot",
                        size=self._get_node_size(child),
                        x=child_pos["x"],
                        y=child_pos["y"],
                        physics=False,
                        font={
                            "size": self.font_size - 2,
                            "color": "black"
                        }
                    )
                
                # Add edge from parent to child
                self.net.add_edge(
                    parent,
                    child,
                    arrows="to",
                    color="#666666",
                    width=2
                )

    def _calculate_positions(self) -> Dict[int, Dict[str, float]]:
        """Calculate explicit positions for nodes to improve layout."""
        positions = {}
        
        # Group nodes by level
        levels_dict = {}
        for node_id, level in self.levels.items():
            if level not in levels_dict:
                levels_dict[level] = []
            levels_dict[level].append(node_id)
        
        # Sort levels
        sorted_levels = sorted(levels_dict.keys())
        
        # Calculate positions level by level with adaptive spacing
        y_spacing = 400  # Increased vertical spacing between levels
        
        for level_idx, level in enumerate(sorted_levels):
            nodes_at_level = levels_dict[level]
            y_pos = level_idx * y_spacing
            
            if len(nodes_at_level) == 1:
                # Single node at this level - center it
                positions[nodes_at_level[0]] = {"x": 0, "y": y_pos}
            else:
                # Group by parent for better layout
                parent_groups = {}
                for node_id in nodes_at_level:
                    # Find parent of this node
                    parent = None
                    for p, children in self.adj.items():
                        if node_id in children:
                            parent = p
                            break
                    
                    parent_key = parent if parent is not None else "orphans"
                    if parent_key not in parent_groups:
                        parent_groups[parent_key] = []
                    parent_groups[parent_key].append(node_id)
                
                # Calculate adaptive spacing based on number of nodes
                total_nodes = len(nodes_at_level)  
                base_spacing = min(self.base_spacing_max, max(self.base_spacing_min, self.base_spacing_divisor // total_nodes))
                group_spacing = base_spacing * self.group_spacing_multiplier  # Spacing between different parent groups
                
                # Position groups
                total_groups = len(parent_groups)
                start_x = -(total_groups - 1) * group_spacing / 2
                
                group_idx = 0
                for parent_key, siblings in parent_groups.items():
                    # Position siblings within group
                    sibling_spacing = base_spacing * self.sibling_spacing_multiplier  # Configurable sibling spacing
                    group_center_x = start_x + group_idx * group_spacing
                    sibling_start_x = group_center_x - (len(siblings) - 1) * sibling_spacing / 2
                    
                    for sibling_idx, sibling in enumerate(siblings):
                        x_pos = sibling_start_x + sibling_idx * sibling_spacing
                        positions[sibling] = {"x": x_pos, "y": y_pos}
                    
                    group_idx += 1
        
        return positions

    def draw(self, output_path: str) -> None:
        """Generate and save the HTML visualization."""
        # Calculate explicit positions
        positions = self._calculate_positions()
        
        # Configure network options with fixed positions
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": False  # Disable hierarchical layout - we'll use fixed positions
                }
            },
            "physics": {
                "enabled": False  # Disable physics to keep fixed positions
            },
            "nodes": {
                "font": {
                    "size": self.font_size,
                    "color": "black"
                },
                "fixed": True,  # Keep nodes in fixed positions
                "chosen": False  # Disable selection effects
            },
            "edges": {
                "smooth": {
                    "enabled": True,
                    "type": "continuous",
                    "roundness": 0.2
                },
                "arrows": {
                    "to": {
                        "enabled": True,
                        "scaleFactor": 0.5
                    }
                }
            },
            "interaction": {
                "dragNodes": False  # Prevent dragging to maintain layout
            }
        }
        
        self.net.set_options(json.dumps(options))
        
        # Add nodes and edges with positions
        self._add_nodes_edges_with_positions(positions)
        
        # Generate HTML
        html_content = self.net.generate_html()
        
        # Add custom title
        title_section = f"""
        <div style="text-align: center; padding: 20px; background-color: #f0f0f0; margin-bottom: 20px;">
            <h1 style="margin: 0; color: #333;">{self.title}</h1>
        </div>
        """
        
        # Insert title after <body> tag
        html_content = html_content.replace("<body>", f"<body>{title_section}")
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)