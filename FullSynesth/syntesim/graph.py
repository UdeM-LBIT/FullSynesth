"""Represent our syntenies species and genes forest or tree"""
from typing import Any
from dataclasses import dataclass, field
from sowing.node import Node, Edge
from sowing.traversal import fold, depth
from immutables import Map
@dataclass
class Graph:
    nodes: set = field(default_factory=set)
    edges: set[tuple[Any, Any]] = field(default_factory=set)

    """Match nodes with their children"""
    def to_neighbors(self) -> dict[Any, set[Any]]:
        result = {node: set() for node in self.nodes}

        for start, end in self.edges:
            result[start].add(end)

        return result

    def get_roots(self):
        child_nodes = set()

        for start, end in self.edges:
            child_nodes.add(end)
        
        roots = self.nodes - child_nodes
        return roots

    """Make sure that the simulation result is a tree"""
    def to_forest(self) -> set[Node]:
        neighbors = self.to_neighbors()
        roots_name = self.get_roots() 
        forest_nodes = {label: Node(Map({"name":label})) for label in self.nodes}
        seen = set()

        for init_node in roots_name:
            stack = [init_node]
            preordre = []

            while stack:
                current = stack.pop()
                preordre.append(current)
                seen.add(current)
                for next_label in sorted(neighbors[current]):
                    if next_label in seen:
                        raise RuntimeError("Not a tree")
                    stack.append(next_label)
            for cur_label in preordre[::-1]:
                cur_node = forest_nodes[cur_label]
    
                for next_label in sorted(neighbors[cur_label]):
                    next_node = forest_nodes[next_label]
                    cur_node = cur_node.add(next_node)

                forest_nodes[cur_label] = cur_node
               

        if seen != self.nodes:
            raise RuntimeError("Cycle in forest")
       
        roots = {forest_nodes[name] for name in roots_name}
        return roots
    
    """Delete a leaf that is not in the final state"""
        
    def get_final_tree(self, final_leaves):
        def delete_leaf_unary(zipper):
             
            if zipper.is_leaf():
                if zipper.node.data["name"] not in final_leaves:
                    return zipper.replace(node = None)
                else:
                    return zipper
                
            elif len(zipper.node.edges) == 1:
                return zipper.replace(node = zipper.down().node)
                
            else:
                return zipper    
            
        roots = self.to_forest()
        final_roots = set()
        
        for tree in roots:
            final_roots.add(fold(delete_leaf_unary,depth(tree)))
       
        return final_roots
        
