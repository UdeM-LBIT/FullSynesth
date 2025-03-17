from .graph import Graph
from unittest import TestCase
from sowing.node import Node
from collections import namedtuple


Test = namedtuple("Test", ["graph", "neighbors", "trees", "final_leaves","final_trees"])


class GraphTest(TestCase):
    test_graphs = [
        Test(
            graph=Graph(
                nodes=set(range(1, 10)),
                edges=set([
                    (1, 2), (1, 3), (3, 4), (2, 5),
                    (2, 6), (4, 7), (4, 8), (4, 9),
                ])
            ),

            neighbors={
                1: set([2, 3]),
                2: set([5, 6]),
                3: set([4]),
                4: set([7, 8, 9]),
                5: set([]),
                6: set([]),
                7: set([]),
                8: set([]),
                9: set([]),
            },

            trees={Node(1).add(Node(2).add(Node(5)).add(Node(6))).add(Node(3).add(Node(4).add(Node(7)).add(Node(8)).add(Node(9))))},

            final_leaves = None,
            final_trees = None,
        ),
        Test(
            graph=Graph(
                nodes=set(range(1, 10)),
                edges=set([
                    (9, 6), (9, 8), (8, 7), (6, 1),
                    (6, 2), (7, 3), (7, 4), (7, 5),
                ])
            ),

            neighbors={
                1: set([]),
                2: set([]),
                3: set([]),
                4: set([]),
                5: set([]),
                6: set([1, 2]),
                7: set([3, 4, 5]),
                8: set([7]),
                9: set([6, 8]),
            },

            trees={Node(9).add(Node(6).add(Node(1)).add(Node(2))).add(Node(8).add(Node(7).add(Node(3)).add(Node(4)).add(Node(5))))},
 
            final_leaves = None,
            final_trees = None,
        ),
        Test(
            graph= Graph(
                nodes=set(range(1, 5)),
                edges=set([
                    (1, 2), (1, 3), (2, 4), (3, 4)
                ])
            ),

            neighbors={
                1:set([2,3]),
                2:set([4]),
                3:set([4]),
                4:set([]),
            },

            trees=None,
            final_leaves = None,
            final_trees = None,
        ),
    
        Test(
            graph = Graph(
                nodes=set(range(1, 15)),
                edges=set([
                    (1, 2), (1, 3), (3, 4), (2, 5),
                    (2, 6), (4, 7), (4, 8), (4, 9),
                    (8,10), (10,11), (10,12), (11,13),
                    (11,14)
                ])
            ),

            neighbors={
                1: set([2,3]),
                2: set([5,6]),
                3: set([4]),
                4: set([7,8,9]),
                5: set([]),
                6: set([]),
                7: set([]),
                8: set([10]),
                9: set([]),
                10: set([11,12]),
                11: set([13,14]),
                12: set([]),
                13: set([]),
                14: set([]),
            },

            trees = {Node(1).add(Node(2).add(Node(5)).add(Node(6))).add(Node(3).add(Node(4).add(Node(7)).add(Node(8).add(Node(10).add(Node(11).add(Node(13)).add(Node(14))).add(Node(12)))).add(Node(9))))},
            
            final_leaves = {5,7,9,12,13,14},

            final_trees ={Node(1).add(Node(5)).add(Node(4).add(Node(7)).add(Node(10).add(Node(11).add(Node(13)).add(Node(14))).add(Node(12))).add(Node(9)))},
                    ),
    ]
    
    test_3 = [Graph(
        nodes=set([1]),
        edges=set()
        ),
              Graph(
        nodes=set(range(2, 4)),
        edges=set([
            (3, 2)
            ])
        ),
              Graph(
        nodes=set(range(4, 9)),
        edges=set([
            (4, 6), (4, 5), (5, 7), (5, 8)
            ])
        )
    ]
    
    test_4 = Graph(
        nodes=set(range(1, 5)),
        edges=set([
            (1, 2), (1, 3), (2, 4), (3, 4)
        ])
    )
    
    test_5 = Graph(
        nodes=set(range(1, 8)),
        edges=set([
            (1, 4), (2, 3), (4, 5), (3, 5),
            (5, 6), (5, 7)
        ])
    )
    
    test_6 = Graph(
        nodes=set(range(1, 5)),
        edges=set([
            (1, 2), (2, 3), (3, 4), (4, 1)
        ])
    )
    test_7 = Graph(
        nodes=set(range(1, 15)),
        edges=set([
            (1, 2), (1, 3), (3, 4), (2, 5),
            (2, 6), (4, 7), (4, 8), (4, 9),
            (8,10), (10,11), (10,12), (11,13),
            (11,14)
        ])
    )
    final_leaves = {5,7,9,12,13,14}


    def test_to_neighbors(self):
        for test in self.test_graphs:
            self.assertEqual(
                test.graph.to_neighbors(),
                test.neighbors,
            )

    def test_to_forest(self):
        for test in self.test_graphs:
            if test.trees is None:
                with self.assertRaisesRegex(RuntimeError,"Not a tree"):
                    test.graph.to_forest()
            else:
                self.assertEqual(test.graph.to_forest(), test.trees)

    def test_delete_leaf_unary(self):
        for test in self.test_graphs:
            if not test.final_leaves is None:
                self.assertEqual(test.graph.get_final_tree(test.final_leaves), test.final_trees)
