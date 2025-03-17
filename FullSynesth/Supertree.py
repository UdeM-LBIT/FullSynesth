from sowing.repr.newick import *
from sowing.node import Node, Edge
from sowing.comb.supertree import Triple
from sowing import traversal
from sowing.util.partition import *
from model.history import (
    Host,
    Associate,
)
import numpy as np 

def breakup(root: Node) -> tuple[list[Node], list[Triple]]:
    """
    Break up a phylogenetic tree into triples that encode its topology.

    This implements the BreakUp algorithm from [Ng and Wormald, 1996] restricted 
    to binary trees and triples only.

    The output representation uniquely determines the input tree, disregarding
    unary nodes, repeated leaves, child order, data in internal nodes, and
    data on edges.

    The input tree can be reconstructed using the :func:`build` function.

    :param root: input tree to be broken up
    :returns: a tuple containing the list of leaves in the tree and a list of
        extracted triples
    """
    triples = []

    def extract_parts(cursor):
        children = tuple(edge.node for edge in cursor.node.edges)

        if cursor.is_leaf() or not all(
            cursor.down(i).is_leaf() for i in range(len(children))
        ):
            return cursor

        # if len(children) >= 3:
        #     # Break up fan
        #     fans.append(Fan(children))
        #     children = children[:2]

        # Break up triple
        base = cursor

        while base.sibling() == base and not base.is_root():
            base = base.up()

        if base.is_root():
            return cursor

        outgroup = next(traversal.leaves(base.sibling().node)).node
        triples.append(Triple(children, outgroup))
        return base.replace(node=children[0])

    traversal.fold(extract_parts, traversal.depth(root))
    leaves = [cursor.node for cursor in traversal.leaves(root)]
    return leaves, triples

def AllTrees(
    leaves: list[Node],
    triples: list[Triple],
    cache: {},
) -> list[Node]  | None:
    """
    Construct all binary phylogenetic trees satisfying the topology constraints given by
    a set of triple.

    This implements the AllTrees algorithm from [Ng and Wormald, 1996] restricted
    to triples only.

    :param leaves: set of leaves
    :param triples: set of triples
    :returns: constructed trees, or None if the constraints are inconsistent
    """
    if not leaves:
        return None

    if OneTreeBinary(leaves,triples) is None:
        return None
    
    if len(leaves) == 1:
        return [leaves[0]]

    if len(leaves) == 2:
        left, right = leaves
        return [Node(Associate()).add(left).add(right)]
    
    #Check if the solution is cached already
    inputCache = (frozenset(leaves),frozenset(triples))
    if inputCache in cache:
        return cache[inputCache]

    partition = Partition(leaves)
    listSuperTree = []
    # Merge groups for triples
    for triple in triples:
        partition.union(*triple.ingroup)

    if len(partition) <= 1:
        return None

    # Recursively build subtrees for each child
    counters = np.zeros(len(partition.groups()))
    done = False
    
    while not done :
        
        groupLeft = []
        groupRight = []
        c = 0
        
        for group in partition.groups():
            match counters[c]:
                case 0: #group goes left
                    groupLeft = groupLeft + group
                
                case 1: #group goes Right
                    groupRight = groupRight + group
        
            c += 1 
        if((len(groupLeft)>0) and (len(groupRight)>0)):
            subTriplesLeft = [triple for triple in triples if triple.is_in(groupLeft)]
            subTreesLeft = AllTrees(groupLeft, subTriplesLeft,cache)

            subTriplesRight = [triple for triple in triples if triple.is_in(groupRight)]
            subTreesRight = AllTrees(groupRight, subTriplesRight,cache)
            
            for treeLeft in subTreesLeft:
                for treeRight in subTreesRight:
                    root = Node(Associate())
                    root = root.add(treeLeft)
                    root = root.add(treeRight)
                    listSuperTree.append(root)
        
        ApplyNextConfig(counters)
        if counters[len(partition.groups())-1] == 1: 
            done = True
            
    
    #Keep solution in cache
    cache[inputCache] = listSuperTree
    
    return listSuperTree

def AllTreesNumber(
    leaves: list[Node],
    triples: list[Triple],
    cache: {},
):
    """
    Return the number of binary phylogenetic trees satisfying the topology constraints given by
    a set of triple.

    This implements the AllTrees algorithm from [Ng and Wormald, 1996] restricted
    to triples only.

    :param leaves: set of leaves
    :param triples: set of triples
    :returns: number of possible supertrees, or 0 if the constraints are inconsistent
    """
    if not leaves:
        return 0

    if OneTreeBinary(leaves,triples) is None:
        return 0
    
    if len(leaves) == 1:
        return 1

    if len(leaves) == 2:
        return 1
    
    #Check if the solution is cached already
    inputCache = (frozenset(leaves),frozenset(triples))
    if inputCache in cache:
        return cache[inputCache]

    partition = Partition(leaves)
    numberSuperTree = 0
    # Merge groups for triples
    for triple in triples:
        partition.union(*triple.ingroup)

    if len(partition) <= 1:
        return None

    # Recursively build subtrees for each child
    counters = np.zeros(len(partition.groups()))
    done = False
    
    while not done :
        
        groupLeft = []
        groupRight = []
        c = 0
        
        for group in partition.groups():
            match counters[c]:
                case 0: #group goes left
                    groupLeft = groupLeft + group
                
                case 1: #group goes Right
                    groupRight = groupRight + group
        
            c += 1 
        if((len(groupLeft)>0) and (len(groupRight)>0)):
            subTriplesLeft = [triple for triple in triples if triple.is_in(groupLeft)]
            subTreesLeft = AllTreesNumber(groupLeft, subTriplesLeft,cache)

            subTriplesRight = [triple for triple in triples if triple.is_in(groupRight)]
            subTreesRight = AllTreesNumber(groupRight, subTriplesRight,cache)
            
            numberSuperTree = numberSuperTree + subTreesLeft*subTreesRight
        
        ApplyNextConfig(counters)
        if counters[len(partition.groups())-1] == 1: 
            done = True
            
    
    #Keep solution in cache
    cache[inputCache] = numberSuperTree
    
    return numberSuperTree

def ApplyNextConfig(counters):
    done = False
    cindex = 0
    
    while not done :
        if len(counters) <= cindex:
            done = True
            break
        counters[cindex] += 1
        if counters[cindex] > 1:
            counters[cindex] = 0
            cindex += 1
        else:
            done = True

def OneTreeBinary(
    leaves: list[Node],
    triples: list[Triple] = [],
) -> Node | None:
    """
    Construct a phylogenetic tree satisfying the topology constraints given by
    a set of triple.

    The returned tree is a binary tree compatible with
    all the triples and fans given as input, if such a tree exists.

    This implements the OneTree algorithm from [Ng and Wormald, 1996] restricted
    to triples only.

    :param leaves: set of leaves
    :param triples: set of triples
    :returns: constructed tree, or None if the constraints are inconsistent
    """
    if not leaves:
        return None

    if len(leaves) == 1:
        return leaves[0]

    if len(leaves) == 2:
        left, right = leaves
        return Node(Associate()).add(left).add(right)

    partition = Partition(leaves)

    # Merge groups for triples
    for triple in triples:
        partition.union(*triple.ingroup)

    if len(partition) <= 1:
        return None

    # Recursively build subtrees for each group
    root = Node(Associate())
    groupLeft = []
    groupRight = []
    
    groupLeft = groupLeft + partition.groups()[0]
    
    for group in partition.groups():
        if group != partition.groups()[0]:
            groupRight = groupRight + group
    
    subTriplesLeft = [triple for triple in triples if triple.is_in(groupLeft)]
    subTreeLeft = OneTreeBinary(groupLeft, subTriplesLeft)

    subTriplesRight = [triple for triple in triples if triple.is_in(groupRight)]
    subTreeRight = OneTreeBinary(groupRight, subTriplesRight)

    root = root.add(subTreeLeft)
    root = root.add(subTreeRight)

    return root

def supertree(trees: set[Node]) -> Node | None:
    """
    Build a supertree from a set of phylogenetic trees.

    The returned tree is the smallest tree compatible with every tree of
    the input, if such a tree exists.

    :param tree: any number of tree to build a supertree from
    :returns: constructed tree, or None if input trees are incompatible
    """
    # Use dictionaries as sets to merge parts while preserving ordering
    all_leaves = {}
    all_triples = {}

    for tree in trees:
        leaves, triples = breakup(tree)
        all_leaves.update(dict.fromkeys(leaves))
        all_triples.update(dict.fromkeys(triples))

    return OneTreeBinary(all_leaves.keys(), all_triples.keys())

def all_binary_supertrees(trees: set[Node]) -> Iterable[Node]:
    """
    Compute all the binary supertrees compatible with each input tree.
    
    :param tree: any number of tree to build the supertrees from
    :returns: all binary trees that are compatible with each input tree
        (may be empty)
    """
    # Use dictionaries as sets to merge parts while preserving ordering
    all_leaves = {}
    all_triples = {}

    for tree in trees:
        leaves, triples = breakup(tree)
        all_leaves.update(dict.fromkeys(leaves))
        all_triples.update(dict.fromkeys(triples))

    return AllTrees(all_leaves.keys(), all_triples.keys(),{})

def all_binary_supertrees_Number(trees: set[Node]) -> Iterable[Node]:
    """
    Compute the number of binary supertrees compatible with each input tree.
    
    :param tree: any number of tree to build the supertrees from
    :returns: the number of binary trees that are compatible with each input tree
        (may be 0)
    """
    # Use dictionaries as sets to merge parts while preserving ordering
    all_leaves = {}
    all_triples = {}

    for tree in trees:
        leaves, triples = breakup(tree)
        all_leaves.update(dict.fromkeys(leaves))
        all_triples.update(dict.fromkeys(triples))

    return AllTreesNumber(all_leaves.keys(), all_triples.keys(),{})