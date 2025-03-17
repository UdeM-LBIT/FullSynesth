from dataclasses import dataclass, field
from typing import List, Mapping, Set
from compute.util import EventCosts, make_cost_algebra, EventCosts,history_counter,history_unit_generator
from sowing.node import Node
from sowing.traversal import depth,leaves
from sowing import traversal
from sowing.indexed import IndexedTree
from sowing.zipper import Zipper
from model.history import (
    Host,
    Associate,
    parse_tree,
    Reconciliation
)
from compute.superdtlx.recurrence import reconcile
from compute.superdtlx.contents import Contents, EXTRA_CONTENTS
from compute.superdtlx.paths import make_path
import numpy as np 
from numpy import inf
import time
from utils.algebras import make_single_selector, make_product

@dataclass(frozen=True, repr=False)
class HistoryInputGeneTrees:
    """Input to the super-reconciliation problem with gene trees in input."""

    # Gene trees to create synteny tree from
    gene_trees: List[Node[Associate, None]]

    # Species tree
    species_tree: Node[Host, None]
        
    # Costs of evolutionary events
    costs: EventCosts 

def FullSynesth(srec_input: HistoryInputGeneTrees): 
    """
    Compute a minimum-cost unordered history from a set of 
    gene trees using a generalization of the SuperGeneTree algorithm 
    (https://github.com/UdeM-LBIT/SuGeT/blob/master/SuperGeneTrees/supergenetreemaker.h). 
    This algorithm can handle all kinds of events (including transfers) 
    and cost values.

    :param srec_input: objects of the history with consistent 
        gene trees in input
    :returns: The cost of an optimal history and a corresponding synteny supertree
    """
    
    #Pre-process the species tree for fast answering of repeated lowest common ancestor queries
    #speciesTree = graft_unsampled_hostsV2(srec_input.species_tree)
    speciesTree = srec_input.species_tree
    indexedSpeciesTree = IndexedTree(speciesTree)
    
    inputTrees = []
    for tree in srec_input.gene_trees:
        inputTrees.append(tree)

    #Find a Core of the gene trees in input
    core, notCore = ComputeCore(inputTrees)
    
    #Pre-process the trees not in the core for fast answering of repeated lowest common ancestor queries
    indexedNotCore = []
    for tree in notCore:
        indexedNotCore.append(IndexedTree(tree))
        
    #Compute a mapping from the nodes in trees in the core to the nodes in trees out of the core
    #to test if a given bipartition is compatible with the trees not in the core    
    mappingCoreNotCore = ComputeMappingCoreNotCore(core, indexedNotCore)    
    
    #Compute all intersections bewtween subtrees of the trees in the core
    intersection = set()
    ComputeAllIntersections(intersection,core)
    
    #Set of gene trees in the core
    trees = set(core)
    
    #Cache
    recursionCache = {}
    
    #Counter of gene families occurences:
    geneFamilyNumberDict = {}
    geneNamesSet = set()
    for tree in trees:
        for nodeZipper in leaves(tree):
            nameCurrent = nodeZipper.node.data.name
            if nameCurrent not in geneNamesSet: 
                geneNamesSet.add(nameCurrent)
                for geneFamily in nodeZipper.node.data.contents:
                    if geneFamily in geneFamilyNumberDict:
                        geneFamilyNumberDict[geneFamily] = geneFamilyNumberDict[geneFamily] + 1
                    else:
                        geneFamilyNumberDict[geneFamily] = 1
    
    #Lists of species:
    speciesListPostOrder = []
    for nodeZipper in depth(speciesTree):
        speciesListPostOrder.append(nodeZipper.node.data.name)
    
    speciesListPreOrder = []
    for nodeZipper in depth(speciesTree,preorder=True):
        speciesListPreOrder.append(nodeZipper.node.data.name)
    
    #Find the optimal gene tree
    c, minInOut, inAlt, outAlt, geneFamilyCounterDict = FullSynesthRecursion( 
                                                        geneFamilyNumberDict,
                                                        recursionCache,
                                                        srec_input.costs,
                                                        intersection,
                                                        mappingCoreNotCore,
                                                        indexedNotCore,
                                                        indexedSpeciesTree,
                                                        trees,
                                                        speciesListPostOrder,
                                                        speciesListPreOrder)

    currentMin = (inf,"")
    for species in speciesListPostOrder:
        if c[("min",species)][0] < currentMin[0]:
            currentMin = c[("min",species)]
    
    
    newickMinGeneTree = currentMin[1] + ";"
    minGeneTree = parse_tree(Associate, newickMinGeneTree)
      
    return currentMin[0], minGeneTree

def FullSynesthRecursion(
    geneFamilyNumberDict,
    recursionCache,
    costs: EventCosts,
    intersection : set(),
    mappingCoreNotCore,
    indexedNotCore,
    indexedSpeciesTree : IndexedTree,
    trees : Set[Node],
    speciesListPostOrder : [str],
    speciesListPreOrder  : [str]
    ) :
        
    #Check if the solution is cached already
    inputCache = frozenset(trees)
    if inputCache in recursionCache:
        return recursionCache[inputCache]
     
    #Stop Condition
    boolStop = True
    leaflabel = ""
    i = 0
    for tree in trees:
        if tree.edges != ():
            boolStop = False
            break
        else:  
            if i == 0:
                leaf = LeafToNewick(tree)
                leaflabel = tree.data.name
                synteny = tree.data.contents
                host = tree.data.host  
                geneFamilyCounterDict = {}
                for geneFamily in synteny:
                    geneFamilyCounterDict[geneFamily] = 1
            elif tree.data.name != leaflabel:
                boolStop = False
                break
        i += 1
    
    #If the current node is a leaf
    if boolStop:
        c = {}
        minInOut = {}
        inAlt = {}
        outAlt = {}
        
        for species in speciesListPostOrder:               
            if species == host:
                c[("min",species)] = 0, leaf
                c[("extra",species)] = inf, leaf
            else:
                c[("min",species)] = inf, leaf
                c[("extra",species)] = inf, leaf
            if indexedSpeciesTree.is_ancestor_of(species,host) :
                inAlt[("min",species)] = 0, leaf
                inAlt[("extra",species)] = costs.loss, leaf
            else:
                inAlt[("min",species)] = inf, leaf
                inAlt[("extra",species)] = inf, leaf
        
        for species in speciesListPreOrder:           
            speciesZipper = indexedSpeciesTree(species)
            if speciesZipper.is_root():
                outAlt[("min",species)] = inf, leaf
                outAlt[("extra",species)] = inf, leaf
            else:
                outAlt[("min",species)] = min(outAlt[("min",speciesZipper.up().node.data.name)],
                                              inAlt[("min",speciesZipper.sibling().node.data.name)]
                                             )
                outAlt[("extra",species)] = min(outAlt[("extra",speciesZipper.up().node.data.name)],
                                              inAlt[("extra",speciesZipper.sibling().node.data.name)]
                                             )
    
        minInOut = ComputeMinInOut(speciesListPostOrder,indexedSpeciesTree,c,inAlt,outAlt,costs)
        return c, minInOut, inAlt, outAlt, geneFamilyCounterDict
    
    #Each tree has 4 ways of being split...each way has its index
    #0 = send whole tree left, 1 = send left side left
    #2 = send whole tree right, 3 = send right side left
    #The deal is, if we fix config 0 for the last tree,
    #all 4^{nbtrees - 1}config combos for other trees are non-redundant
    #Then, same if we fix config 1.
    #Afterwards, at config 2 or 3 for the last tree,
    #everything is symmetric to something we've seen before.
    
    nbTrees = len(trees)
    counters = np.zeros(nbTrees)
    
    ApplyNextConfig(counters)   #skip the all-zeros config
    
    done = False
    
    cCurrent = {}
    minInOutCurrent = {}
    inAltCurrent = {}
    outAltCurrent = {}
    
    contentsSet = {"min", "extra"}
    
    for species in speciesListPostOrder:               
        for contents in contentsSet:
            cCurrent[(contents, species)] = inf,""
            minInOutCurrent[(contents, species)] = inf,""
            inAltCurrent[(contents, species)] = inf,""
            outAltCurrent[(contents, species)] = inf,""
            
            
    while not done :
        
        treesLeft = set()
        treesRight = set()
        
        isConfigFine = True #some configs are impossible if a tree is a leaf
        #this loop sends what goes left to the left, and what goes right right
        #we also check that no leaf gets split
           
        #this loop sends what goes left to the left, and what goes right right
        #we also check that no leaf gets split
        c = 0
        for tree in trees:
            #we can't split a leaf
            if (tree.edges == ()) and (counters[c] == 1 or counters[c] == 3):
                isConfigFine = False
                break
            
            match counters[c]:
                case 0: #all left
                    treesLeft.add(tree)
                
                case 1: #left goes left
                    treesLeft.add(tree.edges[0].node)
                    treesRight.add(tree.edges[1].node)

                case 2: #all right
                    treesRight.add(tree)
                    
                case 3: #right goes left
                    treesLeft.add(tree.edges[1].node)
                    treesRight.add(tree.edges[0].node)

            
            c += 1 
        if (isConfigFine and len(treesLeft) > 0 
                        and len(treesRight) > 0 
                        and not IsPartitionIntersecting(intersection,treesLeft,treesRight)      #Check intersection 
                        and IsPartitionCompatibleWithNotCore(mappingCoreNotCore,indexedNotCore,treesLeft,treesRight)):  
                                                                                                #Check compatibility 
                                                                                                #with trees out of the core                              
             
            #Compute the costs associated to SuperReconciliation left     
            cLeft, minInOutLeft, inAltLeft, outAltLeft, geneFamilyCounterDictLeft = FullSynesthRecursion(
                                                                            geneFamilyNumberDict,
                                                                            recursionCache,
                                                                            costs,
                                                                            intersection,
                                                                            mappingCoreNotCore,
                                                                            indexedNotCore,
                                                                            indexedSpeciesTree,
                                                                            treesLeft,
                                                                            speciesListPostOrder,
                                                                            speciesListPreOrder)            
            
            
            
            #Compute the costs associated to SuperReconciliation right
            cRight, minInOutRight, inAltRight, outAltRight, geneFamilyCounterDictRight = FullSynesthRecursion(
                                                                            geneFamilyNumberDict,
                                                                            recursionCache,
                                                                            costs,
                                                                            intersection,
                                                                            mappingCoreNotCore,
                                                                            indexedNotCore,
                                                                            indexedSpeciesTree,
                                                                            treesRight,
                                                                            speciesListPostOrder,
                                                                            speciesListPreOrder)            
            
            #Compute minimal contents
            geneFamilyCounterDict = {}
            for key in geneFamilyCounterDictLeft:
                geneFamilyCounterDict[key] = geneFamilyCounterDictLeft[key] 
            for key in geneFamilyCounterDictRight:
                if key in geneFamilyCounterDict:
                    geneFamilyCounterDict[key] = geneFamilyCounterDict[key] + geneFamilyCounterDictRight[key]
                else:
                    geneFamilyCounterDict[key] = geneFamilyCounterDictRight[key]
                    
            contentsLeft = set()
            for key in geneFamilyCounterDictLeft:
                contentsLeft.add(key)
                    
            contentsRight = set()
            for key in geneFamilyCounterDictRight:
                contentsRight.add(key)
                    
            contentsP = set()
            for key in geneFamilyCounterDict:
                contentsP.add(key)
                    
            #Remove gene families gained below the current node
            for key in geneFamilyCounterDictLeft:
                if geneFamilyCounterDictLeft[key] == geneFamilyNumberDict[key]:
                    contentsP.remove(key)
            for key in geneFamilyCounterDictRight:
                if geneFamilyCounterDictRight[key] == geneFamilyNumberDict[key]:
                    contentsP.remove(key)                           
            
            costsTr = min(costs.transfer_duplication,costs.transfer_cut)
            
            #C
            cSpe = (inf,"")
            cDup = (inf,"")
            cCut = (inf,"")
            cTrDup = (inf,"")
            cTrCut = (inf,"")
            c = {}             
            inAlt = {}   
            
            #Compute possible contents
            contentsPExtra = set()
            for family in contentsP:
                contentsPExtra.add(family)
            contentsPExtra.add("extra")
            MEdict = {}
            MEdict[("L","P","min")] = ME(contentsLeft, contentsP)
            MEdict[("L","P","extra")] = ME(contentsLeft, contentsPExtra)
            MEdict[("R","P","min")] = ME(contentsRight, contentsP)
            MEdict[("R","P","extra")] = ME(contentsRight, contentsPExtra)
            MEdict[("L","P-R","min")] = ME(contentsLeft, contentsP - contentsRight)
            MEdict[("L","P-R","extra")] = ME(contentsLeft, contentsPExtra - contentsRight)
            MEdict[("R","P-L","min")] = ME(contentsRight, contentsP - contentsLeft)
            MEdict[("R","P-L","extra")] = ME(contentsRight, contentsPExtra - contentsLeft)   
            
            for species in speciesListPostOrder:               
                speciesZipper = indexedSpeciesTree(species)
                for contents in contentsSet:
                      
                    #Speciation
                    if speciesZipper.is_leaf():
                        cSpe = inf,""
                    else:
                        leftChild = speciesZipper.down(0).node.data.name
                        rightChild = speciesZipper.down(1).node.data.name                
                        
                        cSpe = min(AddTuples(minInOutLeft[(MEdict[("L","P",contents)],leftChild)], 
                                                                    minInOutRight[(MEdict[("R","P",contents)],rightChild)]),
                                                        AddTuples(minInOutRight[(MEdict[("R","P",contents)],leftChild)], 
                                                                    minInOutLeft[(MEdict[("L","P",contents)],rightChild)])
                                                        )                    
                    #Duplication
                    cDup = AddTupleInt(min(AddTuples(minInOutLeft[("min",species)], 
                                                                    minInOutRight[(MEdict[("R","P",contents)],species)]),
                                           AddTuples(minInOutLeft[(MEdict[("L","P",contents)],species)], 
                                                                    minInOutRight[("min",species)])
                    
                                            ), costs.duplication)
                    #Cut
                    if not contentsLeft.isdisjoint(contentsRight):
                        cCut = inf,""
                    else:
                        cCut = AddTupleInt(min(AddTuples(minInOutLeft[("min",species)], 
                                                    minInOutRight[(MEdict[("R","P-L",contents)],species)]),
                                              AddTuples(minInOutLeft[(MEdict[("L","P-R",contents)],species)], 
                                                    minInOutRight[("min",species)])
                                            ), costs.cut)
                    #Transfer-Dup
                    if speciesZipper.is_root():
                        cTrDup = inf,""
                    else:
                        cTrDup = AddTupleInt(min(AddTuples(minInOutLeft[(MEdict[("L","P",contents)],species)],
                                                                           min(outAltRight[("min",species)],
                                                                               AddTupleInt(inAltRight[("min",species)],costsTr)))
                                                ,AddTuples(min(outAltLeft[("min",species)],
                                                               AddTupleInt(inAltLeft[("min",species)],costsTr))           
                                                           ,minInOutRight[(MEdict[("R","P",contents)],species)])
                                            ),costs.transfer_duplication)

                    #Transfer-Cut
                    if speciesZipper.is_root() or (not contentsLeft.isdisjoint(contentsRight)):
                        cTrCut = inf,""
                    elif speciesZipper.up().is_root() and (not speciesZipper.node.data.sampled) :
                        Rll = speciesZipper.sibling().down(0).node.data.name
                        Rlr = speciesZipper.sibling().down(1).node.data.name
                        cTrCut =  AddTupleInt(min(AddTuples(minInOutLeft[(MEdict[("L","P-R",contents)],species)],
                                                           min(outAltRight[("min",species)],
                                                                               AddTupleInt(inAltRight[("min",species)],costsTr)))
                                                ,AddTuples(min(outAltLeft[("min",species)],
                                                               AddTupleInt(inAltLeft[("min",species)],costsTr))           
                                                           ,minInOutRight[(MEdict[("R","P-L",contents)],species)])
                                                ,AddTuples(minInOutLeft[("min",species)],
                                                              min(outAltRight[(MEdict[("R","P-L",contents)],species)],
                                                                  AddTupleInt(inAltRight[("min",species)],costsTr),
                                                                  AddTupleInt(min(inAltRight[("min",Rll)],
                                                                                  inAltRight[("min",Rlr)]                                                      
                                                                                    ),costsTr)
                                                            ))
                                                ,AddTuples(minInOutRight[("min",species)],
                                                              min(outAltLeft[(MEdict[("L","P-R",contents)],species)],
                                                                  AddTupleInt(inAltLeft[("min",species)],costsTr),
                                                                  AddTupleInt(min(inAltLeft[("min",Rll)],
                                                                                  inAltLeft[("min",Rlr)]                                                      
                                                                                    ),costsTr)
                                                            ))
                                                 ),costs.transfer_cut)
                    elif speciesZipper.up().is_root():
                        cTrCut =  AddTupleInt(min(AddTuples(minInOutLeft[(MEdict[("L","P-R",contents)],species)],
                                                           min(outAltRight[("min",species)],
                                                                               AddTupleInt(inAltRight[("min",species)],costsTr)))
                                                ,AddTuples(min(outAltLeft[("min",species)],
                                                               AddTupleInt(inAltLeft[("min",species)],costsTr))           
                                                           ,minInOutRight[(MEdict[("R","P-L",contents)],species)])
                                                ,AddTuples(minInOutLeft[("min",species)],
                                                              min(outAltRight[(MEdict[("R","P-L",contents)],species)],
                                                                  AddTupleInt(inAltRight[("min",species)],costsTr)
                                                            ))
                                                ,AddTuples(minInOutRight[("min",species)],
                                                              min(outAltLeft[(MEdict[("L","P-R",contents)],species)],
                                                                  AddTupleInt(inAltLeft[("min",species)],costsTr)
                                                            ))
                                                 ),costs.transfer_cut)
                    else:
                        cTrCut =  AddTupleInt(min(AddTuples(minInOutLeft[(MEdict[("L","P-R",contents)],species)],
                                                           min(outAltRight[("min",species)],
                                                                               AddTupleInt(inAltRight[("min",species)],costsTr)))
                                                ,AddTuples(min(outAltLeft[("min",species)],
                                                               AddTupleInt(inAltLeft[("min",species)],costsTr))           
                                                           ,minInOutRight[(MEdict[("R","P-L",contents)],species)])
                                                ,AddTuples(minInOutLeft[("min",species)],
                                                              min(outAltRight[(MEdict[("R","P-L",contents)],species)],
                                                                  AddTupleInt(inAltRight[("min",species)],costsTr),
                                                                  AddTupleInt(outAltRight[("min",species)],costsTr)
                                                            ))
                                                ,AddTuples(minInOutRight[("min",species)],
                                                              min(outAltLeft[(MEdict[("L","P-R",contents)],species)],
                                                                  AddTupleInt(inAltLeft[("min",species)],costsTr),
                                                                  AddTupleInt(outAltLeft[("min",species)],costsTr)
                                                            ))
                                                 ),costs.transfer_cut) 
                    #C
                    c[(contents, species)] = min(cSpe,cDup,cCut,cTrDup,cTrCut)

                    
            #inAlt        
            for species in speciesListPostOrder:
                speciesZipper = indexedSpeciesTree(species)
                for contents in contentsSet:                                  
                    if speciesZipper.is_leaf():
                        inAlt[(contents, species)] = min(c[(contents, species)],AddTupleInt(c[("min", species)],costs.loss))
                    else:
                        leftChild = speciesZipper.down(0).node.data.name
                        rightChild = speciesZipper.down(1).node.data.name    
                        inAlt[(contents, species)] = min(inAlt[(contents, leftChild)],
                                                         inAlt[(contents, rightChild)],
                                                         c[(contents, species)],
                                                         AddTupleInt(c[("min", species)],costs.loss)
                                                        )
                             
            #OutAlt
            outAlt = {}
            for species in speciesListPreOrder:               
                speciesZipper = indexedSpeciesTree(species)
                for contents in contentsSet:
                    if speciesZipper.is_root():
                        outAlt[(contents, species)] = inf,""
                    else:
                        parent  = speciesZipper.up().node.data.name
                        sibling = speciesZipper.sibling().node.data.name
                        outAlt[(contents, species)] = min(outAlt[(contents, parent)],inAlt[(contents, sibling)])
                    
            #minInOut       
            minInOut = ComputeMinInOut(speciesListPostOrder,indexedSpeciesTree,c,inAlt,outAlt,costs)                         
              
            for species in speciesListPostOrder:               
                for contents in contentsSet:
                    if c[(contents, species)] < cCurrent[(contents, species)]:
                        cCurrent[(contents, species)] = c[(contents, species)]
                    if inAlt[(contents, species)] < inAltCurrent[(contents, species)]:
                        inAltCurrent[(contents, species)] = inAlt[(contents, species)]
                    if outAlt[(contents, species)] < outAltCurrent[(contents, species)]:
                        outAltCurrent[(contents, species)] = outAlt[(contents, species)]
                    if minInOut[(contents, species)] < minInOutCurrent[(contents, species)]:
                        minInOutCurrent[(contents, species)] = minInOut[(contents, species)]
                        
        
        ApplyNextConfig(counters)
        
        if counters[nbTrees - 1] > 1: #see long comment above
            done = True
           
    #Keep solution in cache
    recursionCache[inputCache] = cCurrent, minInOutCurrent, inAltCurrent, outAltCurrent, geneFamilyCounterDict
    
    return cCurrent, minInOutCurrent, inAltCurrent, outAltCurrent, geneFamilyCounterDict

def ComputeAllIntersections(intersection : set(),trees : List[Node]):
    for i in range(len(trees)):
        for j in range(i+1,len(trees)):
            ComputeIntersections(intersection, trees[i],trees[j])

def ComputeIntersections(intersection : set(), tree1 : Node, tree2 : Node):
    for node1Zipper in depth(tree1):
        for node2Zipper in depth(tree2):
            node1 = node1Zipper.node
            node2 = node2Zipper.node
            intersect = False  
            if node1Zipper.is_leaf() and node2Zipper.is_leaf():
                if node1.data.name == node2.data.name:
                    intersect = True
            elif not node1Zipper.is_leaf():
                node1_l = node1.edges[0].node
                node1_r = node1.edges[1].node
                if ((node1_l,node2) in intersection) or ((node1_r,node2) in intersection):
                    intersect = True
            else: #node1 is a leaf, node2 is not
                node2_l = node2.edges[0].node
                node2_r = node2.edges[1].node
                if ((node2_l,node1) in intersection) or ((node2_r,node1) in intersection):
                    intersect = True
            
            if intersect:
                intersection.add((node1,node2))
                intersection.add((node2,node1))

def ApplyNextConfig(counters):
    done = False
    cindex = 0
    
    while not done :
        if len(counters) <= cindex:
            done = True
            break
        counters[cindex] += 1
        if counters[cindex] > 3:
            counters[cindex] = 0
            cindex += 1
        else:
            done = True

def IsPartitionIntersecting(intersection : set(), # Set[(Node,Node)]
                            treesLeft : Set[Node], 
                            treesRight : Set[Node]):
    for tree1 in treesLeft:
        for tree2 in treesRight:
            if (tree1, tree2) in intersection:
                return True
    
    return False

def AddTuples(Tuple1, Tuple2):
    cost = Tuple1[0] + Tuple2[0]
    tree = "(" + Tuple1[1] + "," + Tuple2[1] + ")"
    return cost, tree

def AddTupleInt(Tuple, cost):
    cost = Tuple[0] + cost
    tree = Tuple[1]
    return cost, tree

def LeafToNewick(leaf):
    newick = leaf.data.name
    newick += "[&host=" + leaf.data.host
    newick += ",contents='{"
    for content in leaf.data.contents:
        newick += "''" + content + "''" + ","
    newick += "}']"
    return newick

def ComputeMinInOut(speciesListPostOrder,indexedSpeciesTree,c,inAlt,outAlt,costs):
    contentsSet = {"min", "extra"}
    in01 = {}
    in02 = {}
    in03 = {}
    in11 = {}
    in12 = {}
    in21 = {}
    in22 = {}
    in23 = {}
    In = {}
    out11 = {}
    out12 = {}
    out22 = {}
    out23 = {}
    Out = {}
    minInOut = {}
    
    for species in speciesListPostOrder:
        speciesZipper = indexedSpeciesTree(species)
        isSampled = speciesZipper.node.data.sampled
        for contents in contentsSet :
            #In
            if speciesZipper.is_leaf():
                
                in01[(contents, species)] = min(c[(contents, species)],AddTupleInt(c[("min", species)], min(costs.loss, costs.transfer_cut)))
                
                if not(isSampled):
                    minCostDupCut = min(costs.duplication, costs.cut)
                    in02[(contents, species)] = min(c[(contents, species)],AddTupleInt(c[("min", species)], minCostDupCut))
                    #in02[(contents, species)] = min(c[(contents, species)],AddTupleInt(c[("min", species)], costs.duplication))
                    #in03[(contents, species)] = min(c[(contents, species)],AddTupleInt(c[("min", species)], costs.cut))
                else:
                    in02[(contents, species)] = inf,""
                    #in03[(contents, species)] = inf,""

                in11[(contents, species)] = inf,""
                in12[(contents, species)] = inf,""   
                
                in21[(contents, species)] = AddTupleInt(c[("min", species)],2*costs.transfer_duplication + isSampled*costs.loss)
                in22[(contents, species)] = AddTupleInt(c[("min", species)],costs.transfer_duplication + costs.transfer_cut)
                in23[(contents, species)] = AddTupleInt(c[("min", species)],2*costs.transfer_cut)
            else:
                leftChild = speciesZipper.down(0).node.data.name
                rightChild = speciesZipper.down(1).node.data.name
                isSampledLeft = speciesZipper.down(0).node.data.sampled
                isSampledRight = speciesZipper.down(1).node.data.sampled

                if speciesZipper.is_root():
                    in01[(contents, species)] = min(AddTupleInt(in01[(contents, leftChild)], isSampledRight*min(costs.loss, costs.transfer_cut)),
                                               AddTupleInt(in01[(contents, rightChild)],isSampledLeft *min(costs.loss, costs.transfer_cut)),
                                               c[(contents, species)],
                                               AddTupleInt(c[("min", species)], min(costs.loss,min(costs.duplication, costs.cut)+ costs.transfer_cut)))
                    
                    in21[(contents, species)] = inf,""
                    
                    in22[(contents, species)] = AddTupleInt(min(inAlt[("min", leftChild)],
                                                            AddTupleInt(c[("min",rightChild)] ,costs.loss)
                                                               ),costs.transfer_duplication + costs.transfer_cut)

                    in23[(contents, species)] = AddTupleInt(min(inAlt[("min", leftChild)],
                                                            AddTupleInt(c[("min",rightChild)] ,costs.loss)
                                                               ),2*costs.transfer_cut)
                else:
                    in21[(contents, species)] = AddTupleInt(inAlt[("min", species)],2*costs.transfer_duplication + costs.loss)
                    
                    in22[(contents, species)] = AddTupleInt(inAlt[("min", species)],costs.transfer_duplication + costs.transfer_cut)
                    
                    in23[(contents, species)] = AddTupleInt(inAlt[("min", species)],2*costs.transfer_cut)                    
                
                    in01[(contents, species)] = min(AddTupleInt(in01[(contents, leftChild)], isSampledRight*min(costs.loss, costs.transfer_cut)),
                                               AddTupleInt(in01[(contents, rightChild)],isSampledLeft *min(costs.loss, costs.transfer_cut)),
                                               c[(contents, species)],
                                               AddTupleInt(c[("min", species)], min(costs.loss, costs.transfer_cut)))
                
                in02[(contents, species)] = min(AddTupleInt(in02[(contents, leftChild)], isSampledRight*costs.loss),
                                               AddTupleInt(in02[(contents, rightChild)],isSampledLeft *costs.loss))
                
                #in03[(contents, species)] = min(AddTupleInt(in03[(contents, leftChild)], isSampledRight*costs.loss),
                #                               AddTupleInt(in03[(contents, rightChild)],isSampledLeft *costs.loss))
                
                in11[(contents, species)] = AddTupleInt(min(AddTupleInt(inAlt[("min",leftChild)],isSampledRight*costs.loss + isSampledLeft *costs.loss ),
                                                            AddTupleInt(inAlt[("min",rightChild)],isSampledRight*costs.loss + isSampledLeft *costs.loss )
                                                           ),costs.transfer_duplication)
                
                in12[(contents, species)] = AddTupleInt(min(AddTupleInt(inAlt[("min",leftChild)],isSampledRight*costs.loss + isSampledLeft *costs.loss ),
                                                            AddTupleInt(inAlt[("min",rightChild)],isSampledRight*costs.loss + isSampledLeft *costs.loss ),
                                                            AddTupleInt(inAlt[(contents,leftChild)],isSampledLeft *costs.loss ),
                                                            AddTupleInt(inAlt[(contents,rightChild)],isSampledRight*costs.loss)
                                                           ),costs.transfer_cut)

            In[(contents, species)] = min(in01[(contents, species)],in02[(contents, species)],#in03[(contents, species)],
                                          in11[(contents, species)],in12[(contents, species)],in21[(contents, species)]
                                          ,in22[(contents, species)],in23[(contents, species)])
            
            #Out
            out11[(contents, species)] = AddTupleInt(outAlt[("min",species)],costs.transfer_duplication + isSampled*costs.loss)

            out12[(contents, species)] = AddTupleInt(min(AddTupleInt(outAlt[("min",species)], isSampled*costs.loss),
                                                        outAlt[(contents, species)]
                                                        ),costs.transfer_cut)
            
            out22[(contents, species)] = AddTupleInt(outAlt[("min",species)],costs.transfer_duplication + costs.transfer_cut + costs.loss)
            
            out23[(contents, species)] =  AddTupleInt(outAlt[("min",species)],2*costs.transfer_cut)
            
            Out[(contents, species)] = min(out11[(contents, species)],out12[(contents, species)]
                                           ,out22[(contents, species)],out23[(contents, species)])  
            
        
            #minInOut 
            minInOut[(contents, species)] = min(In[(contents, species)], Out[(contents, species)])
    
    return minInOut

def ME(contents1, contents2):
    if contents2.issubset(contents1):
        return "min"
    else:
        return "extra"

def ComputeCore(trees):
    leavesSet = set()
    leavesDict = {}
    core = []
    notCore = []
    maxLeavesNumber = 0
    maxLeavesTree = None
    for tree in trees:
        currentCounter = 0
        leavesDict[tree] = set()
        for leaf in leaves(tree):
            leavesSet.add(leaf.node.data.name)
            leavesDict[tree].add(leaf.node.data.name)
            currentCounter = currentCounter + 1
        if currentCounter >= maxLeavesNumber:
            maxLeavesNumber = currentCounter
            maxLeavesTree = tree
    core.append(maxLeavesTree)
    trees.remove(maxLeavesTree)
    leavesSet = leavesSet - leavesDict[maxLeavesTree]
    while(len(leavesSet) != 0):
        for tree in trees:
            if not leavesSet.isdisjoint(leavesDict[tree]):
                core.append(tree)
                leavesSet = leavesSet - leavesDict[tree]
                trees.remove(tree)
    
    for tree in trees:
        notCore.append(tree)
    
    return core, notCore

def ComputeMappingCoreNotCore(core, indexedNotCore):
    CoreNotCoreArray = []
    for treeNotCore in indexedNotCore:
        mapNode = {}    
        for treeCore in core:
            for nodeZipper in depth(treeCore):
                if nodeZipper.is_leaf():
                    try:
                        mapNode[nodeZipper.node] = treeNotCore(nodeZipper.node.data.name)
                    except:
                        mapNode[nodeZipper.node] = None
                else:
                    leftChild = mapNode[nodeZipper.down(0).node]
                    rightChild = mapNode[nodeZipper.down(1).node]
                    if (leftChild == None) and (rightChild == None):
                        mapNode[nodeZipper.node] = None
                    elif leftChild == None:
                        mapNode[nodeZipper.node] = rightChild
                    elif rightChild == None:
                        mapNode[nodeZipper.node] = leftChild
                    else: 
                        mapNode[nodeZipper.node] = treeNotCore(leftChild,rightChild)
        CoreNotCoreArray.append(mapNode)
    return CoreNotCoreArray

def IsPartitionCompatibleWithNotCore(mappingCoreNotCore,
                                     indexedNotCore,
                                     treesLeft : Set[Node], 
                                     treesRight : Set[Node]):
    i = 0
    for mapping in mappingCoreNotCore:
        leftSubtree = None
        rightSubtree = None
        for treeLeft in treesLeft:
            mapTreeLeft = mapping[treeLeft]
            if (leftSubtree == None) and (mapTreeLeft == None):
                pass
            elif leftSubtree == None:
                leftSubtree = mapTreeLeft
            elif mapTreeLeft == None:
                pass
            else:
                leftSubtree = indexedNotCore[i](leftSubtree,mapTreeLeft)
        for treeRight in treesRight:
            mapTreeRight = mapping[treeRight]
            if (rightSubtree == None) and (mapTreeRight == None):
                pass
            elif rightSubtree == None:
                rightSubtree = mapTreeRight
            elif mapTreeRight == None:
                pass
            else:
                rightSubtree = indexedNotCore[i](rightSubtree,mapTreeRight)
        if (rightSubtree != None) and (leftSubtree != None):
            if indexedNotCore[i].is_comparable(leftSubtree,rightSubtree):
                return False                                        
        i = i + 1      
    return True
