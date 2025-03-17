from sowing.repr.newick import *
from sowing.node import Node, Edge
from sowing.comb.supertree import Triple
from sowing.traversal import depth,leaves
from sowing import traversal
from sowing.util.partition import *
#from sowing.comb.binary import *
from itertools import product, combinations_with_replacement
from collections import Counter
from typing import Iterable
from dataclasses import dataclass, field
from model.history import *
from compute.util import *
from utils.algebras import make_single_selector, make_product, make_unit_cost
from sowing.node import Node, Edge
from numpy import inf
from typing import List, Mapping, Set
from Supertree import *
import time
from compute.superdtlx.recurrence import reconcile

@dataclass(frozen=True, repr=False)
class HistoryInputSyntenyTree:
    """Input to the super-reconciliation problem with gene trees in input."""

    # Synteny tree
    synteny_tree: Node[Associate, None]

    # Species tree
    species_tree: Node[Host, None]
        
    # Costs of evolutionary events
    costs: EventCosts 
               
@dataclass(frozen=True, repr=False)
class HistoryInputGeneTrees:
    """Input to the super-reconciliation problem with gene trees in input."""

    # Gene trees to create synteny tree from
    gene_trees: List[Node[Associate, None]]

    # Species tree
    species_tree: Node[Host, None]
        
    # Costs of evolutionary events
    costs: EventCosts         

def SynesthHistory(srec_input: HistoryInputSyntenyTree):
    setting = Reconciliation(srec_input.species_tree,srec_input.synteny_tree)
    algebra = make_single_selector(
        "single_solution_algebra",
        make_cost_algebra("cost", srec_input.costs),
        make_product("history_count_unit_gen", history_counter, history_unit_generator),
    )
    results = reconcile(setting, algebra)
    
    return results.key.value, results.value.value[1].value.value

def NaiveSynesth(srec_input: HistoryInputGeneTrees): 
    """
    Compute a minimum-cost unordered history from a set of 
    gene trees using the naive approach consisting in running Synesth on all possible 
    supersynteny trees. This algorithm can handle all kinds of events (including transfers) 
    and cost values.

    :param srec_input: objects of the history with consistent 
        gene trees in input
    :returns: The cost of an optimal history and a corresponding supersynteny tree
    """
    
    optimalCost = inf
    optimalSytenyTree = ""
    supertrees = all_binary_supertrees(srec_input.gene_trees)
    for tree in supertrees:
        inputSynesth = HistoryInputSyntenyTree(tree,srec_input.species_tree,srec_input.costs)
        currentCost = Synesth(inputSynesth)
        if currentCost < optimalCost:
            optimalCost = currentCost
            optimalSytenyTree = tree
    
    return optimalCost, optimalSytenyTree


def Synesth(srec_input: HistoryInputSyntenyTree): 
    """
    Compute a minimum-cost unordered history explaining a given synteny tree
    and species tree. This algorithm can handle all kinds of events 
    (including transfers) and cost values.

    :param srec_input: objects of the super-reconciliation with consistent 
        gene trees in input
    :param policy: whether to generate any minimal solution or all possible
        minimal solutions
    :returns: The cost of an optimal history
    """
    
    #Synteny tree in input
    syntenytree = srec_input.synteny_tree
    
    #Pre-process the species tree for fast answering of repeated lowest common ancestor queries
    #speciesTree = graft_unsampled_hostsV2(srec_input.species_tree)
    speciesTree = srec_input.species_tree
    indexedSpeciesTree = IndexedTree(speciesTree)

    #Counter of gene families occurences:
    geneFamilyNumberDict = {}
    geneNamesSet = set()
    for nodeZipper in leaves(syntenytree):
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
    
    #Possible contents
    contentsSet = {"min", "extra"}

    #Minimum cost of a transfer event
    costs = srec_input.costs
    costsTr = min(costs.transfer_duplication,costs.transfer_cut)
    
    #Find the optimal History cost
    c = {}
    minInOut = {}
    inAlt = {}
    outAlt = {}   
    geneFamilyCounterDict = {}
    for nodeZipper in depth(syntenytree):
        geneFamilyCounterDictCurrent = {}
        if nodeZipper.is_leaf():
            host = nodeZipper.node.data.host 
            syn = nodeZipper.node.data.contents
            for geneFamily in syn:
                geneFamilyCounterDictCurrent[geneFamily] = 1
                geneFamilyCounterDict[nodeZipper] = geneFamilyCounterDictCurrent
        else:
            left = nodeZipper.down(0)
            right = nodeZipper.down(1)
            #Compute minimal contents
            geneFamilyCounterDict[nodeZipper] = geneFamilyCounterDictCurrent
            for key in geneFamilyCounterDict[left]:
                geneFamilyCounterDict[nodeZipper][key] = geneFamilyCounterDict[left][key] 
            for key in geneFamilyCounterDict[right]:
                if key in geneFamilyCounterDict[nodeZipper]:
                    geneFamilyCounterDict[nodeZipper][key] = geneFamilyCounterDict[nodeZipper][key] + geneFamilyCounterDict[right][key]
                else:
                    geneFamilyCounterDict[nodeZipper][key] = geneFamilyCounterDict[right][key]
                    
            contentsLeft = set()
            for key in geneFamilyCounterDict[left]:
                contentsLeft.add(key)
                    
            contentsRight = set()
            for key in geneFamilyCounterDict[right]:
                contentsRight.add(key)
                    
            contentsP = set()
            for key in geneFamilyCounterDict[nodeZipper]:
                contentsP.add(key)
                    
            #Remove gene families gained below the current node
            for key in geneFamilyCounterDict[left]:
                if geneFamilyCounterDict[left][key] == geneFamilyNumberDict[key]:
                    contentsP.remove(key)
            for key in geneFamilyCounterDict[right]:
                if geneFamilyCounterDict[right][key] == geneFamilyNumberDict[key]:
                    contentsP.remove(key)       
        
        if nodeZipper.is_leaf():
            for species in speciesListPostOrder:
                #C
                if species == host:
                    c[(nodeZipper,"min",species)] = 0
                    c[(nodeZipper,"extra",species)] = inf
                else:
                    c[(nodeZipper,"min",species)] = inf
                    c[(nodeZipper,"extra",species)] = inf
                #inAlt
                if indexedSpeciesTree.is_ancestor_of(species,host) :
                    inAlt[(nodeZipper,"min",species)] = 0
                    inAlt[(nodeZipper,"extra",species)] = costs.loss
                else:
                    inAlt[(nodeZipper,"min",species)] = inf
                    inAlt[(nodeZipper,"extra",species)] = inf
        else:
            left = nodeZipper.down(0)
            right = nodeZipper.down(1)
            #C
            cSpe = inf
            cDup = inf
            cCut = inf
            cTrDup = inf
            cTrCut = inf
            
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
                        cSpe = inf
                    else:
                        leftChild = speciesZipper.down(0).node.data.name
                        rightChild = speciesZipper.down(1).node.data.name                
                        
                        cSpe = min(minInOut[(left,MEdict[("L","P",contents)],leftChild)] + 
                                                                    minInOut[(right,MEdict[("R","P",contents)],rightChild)],
                                                        minInOut[(right,MEdict[("R","P",contents)],leftChild)] +
                                                                    minInOut[(left,MEdict[("L","P",contents)],rightChild)]
                                                        )                    
                    #Duplication
                    cDup = min(minInOut[(left,"min",species)]+
                                                                    minInOut[(right,MEdict[("R","P",contents)],species)],
                                           minInOut[(left,MEdict[("L","P",contents)],species)]+
                                                                    minInOut[(right,"min",species)]
                    
                                            )+ costs.duplication
                    #Cut
                    if not contentsLeft.isdisjoint(contentsRight):
                        cCut = inf
                    else:
                        cCut = min(minInOut[(left,"min",species)]+
                                                    minInOut[(right,MEdict[("R","P-L",contents)],species)],
                                              minInOut[(left,MEdict[("L","P-R",contents)],species)]+
                                                    minInOut[(right,"min",species)]
                                            )+ costs.cut
                    #Transfer-Dup
                    if speciesZipper.is_root():
                        cTrDup = inf
                    else:
                        cTrDup = min(minInOut[(left,MEdict[("L","P",contents)],species)]+
                                                                           min(outAlt[(right,"min",species)],
                                                                               inAlt[(right,"min",species)]+costsTr)
                                                ,min(outAlt[(left,"min",species)],
                                                               inAlt[(left,"min",species)]+costsTr)           
                                                           +minInOut[(right,MEdict[("R","P",contents)],species)]) +costs.transfer_duplication

                    #Transfer-Cut
                    if speciesZipper.is_root() or (not contentsLeft.isdisjoint(contentsRight)):
                        cTrCut = inf
                    elif speciesZipper.up().is_root() and (not speciesZipper.node.data.sampled) :
                        Rll = speciesZipper.sibling().down(0).node.data.name
                        Rlr = speciesZipper.sibling().down(1).node.data.name
                        cTrCut =  min(minInOut[(left,MEdict[("L","P-R",contents)],species)]+
                                                           min(outAlt[(right,"min",species)],
                                                                               inAlt[(right,"min",species)]+costsTr)
                                                ,min(outAlt[(left,"min",species)],
                                                               inAlt[(left,"min",species)]+costsTr)          
                                                           +minInOut[(right,MEdict[("R","P-L",contents)],species)]
                                                ,minInOut[(left,"min",species)]+
                                                              min(outAlt[(right,MEdict[("R","P-L",contents)],species)],
                                                                  inAlt[(right,"min",species)]+costsTr,
                                                                  (min(inAlt[(right,"min",Rll)],
                                                                                  inAlt[(right,"min",Rlr)]                                                      
                                                                                    )+costsTr
                                                            ))
                                                ,minInOut[(right,"min",species)]+
                                                              min(outAlt[(left,MEdict[("L","P-R",contents)],species)],
                                                                  inAlt[(left,"min",species)]+costsTr,
                                                                  min(inAlt[(left,"min",Rll)],
                                                                                  inAlt[(left,"min",Rlr)]                                                      
                                                                                    )+costsTr
                                                            )
                                                 )+costs.transfer_cut
                    elif speciesZipper.up().is_root():
                        cTrCut =  min(minInOut[(left,MEdict[("L","P-R",contents)],species)]+
                                                           min(outAlt[(right,"min",species)],
                                                                               inAlt[(right,"min",species)]+costsTr)
                                                ,min(outAlt[(left,"min",species)],
                                                               inAlt[(left,"min",species)]+costsTr)           
                                                           +minInOut[(right,MEdict[("R","P-L",contents)],species)]
                                                ,minInOut[(left,"min",species)]+
                                                              min(outAlt[(right,MEdict[("R","P-L",contents)],species)],
                                                                  inAlt[(right,"min",species)]+costsTr
                                                            )
                                                ,minInOut[(right,"min",species)]+
                                                              min(outAlt[(left,MEdict[("L","P-R",contents)],species)],
                                                                  inAlt[(left,"min",species)]+costsTr
                                                            )
                                                 )+costs.transfer_cut
                    else:
                        cTrCut =  min(minInOut[(left,MEdict[("L","P-R",contents)],species)]+
                                                           min(outAlt[(right,"min",species)],
                                                                               inAlt[(right,"min",species)]+costsTr)
                                                ,min(outAlt[(left,"min",species)],
                                                               inAlt[(left,"min",species)]+costsTr)           
                                                           +minInOut[(right,MEdict[("R","P-L",contents)],species)]
                                                ,minInOut[(left,"min",species)]+
                                                              min(outAlt[(right,MEdict[("R","P-L",contents)],species)],
                                                                  inAlt[(right,"min",species)]+costsTr,
                                                                  outAlt[(right,"min",species)]+costsTr
                                                            )
                                                ,minInOut[(right,"min",species)]+
                                                              min(outAlt[(left,MEdict[("L","P-R",contents)],species)],
                                                                  inAlt[(left,"min",species)]+costsTr,
                                                                  outAlt[(left,"min",species)]+costsTr
                                                            )
                                                 )+costs.transfer_cut
                    #C
                    c[(nodeZipper,contents,species)] = min(cSpe,cDup,cCut,cTrDup,cTrCut)
                
            #inAlt
            for species in speciesListPostOrder:
                speciesZipper = indexedSpeciesTree(species)
                for contents in contentsSet:                                  
                    if speciesZipper.is_leaf():
                        inAlt[(nodeZipper,contents, species)] = min(c[(nodeZipper,contents, species)],c[(nodeZipper,"min", species)]+costs.loss)
                    else:
                        leftChild = speciesZipper.down(0).node.data.name
                        rightChild = speciesZipper.down(1).node.data.name    
                        inAlt[(nodeZipper,contents, species)] = min(inAlt[(nodeZipper,contents, leftChild)],
                                                         inAlt[(nodeZipper,contents, rightChild)],
                                                         c[(nodeZipper,contents, species)],
                                                         c[(nodeZipper,"min", species)]+costs.loss
                                                        )
        
        #OutAlt
        for species in speciesListPreOrder:
            speciesZipper = indexedSpeciesTree(species)
            for contents in contentsSet:
                if speciesZipper.is_root():
                    outAlt[(nodeZipper,contents, species)] = inf
                else:
                    parent  = speciesZipper.up().node.data.name
                    sibling = speciesZipper.sibling().node.data.name
                    outAlt[(nodeZipper,contents, species)] = min(outAlt[(nodeZipper,contents, parent)],inAlt[(nodeZipper,contents, sibling)])

        #minInOut
        minInOut.update(ComputeMinInOut(speciesListPostOrder,indexedSpeciesTree,nodeZipper,c,inAlt,outAlt,costs))

    currentMin = inf
    for nodeZipper in depth(syntenytree,preorder=True): 
        if nodeZipper.is_root():
            for species in speciesListPostOrder:
                if c[(nodeZipper,"min",species)] < currentMin:
                    currentMin = c[(nodeZipper,"min",species)]
        else:
            break
    
    return currentMin

def ComputeMinInOut(speciesListPostOrder,indexedSpeciesTree,nodeZipper,c,inAlt,outAlt,costs):
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
                
                in01[(contents, species)] = min(c[(nodeZipper,contents, species)],c[(nodeZipper,"min", species)] + costs.loss)
                
                if not(isSampled):
                    minCostDupCut = min(costs.duplication, costs.cut)
                    in02[(contents, species)] = min(c[(nodeZipper,contents, species)],c[(nodeZipper,"min", species)]+ minCostDupCut)
                else:
                    in02[(contents, species)] = inf

                in11[(contents, species)] = inf
                in12[(contents, species)] = inf 
                
                in21[(contents, species)] = c[(nodeZipper,"min", species)]+2*costs.transfer_duplication + isSampled*costs.loss
                in22[(contents, species)] = c[(nodeZipper,"min", species)]+costs.transfer_duplication + costs.transfer_cut
                in23[(contents, species)] = c[(nodeZipper,"min", species)]+2*costs.transfer_cut
            else:
                leftChild = speciesZipper.down(0).node.data.name
                rightChild = speciesZipper.down(1).node.data.name
                isSampledLeft = speciesZipper.down(0).node.data.sampled
                isSampledRight = speciesZipper.down(1).node.data.sampled

                if speciesZipper.is_root():
                    in21[(contents, species)] = inf
                    
                    in22[(contents, species)] = min(inAlt[(nodeZipper,"min", leftChild)],
                                                            c[(nodeZipper,"min",rightChild)] +costs.loss
                                                               )+costs.transfer_duplication + costs.transfer_cut

                    in23[(contents, species)] = min(inAlt[(nodeZipper,"min", leftChild)],
                                                            c[(nodeZipper,"min",rightChild)] +costs.loss
                                                               )+2*costs.transfer_cut
                else:
                    in21[(contents, species)] = inAlt[(nodeZipper,"min", species)]+2*costs.transfer_duplication + costs.loss
                    
                    in22[(contents, species)] = inAlt[(nodeZipper,"min", species)]+costs.transfer_duplication + costs.transfer_cut
                    
                    in23[(contents, species)] = inAlt[(nodeZipper,"min", species)]+2*costs.transfer_cut                    
                
                in01[(contents, species)] = min(in01[(contents, leftChild)]+ isSampledRight*costs.loss,
                                                in01[(contents, rightChild)]+isSampledLeft *costs.loss,
                                                c[(nodeZipper,contents, species)],
                                                c[(nodeZipper,"min", species)]+ costs.loss)
                
                in02[(contents, species)] = min(in02[(contents, leftChild)]+ isSampledRight*costs.loss,
                                                in02[(contents, rightChild)]+ isSampledLeft *costs.loss)
                
                in11[(contents, species)] = min(inAlt[(nodeZipper,"min",leftChild)]+isSampledRight*costs.loss + isSampledLeft *costs.loss,
                                                inAlt[(nodeZipper,"min",rightChild)]+isSampledRight*costs.loss + isSampledLeft *costs.loss 
                                                )+costs.transfer_duplication
                
                in12[(contents, species)] = min(inAlt[(nodeZipper,"min",leftChild)]+isSampledRight*costs.loss + isSampledLeft *costs.loss,
                                                inAlt[(nodeZipper,"min",rightChild)]+isSampledRight*costs.loss + isSampledLeft *costs.loss,
                                                inAlt[(nodeZipper,contents,leftChild)]+isSampledLeft *costs.loss,
                                                inAlt[(nodeZipper,contents,rightChild)]+isSampledRight*costs.loss
                                                )+costs.transfer_cut

            In[(contents, species)] = min(in01[(contents, species)],in02[(contents, species)],#in03[(contents, species)],
                                          in11[(contents, species)],in12[(contents, species)],in21[(contents, species)]
                                          ,in22[(contents, species)],in23[(contents, species)])
            
            #Out
            out11[(contents, species)] = outAlt[(nodeZipper,"min",species)]+costs.transfer_duplication + isSampled*costs.loss

            out12[(contents, species)] = min(outAlt[(nodeZipper,"min",species)]+ isSampled*costs.loss,
                                             outAlt[(nodeZipper,contents, species)]
                                             )+costs.transfer_cut
            
            out22[(contents, species)] = outAlt[(nodeZipper,"min",species)]+costs.transfer_duplication + costs.transfer_cut + costs.loss
            
            out23[(contents, species)] =  outAlt[(nodeZipper,"min",species)]+2*costs.transfer_cut
            
            Out[(contents, species)] = min(out11[(contents, species)],out12[(contents, species)]
                                           ,out22[(contents, species)],out23[(contents, species)])  
            
        
            #minInOut 
            minInOut[(nodeZipper,contents, species)] = min(In[(contents, species)], Out[(contents, species)])
    
    return minInOut

def ME(contents1, contents2):
    if contents2.issubset(contents1):
        return "min"
    else:
        return "extra"
