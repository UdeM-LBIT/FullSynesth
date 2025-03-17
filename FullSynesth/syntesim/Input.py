from model.history import (
    Host,
    Associate,
    parse_tree,
    Reconciliation,
    graft_unsampled_hostsV2
)
from sowing.traversal import depth,leaves

def RemoveBackslashN(TreesArray):
    i = 0
    for tree in TreesArray:
        TreesArray[i] = tree.replace("\n", "")
        i = i + 1

def RemoveInternalNodesTree(newickTree):
    newTree = ""
    skip = False
    for caracter in newickTree:

        if caracter == ")":
            newTree = newTree + caracter
            skip = True


        if (caracter == ",") or (caracter == ";"):
            skip = False

        if skip == False:
            newTree = newTree + caracter
    return newTree

def RemoveInternalNodesTreesArray(TreesArray):    
    i = 0
    for tree in TreesArray:
        TreesArray[i] = RemoveInternalNodesTree(tree)
        i = i + 1

def AddInternalNodesTreesArray(TreesArray):    
    i = 0
    for tree in TreesArray:
        TreesArray[i] = AddInternalNodesTree(tree)
        i = i + 1

def AddInternalNodesTree(newickTree):
    newTree = ""
    skip = False
    i = 0
    for caracter in newickTree:
        newTree = newTree + caracter
        if caracter == ")":
            newTree = newTree + "I" + str(i)
            i = i + 1
    return newTree

def SyntenyMapping(gene,SimulationArray,geneTreesSetArray):
    gene = "\"" + gene + "\""   
    simulationLine = SimulationArray[len(SimulationArray)-1]
    index = simulationLine.find(gene) 
    if index != -1:
        return SyntenyMapping2(simulationLine,index,geneTreesSetArray)
    else:
        return "Extinct"

def SyntenyContents(synteny,simulationLine,geneTreesSetArray):
    index1 = simulationLine.find("\"" + synteny + "\"")
    index2 = simulationLine.find("\"X",index1+1)
    if index2 == -1:
        index2 = len(simulationLine)
    genesArray = list()
    index3 = index1
    while(True):
        index3 = simulationLine.find("G",index3+1)
        if (index3 > index2) or (index3 == -1) :
            break
        currentGene = ""
        for i in range(index3,len(simulationLine)):
            if simulationLine[i] == "\"":
                break
            else:
                currentGene = currentGene + simulationLine[i]
        genesArray.append(currentGene)
    geneFamilyArray = list()
    for gene in genesArray:
        for i in range(0,len(geneTreesSetArray)):
            if gene in geneTreesSetArray[i]:
                geneFamilyArray.append("F" + str(i))
    
    geneFamilyString = ""
    geneFamilyString = geneFamilyString + "'{"
    for family in geneFamilyArray:
        geneFamilyString = geneFamilyString + "''" + family +"''"+ ","
    
    geneFamilyString = geneFamilyString + "}'"

    return geneFamilyString  

def SyntenyMapping2(SimulationLine,index,geneTreesSetArray):
    species = ""
    synteny = ""
    contents = ""
    obj = ""
    syntenyBool = False
    speciesBool = False
    for c in SimulationLine[index:0:-1]:
        obj = obj + c
        if c == "\"":
            obj = ""
        if (c == "X") and (syntenyBool == False):
            synteny = obj[::-1]
            syntenyBool = True
        if c == "S" and (speciesBool == False):
            species = obj[::-1]
            speciesBool = True
    contents = SyntenyContents(synteny,SimulationLine,geneTreesSetArray)
    return synteny + "[&host="+ species +",contents="+ contents+ "]"

def GeneToSynteny(SimulationArray,geneTrees):
    geneTreesSetArray = []
    for treeNewick in geneTrees:
        tree = parse_tree(Associate, treeNewick)
        leavesSet = set()
        for nodeZipper in leaves(tree):
            leavesSet.add(nodeZipper.node.data.name)
        geneTreesSetArray.append(leavesSet)

    treeArray = []
    for genetree in geneTrees:
        genetreeCurrent = ""
        gene = ""
        geneBool = False
        for c in genetree:
            if c == "G":
                geneBool = True
            elif (not c.isnumeric()) and (gene != ""):
                genetreeCurrent = genetreeCurrent + SyntenyMapping(gene,SimulationArray,geneTreesSetArray)
                gene = ""
                geneBool = False         
            if geneBool:
                gene = gene + c
            else:
                genetreeCurrent = genetreeCurrent + c
        
        treeArray.append(genetreeCurrent)    
    
    return treeArray

def removeExtinctGeneTree(geneTree):
    index = geneTree.find("Extinct")
    if index == -1:
        return geneTree
    if geneTree == "Extinct;":
        return "" 
    newGeneTree1 = geneTree[0:index-1]
    newGeneTree2 = ""
    if geneTree[index-1] == ",":
        skip = True
        counter = 1
        for c in reversed(newGeneTree1):
            if (c == "("):
                counter = counter - 1
                if skip and (counter == 0):
                    skip = False
                else:
                    newGeneTree2 = newGeneTree2 + c
            else:
                if (c == ")"):
                    counter = counter + 1
                newGeneTree2 = newGeneTree2 + c
        newGeneTree2 = newGeneTree2[::-1]        
    else:
        newGeneTree2 = newGeneTree1        
    counter = 1
    skip = True
    index2 = -1
    for i in range(index,len(geneTree)):
        if geneTree[i] == ")":
            counter = counter - 1
            if counter == 0:
                index2 = i
                break
        if geneTree[i] == "(":
            counter = counter + 1
        if skip == False:
            newGeneTree2 = newGeneTree2 + geneTree[i]
        if geneTree[i] == ",":
            skip = False
    newGeneTree2 = newGeneTree2 + geneTree[index2+1:]
    return newGeneTree2

def removeExtinctGeneTreeArray(geneTreeArray):
    treeArray = []
    for genetree in geneTreeArray:
        genetreeCurrent1 = removeExtinctGeneTree(genetree)
        genetreeCurrent2 = removeExtinctGeneTree(genetreeCurrent1)
        while(genetreeCurrent1 != genetreeCurrent2):
            genetreeCurrent1 = genetreeCurrent2
            genetreeCurrent2 = removeExtinctGeneTree(genetreeCurrent2)
            
        if genetreeCurrent2 != "":
            treeArray.append(genetreeCurrent2) 
        
    return treeArray

def dataSet(SimulationArray):
    #Species tree
    SpeciesFile =  open("Species.json", "r")
    SpeciesLine = SpeciesFile.readlines()
    RemoveBackslashN(SpeciesLine)
    SpeciesTreeNewick = SpeciesLine[0]
    speciesTree = parse_tree(Host, SpeciesTreeNewick)
        
    #Genes tree
    geneTreesFile = open("Trees.json", "r")
    TreesArray = geneTreesFile.readlines()
    RemoveBackslashN(TreesArray)
    RemoveInternalNodesTreesArray(TreesArray)

    TreesArraySynteny = removeExtinctGeneTreeArray(GeneToSynteny(SimulationArray,TreesArray))
        
    AddInternalNodesTreesArray(TreesArraySynteny)

    trees = []
    for tree in TreesArraySynteny:
        trees.append(parse_tree(Associate, tree))

    return trees, speciesTree
