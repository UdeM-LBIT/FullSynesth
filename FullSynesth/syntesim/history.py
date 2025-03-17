import json
from dataclasses import dataclass, field
from collections.abc import Generator, Iterable
from typing import Optional
from .simulator import State, Event
from .graph import Graph
from sowing.node import Node, Edge
from sowing.traversal import depth
from model.history import Extant, Codiverge, Diverge, Gain, Loss

@dataclass
class SpeciesGraph(Graph):
    speciations: set = field(default_factory=set)
    losses: set = field(default_factory=set)


@dataclass
class GeneGraph(SpeciesGraph):
    duplications: set = field(default_factory=set)
    transfers: set = field(default_factory=set)


@dataclass
class SyntenyGraph(GeneGraph):
    cuts: set = field(default_factory=set)
    joins: set = field(default_factory=set)


Record = tuple[State, State, Event, dict]


def read_log(records: Iterable[str]) -> Generator[Record, None, None]:
    last_state: Optional[State] = None
    last_event: Optional[Event] = None
    last_data: Optional[dict] = None

    for line in records:
        data: dict = json.loads(line)

        if "event" in data:
            last_event = getattr(Event, data["event"])
            last_data = data
        elif "state" in data:
            state = State.unserialize(data["state"])

            if (
                last_state is not None
                and last_event is not None
                and last_data is not None
            ):
                yield last_state, state, last_event, last_data

            last_state = state


def build_history(records: Iterable[str]) \
        -> tuple[SpeciesGraph, SyntenyGraph, GeneGraph]:
    spe_forest = SpeciesGraph()
    synt_forest = SyntenyGraph()
    gene_forest = GeneGraph()
    object_species_hist = {}
    syntenies_hist = {}
    all_event = {}
    dup_trans = {}

    for last_state, state, event, data in read_log(records):
        match event:
            case Event.Speciation: # done
                parent_spe = data["parent_id"]
                child1_spe = data["child1_id"]
                child2_spe = data["child2_id"]
                for synteny, synt_content in last_state[parent_spe].syntenies.items():
                    gene_tuple = []
                    for gene_name in synt_content:
                        gene_tuple.append(gene_name.id)
                    gene_tuple = frozenset(gene_tuple)

                    new_event = Codiverge(name=synteny, host= parent_spe, contents=gene_tuple)           
                    all_event[synteny] = new_event


                # Record to species forest
                spe_forest.nodes.add(parent_spe)
                spe_forest.nodes.add(child1_spe)
                spe_forest.nodes.add(child2_spe)
                spe_forest.edges.add((parent_spe, child1_spe))
                spe_forest.edges.add((parent_spe, child2_spe))
                spe_forest.speciations.add(parent_spe)

                # Record to synteny forest
                parent_synts = sorted(last_state[parent_spe].syntenies.keys())
                child1_synts = sorted(state[child1_spe].syntenies.keys())
                child2_synts = sorted(state[child2_spe].syntenies.keys())
                
                for parent_synt, child1_synt, child2_synt in zip(
                    parent_synts, child1_synts, child2_synts
                ):
                    synt_forest.nodes.add(parent_synt)
                    synt_forest.nodes.add(child1_synt)
                    synt_forest.nodes.add(child2_synt)
                    synt_forest.edges.add((parent_synt, child1_synt))
                    synt_forest.edges.add((parent_synt, child2_synt))
                    synt_forest.speciations.add(parent_synt)
                  

            
                    # Record to gene forest
                    for parent_gene, child1_gene, child2_gene in zip(
                        last_state[parent_spe].syntenies[parent_synt],
                        state[child1_spe][child1_synt],
                        state[child2_spe][child2_synt],
                    ):
                        gene_forest.nodes.add(parent_gene.id)
                        gene_forest.nodes.add(child1_gene.id)
                        gene_forest.nodes.add(child2_gene.id)
                        gene_forest.edges.add((parent_gene.id, child1_gene.id))
                        gene_forest.edges.add((parent_gene.id, child2_gene.id))
                        gene_forest.speciations.add(parent_gene.id)
                        

            case Event.Extinction:
                parent_spe = data["parent_id"]

                # Record to species forest
                spe_forest.nodes.add(parent_spe)
                spe_forest.losses.add(parent_spe)

                # Record to synteny forest
                for parent_synt in last_state[parent_spe].syntenies:

                    synt_forest.nodes.add(parent_synt)
                    synt_forest.losses.add(parent_synt)

                    # Record to gene forest
                    for parent_gene in last_state[parent_spe].syntenies[parent_synt]:
                        gene_forest.nodes.add(parent_gene.id)
                        gene_forest.losses.add(parent_gene.id)

            case Event.Duplication|Event.Transfer: # How to get result ?
                outgoing_spe = data["outgoing_id"] #nouvelle espece
                incoming_spe = data["incoming_id"] #espece existante

                start = int(data["start"])
                end = int(data["end"])

                parent = data["synteny_id"]
                original = data["original_id"] #syntenie tranfere
                copy = data["copy_id"] #syntenie de l'espece existante
                
                # Record to synteny forest
                synt_forest.nodes.add(parent)
                synt_forest.nodes.add(original)
                synt_forest.nodes.add(copy)
                synt_forest.edges.add((parent, original))
                synt_forest.edges.add((parent, copy))
                
                dup_trans[parent] = [original,copy]
                synt_content = last_state[outgoing_spe].syntenies[parent]
                gene_tuple = []
                for gene_name in synt_content:
                    gene_tuple.append(gene_name.id)
                gene_tuple = frozenset(gene_tuple)

                if event == Event.Duplication:
                    synt_forest.duplications.add(parent)

                    new_event = Diverge(
                            name= parent,
                            host= outgoing_spe,
                            contents= gene_tuple,
                            result=1, #------------------------
                            segment=(start, end),
                            cut=False,)
                else:
                    synt_forest.transfers.add(parent)
                    new_event = Diverge(
                        name=parent,
                        host=outgoing_spe,
                        contents=gene_tuple,
                        result=1, #-----------------------------
                        segment=(start, end),
                        cut=False,
                        transfer=True,
                    )

                
                # Record to gene forest
                for (parent_gene, original_gene, copy_gene) in zip(
                    last_state[outgoing_spe].syntenies[parent][start:end],
                    state[outgoing_spe][original][start:end],
                    state[incoming_spe][copy],
                ):
                    gene_forest.nodes.add(parent_gene.id)
                    gene_forest.nodes.add(original_gene.id)
                    gene_forest.nodes.add(copy_gene.id)
                    gene_forest.edges.add((parent_gene.id, original_gene.id))
                    gene_forest.edges.add((parent_gene.id, copy_gene.id))
                    
                    if event == Event.Duplication:
                        gene_forest.duplications.add(parent_gene.id)
                    else:
                        gene_forest.transfers.add(parent_gene.id)

            case Event.Gain:#done
                spe = data["species_id"]
                synt = data["synteny_id"]

                gene_tuple = []
                for synt_gene in last_state[spe].syntenies[synt]:
                    gene_tuple.append(synt_gene.id)
                gene_tuple = frozenset(gene_tuple)

                new_event = Gain(name=synt, 
                            host=spe, 
                            contents=gene_tuple, 
                            gained=frozenset([data["gene"]])
                        )
                
                all_event[synt] = new_event
                gene_forest.nodes.add(data["gene"])
                
            case Event.Loss: #done
               
                species = data["species_id"]
                parent = data["synteny_id"]

                start = int(data["start"])
                end = int(data["end"])

                synteny = last_state[species].syntenies[parent]

                gene_tuple = []
                for synt_gene in synteny:
                    gene_tuple.append(synt_gene.id)
                gene_tuple = frozenset(gene_tuple)

                new_event = Loss(name=parent, 
                            host=species,
                            contents=gene_tuple,
                            segment=(start, end))
                
                all_event[parent] = new_event

                # Record to synteny forest
                if start == 0 and end == len(synteny):
                    synt_forest.losses.add(parent)

                # Record to gene forest
                for lost_gene in synteny[start:end]:
                    gene_forest.nodes.add(lost_gene.id)
                    gene_forest.losses.add(lost_gene.id)

            case Event.Cut: # done
                species = data["species_id"]
                parent = data["synteny_id"]
                child1 = data["child1_id"]
                child2 = data["child2_id"]

                synteny = last_state[species].syntenies[parent]
                gene_tuple = []
                for synt_gene in synteny:
                    gene_tuple.append(synt_gene.id)
                seg_cut = gene_tuple[int(data["position"]):]
                gene_tuple = frozenset(gene_tuple)


                new_event = Diverge(
                    name=parent,
                    host=species,
                    contents=gene_tuple,
                    result=1,
                    segment=frozenset(seg_cut),
                    cut=True,
                )
                all_event[parent] = new_event 
                # Record to synteny forest
                synt_forest.nodes.add(parent)
                synt_forest.nodes.add(child1)
                synt_forest.nodes.add(child2)
                synt_forest.edges.add((parent, child1))
                synt_forest.edges.add((parent, child2))
                synt_forest.cuts.add(parent)

            case Event.Join:
                parent1 = data["synteny1_id"]
                parent2 = data["synteny2_id"]
                child = data["child_id"]

                # Record to synteny forest
                synt_forest.nodes.add(parent1)
                synt_forest.nodes.add(parent2)
                synt_forest.nodes.add(child)
                synt_forest.edges.add((parent1, child))
                synt_forest.edges.add((parent2, child))
                synt_forest.joins.add(parent1)
                synt_forest.joins.add(parent2)
              
        for spe_id, spe in state.species.items():
            for synt_id, genes in spe.syntenies.items():
                object_species_hist[synt_id] = spe_id
                
                genes_tab = []
                for gene in genes:
                    genes_tab.append(gene.id)
                
                new_event = Extant(name=synt_id, host=spe_id, contents=frozenset(genes_tab))
                all_event[synt_id] = new_event

                syntenies_hist[synt_id] = genes_tab
  
    
    return spe_forest, synt_forest, gene_forest, state, object_species_hist, syntenies_hist
