from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from numpy import array
from numpy.random import default_rng, Generator
from typing import Any, Optional, TypeVar, overload
import json
from io import IOBase, StringIO
from .model import (
    Id, GeneId, SyntenyId, SpeciesId, Orientation, Gene, Synteny, Species,
    SpeciesId, SerializedSpecies,
)


T = TypeVar("T")


class InvalidEventError(RuntimeError):
    pass


class NullFile(IOBase):
    def write(self, data: str) -> None: pass
    def writable(self) -> bool: return True


class Event(Enum):
    Speciation = auto()
    Extinction = auto()
    Duplication = auto()
    Transfer = auto()
    Gain = auto()
    Loss = auto()
    Cut = auto()
    Join = auto()


SerializedState = dict[SpeciesId, SerializedSpecies]


@dataclass
class State:
    generator: Generator = field(default_factory=lambda: default_rng(42))
    species: dict[SpeciesId, Species] = field(default_factory=dict)
    logfile: IOBase = NullFile()

    @classmethod
    def of(cls, *species: Species) -> State:
        return State(species={
            item.id: item for item in species
        })

    @classmethod
    def unit(cls) -> State:
        gene = Gene()
        synteny = Synteny.of(gene)
        species = Species.of(synteny)
        return State.of(species)

    def __getitem__(self, key: SpeciesId) -> Species:
        """Get a species from the current state."""
        return self.species[key]

    def add(self, species: Species) -> None:
        """Add a species to the current state."""
        self.species[species.id] = species

    def remove(self, species: Species|SpeciesId) -> None:
        """Remove a species from the current state."""
        if isinstance(species, Species):
            del self.species[species.id]
        else:
            del self.species[species]

    @overload
    def _select(self, container: dict[Id, T], key: Optional[Id] = ...) -> T: ...

    @overload
    def _select(self, container: list[T], key: Optional[int] = ...) -> T: ...

    def _select(
        self,
        container: dict[Id, T]|list[T],
        key: Optional[Id|int] = None
    ) -> T:
        if key is None:
            if isinstance(container, dict):
                id_key = self.generator.choice(list(container.keys()))
                return container[id_key]
            else:
                int_key = self.generator.integers(len(container))
                return container[int_key]
        else:
            if isinstance(container, dict):
                if not isinstance(key, Id):
                    raise TypeError("must select in dict with id")

                return container[key]
            else:
                if not isinstance(key, int):
                    raise TypeError("must select in list with int")

                return container[key]

    def log_to(self, file: Optional[IOBase] = None) -> None:
        """Start or stop logging events to the given file."""
        self.logfile = NullFile() if file is None else file
        self._log_state()

    def _log_state(self) -> None:
        data = {"state": self.serialize()}
        print(json.dumps(data), file=self.logfile)

    def _log_event(self, event: Event, **args: Any) -> None:
        data = {"event": event.name}
        data.update({
            key: str(value)
            for key, value in args.items()
        })
        print(json.dumps(data), file=self.logfile)
        self._log_state()

    @overload
    def event(self, kind: Optional[dict[Event, float]] = ...) -> None: ...

    @overload
    def event(self, kind: Event, *args: Any) -> None: ...

    def event(
        self,
        kind: Optional[dict[Event, float]]|Event = None,
        *args: Any,
    ) -> None:
        if isinstance(kind, Event):
            match kind:
                case Event.Speciation: self.speciation(*args)
                case Event.Extinction: self.extinction(*args)
                case Event.Duplication: self.duplication(*args)
                case Event.Transfer: self.transfer(*args)
                case Event.Gain: self.gain(*args)
                case Event.Loss: self.loss(*args)
                case Event.Cut: self.cut(*args)
                case Event.Join: self.join(*args)
        else:
            if kind is None:
                events = list(Event)
                probs = [1.] * len(events)
            else:
                events = list(kind.keys())
                probs = list(kind.values())

            while events:
                norm_probs = array(probs, dtype=float)
                if(sum(norm_probs) == 0):
                    raise InvalidEventError("no event applicable")
                norm_probs /= sum(norm_probs)
                index = self.generator.choice(len(events), p=norm_probs)
                event = events[index]

                try:
                    self.event(event)
                except InvalidEventError:
                    events = events[:index] + events[index + 1:]
                    probs = probs[:index] + probs[index + 1:]
                else:
                    return

            raise InvalidEventError("no event applicable")

    def speciation(
        self,
        parent_id: Optional[SpeciesId] = None,
    ) -> None:
        """Replace a parent species with two independent copies."""
        if not self.species:
            raise InvalidEventError("needs at least one species")

        parent = self._select(self.species, parent_id)
        child1 = parent.deep_clone()
        child2 = parent.deep_clone()

        self.remove(parent)
        self.add(child1)
        self.add(child2)

        self._log_event(
            Event.Speciation,
            parent_id=parent.id,
            child1_id=child1.id,
            child2_id=child2.id,
        )

    def extinction(
        self,
        parent_id: Optional[SpeciesId] = None,
    ) -> None:
        """Remove a species."""
        if not self.species:
            raise InvalidEventError("needs at least one species")

        parent = self._select(self.species, parent_id)
        self.remove(parent.id)

        self._log_event(
            Event.Extinction,
            parent_id=parent.id,
        )

    def duplication(
        self,
        species_id: Optional[SpeciesId] = None,
        synteny_id: Optional[SyntenyId] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Duplicate a segment of a synteny to a new synteny inside
        the same species.
        """
        if not self.species:
            raise InvalidEventError("needs at least one species")

        species = self._select(self.species, species_id)
        self._copy_segment(
            Event.Duplication, species, species, synteny_id, start, end,
        )

    def transfer(
        self,
        outgoing_id: Optional[SpeciesId] = None,
        incoming_id: Optional[SpeciesId] = None,
        synteny_id: Optional[SyntenyId] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Transfer a segment of a synteny to a new synteny inside
        another co-existing species.
        """
        if len(self.species) < 2:
            raise InvalidEventError("needs at least two distinct species")

        outgoing = self._select(self.species, outgoing_id)
        incoming = self._select(self.species, incoming_id)

        if outgoing_id is None:
            while incoming == outgoing:
                outgoing = self._select(self.species)
        elif incoming_id is None:
            while incoming == outgoing:
                incoming = self._select(self.species)
        else:
            if incoming == outgoing:
                raise InvalidEventError("cannot transfer to the same species")

        self._copy_segment(
            Event.Transfer, outgoing, incoming, synteny_id, start, end,
        )

    def _copy_segment(
        self,
        kind: Event,
        outgoing: Species,
        incoming: Species,
        synteny_id: Optional[SyntenyId] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        synteny = self._select(outgoing.syntenies, synteny_id)
        genes = len(synteny.genes)

        if start is None:
            start = self.generator.integers(genes)

        if end is None:
            size = max(1, min(genes - start, self.generator.geometric(0.1)))
            end = start + size

        if start >= end:
            raise InvalidEventError("cannot copy empty segment")

        if start < 0 or end > genes:
            raise InvalidEventError("cannot copy invalid segment")

        # Make fresh IDs for cloned genes, keep original IDs for others
        original = synteny[start:end].deep_clone()

        if start > 0:
            original[0:0] = synteny[:start]

        if end < len(synteny):
            original[len(synteny):0] = synteny[end:]

        outgoing.remove(synteny)
        outgoing.add(original)

        copy = synteny[start:end].deep_clone()
        incoming.add(copy)

        self._log_event(
            kind,
            outgoing_id=outgoing.id,
            incoming_id=incoming.id,
            synteny_id=synteny.id,
            start=start,
            end=end,
            original_id=original.id,
            copy_id=copy.id,
        )

    def gain(
        self,
        species_id: Optional[SpeciesId] = None,
        synteny_id: Optional[SyntenyId] = None,
        position: Optional[int] = None,
        orientation: Optional[Orientation] = None
    ) -> None:
        """Add a new gene inside an existing synteny."""
        if not self.species:
            raise InvalidEventError("needs at least one species")

        species = self._select(self.species, species_id)
        synteny = self._select(species.syntenies, synteny_id)
        genes = len(synteny.genes)

        if position is None:
            position = self.generator.integers(genes + 1)

        if orientation is None:
            orientation = bool(self.generator.integers(2))

        gene = Gene()
        gene.orientation = orientation
        synteny.genes.insert(position, gene)

        self._log_event(
            Event.Gain,
            species_id=species.id,
            synteny_id=synteny.id,
            position=position,
            gene=gene.id,
            orient=orientation,
        )

    def loss(
        self,
        species_id: Optional[SpeciesId] = None,
        synteny_id: Optional[SyntenyId] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """Loose a synteny segment."""
        if not self.species:
            raise InvalidEventError("needs at least one species")

        species = self._select(self.species, species_id)
        synteny = self._select(species.syntenies, synteny_id)
        genes = len(synteny.genes)

        if start is None:
            start = self.generator.integers(genes)

        if end is None:
            size = max(1, min(genes - start, self.generator.geometric(0.5)))
            end = start + size

        del synteny[start:end]

        if not len(synteny):
            species.remove(synteny)

        if not len(species):
            self.remove(species)

        self._log_event(
            Event.Loss,
            species_id=species.id,
            synteny_id=synteny.id,
            start=start,
            end=end,
        )

    def cut(
        self,
        species_id: Optional[SpeciesId] = None,
        synteny_id: Optional[SyntenyId] = None,
        position: Optional[int] = None,
    ) -> None:
        """Cut a synteny into two segments."""
        if all(
            len(synteny) <= 1
            for species in self.species.values()
            for synteny in species.syntenies.values()
        ):
            raise InvalidEventError(
                "needs at least one species with "
                "a synteny of length at least 2"
            )

        species = self._select(self.species, species_id)

        if species_id is None:
            while all(
                len(synteny) <= 1
                for synteny in species.syntenies.values()
            ):
                species = self._select(self.species)
        else:
            if all(
                len(synteny) <= 1
                for synteny in species.syntenies.values()
            ):
                raise InvalidEventError(
                    "needs a species with a synteny of "
                    "length at least 2"
                )

        synteny = self._select(species.syntenies, synteny_id)

        if synteny_id is None:
            while len(synteny) <= 1:
                synteny = self._select(species.syntenies)
        else:
            if len(synteny) <= 1:
                raise InvalidEventError("needs a synteny of length at least 2")

        genes = len(synteny.genes)

        if position is None:
            position = self.generator.integers(1, genes)

        if position <= 0 or position >= genes:
            raise InvalidEventError("cannot cut at start or end of synteny")

        synteny1 = synteny[:position].clone()
        synteny2 = synteny[position:].clone()

        species.remove(synteny)
        species.add(synteny1)
        species.add(synteny2)

        self._log_event(
            Event.Cut,
            species_id=species.id,
            synteny_id=synteny.id,
            position=position,
            child1_id=synteny1.id,
            child2_id=synteny2.id,
        )

    def join(
        self,
        species_id: Optional[SpeciesId] = None,
        synteny1_id: Optional[SyntenyId] = None,
        orient1: Orientation = None,
        synteny2_id: Optional[SyntenyId] = None,
        orient2: Orientation = None,
    ) -> None:
        """Replace a pair of syntenies in a species with their concatenation."""
        if not self.species:
            raise InvalidEventError("needs at least one species")

        species = self._select(self.species, species_id)

        if len(species.syntenies) < 2:
            raise InvalidEventError("needs at least two distinct syntenies")

        synteny1 = self._select(species.syntenies, synteny1_id)
        synteny2 = self._select(species.syntenies, synteny2_id)

        if synteny1_id is None:
            while synteny1 == synteny2:
                synteny1 = self._select(species.syntenies)
        elif synteny2_id is None:
            while synteny1 == synteny2:
                synteny2 = self._select(species.syntenies)
        else:
            if synteny1 == synteny2:
                raise InvalidEventError("cannot join synteny with itself")

        if orient1 is None:
            orient1 = bool(self.generator.integers(2))

        if not orient1:
            synteny1 = -synteny1

        if orient2 is None:
            orient2 = bool(self.generator.integers(2))

        if not orient2:
            synteny2 = -synteny2

        synteny3 = Synteny.of(*synteny1.genes, *synteny2.genes)
        species.remove(synteny1)
        species.remove(synteny2)
        species.add(synteny3)

        self._log_event(
            Event.Join,
            species_id=species.id,
            synteny1_id=synteny1.id,
            orient1=orient1,
            synteny2_id=synteny2.id,
            orient2=orient2,
            child_id=synteny3.id,
        )

    def __len__(self) -> int:
        return len(self.species)

    def __str__(self) -> str:
        species = [str(item) for item in self.species.values()]
        return f"{', '.join(species)}"

    def serialize(self) -> SerializedState:
        return {
            key: species.serialize()
            for key, species in self.species.items()
        }

    @classmethod
    def unserialize(self, data: SerializedState) -> State:
        return State.of(*(
            replace(
                Species.unserialize(species),
                id=key
            )
            for key, species in data.items()
        ))
