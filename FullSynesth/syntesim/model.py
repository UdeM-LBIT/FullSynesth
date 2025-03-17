from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, overload


Id = str
GeneId = Id
SyntenyId = Id
SpeciesId = Id
Orientation = bool


class IdGenerator:
    """Generate sequential IDs with a common prefix."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.seq = 0

    def __call__(self) -> str:
        self.seq += 1
        return f"{self.prefix}{self.seq}"

    def reset(self) -> None:
        self.seq = 0


SerializedGene = tuple[GeneId, bool]

@dataclass
class Gene:
    """A single gene which can be oriented."""
    id: GeneId = field(default_factory=IdGenerator("G"))
    orientation: Orientation = True

    def __neg__(self) -> Gene:
        """Reverse the gene orientation."""
        return Gene(
            id=self.id,
            orientation=not self.orientation,
        )

    def clone(self) -> Gene:
        """Make a copy of this gene with a fresh ID."""
        return Gene(orientation=self.orientation)

    def __str__(self) -> str:
        return self.id if self.orientation else "-" + self.id

    def serialize(self) -> SerializedGene:
        """Convert the gene to a JSON-serializable structure."""
        return (self.id, self.orientation)

    @classmethod
    def unserialize(cls, data: SerializedGene) -> Gene:
        """Convert back a gene from its serialized form."""
        return Gene(id=data[0], orientation=data[1])


SerializedSynteny = list[SerializedGene]


@dataclass
class Synteny:
    """A synteny made up of an oriented sequence of genes."""
    id: SyntenyId = field(default_factory=IdGenerator("X"))
    genes: list[Gene] = field(default_factory=list)

    @classmethod
    def of(cls, *genes: Gene) -> Synteny:
        """Make a synteny from a sequence of genes."""
        return Synteny(genes=list(genes))

    def __post_init__(self) -> None:
        if not self.genes:
            raise RuntimeError("synteny must have at least one gene")

    def __add__(self, other: Synteny) -> Synteny:
        """Concatenate two syntenies."""
        return Synteny(id=self.id, genes=self.genes + other.genes)

    def __neg__(self) -> Synteny:
        """Reverse the synteny."""
        return self[::-1]

    def __iter__(self) -> Iterator[Gene]:
        return iter(self.genes)

    @overload
    def __getitem__(self, key: slice) -> Synteny: ...

    @overload
    def __getitem__(self, key: int) -> Gene: ...

    def __getitem__(self, key: slice|int) -> Synteny|Gene:
        """Get a gene or segment of genes from the synteny."""
        if isinstance(key, slice):
            result = Synteny(id=self.id, genes=self.genes[key])

            # Reverse individual genes if segment is reversed
            if key.step is not None and key.step < 0:
                for i in range(len(result.genes)):
                    result.genes[i] = -result.genes[i]

            return result

        return self.genes[key]

    def __delitem__(self, key: slice|int) -> None:
        """Remove a gene or segment of genes from the synteny."""
        del self.genes[key]

    @overload
    def __setitem__(self, key: slice, value: list[Gene]|Synteny) -> None: ...

    @overload
    def __setitem__(self, key: int, value: Gene) -> None: ...

    def __setitem__(self, key: slice|int, value: list[Gene]|Synteny|Gene):
        """Replace or insert genes in the synteny."""

        if isinstance(key, int):
            if not isinstance(value, Gene):
                raise TypeError("can only assign a single gene")

            self.genes[key] = value
        else:
            if isinstance(value, Synteny):
                value = value.genes
            elif not isinstance(value, list):
                raise TypeError("can only assign an iterable")

            self.genes[key] = value

    def clone(self) -> Synteny:
        """Make a copy of this synteny with a fresh synteny ID."""
        return Synteny(genes=self.genes)

    def deep_clone(self) -> Synteny:
        """Make a copy of this synteny with fresh IDs for all of its parts."""
        return Synteny(genes=[gene.clone() for gene in self.genes])

    def __len__(self) -> int:
        return len(self.genes)

    def __str__(self) -> str:
        return f"{self.id}: ({', '.join(map(str, self.genes))})"

    def serialize(self) -> SerializedSynteny:
        """Convert the synteny to a JSON-serializable structure."""
        return [gene.serialize() for gene in self.genes]

    @classmethod
    def unserialize(cls, data: SerializedSynteny) -> Synteny:
        """Convert back a synteny from its serialized form."""
        return Synteny.of(*(Gene.unserialize(gene) for gene in data))


SerializedSpecies = dict[SyntenyId, SerializedSynteny]


@dataclass
class Species:
    """A species holding an unordered set of independent syntenies."""
    id: SpeciesId = field(default_factory=IdGenerator("S"))
    syntenies: dict[SyntenyId, Synteny] = field(default_factory=dict)

    @classmethod
    def of(cls, *syntenies: Synteny) -> Species:
        """Make a species from a set of syntenies."""
        return Species(syntenies={
            synteny.id: synteny for synteny in syntenies
        })

    def __post_init__(self) -> None:
        if not self.syntenies:
            raise RuntimeError("species must have at least one synteny")

    def __getitem__(self, key: SyntenyId) -> Synteny:
        """Get a synteny from the species."""
        return self.syntenies[key]

    def add(self, synteny: Synteny) -> None:
        """Add a synteny to this species."""
        self.syntenies[synteny.id] = synteny

    def remove(self, synteny: Synteny|SyntenyId) -> None:
        """Remove a synteny from this species."""
        if isinstance(synteny, Synteny):
            del self.syntenies[synteny.id]
        else:
            del self.syntenies[synteny]

    def clone(self) -> Species:
        """Make a copy of this species with a fresh species ID."""
        return Species(syntenies=self.syntenies)

    def deep_clone(self) -> Species:
        """Make a copy of this species with fresh IDs for all of its parts."""
        return Species(syntenies={
            clone.id: clone
            for key in sorted(self.syntenies.keys())
            if (clone := self.syntenies[key].deep_clone())
        })

    def __len__(self) -> int:
        return len(self.syntenies)

    def __str__(self) -> str:
        syntenies = [str(synteny) for key, synteny in self.syntenies.items()]
        return f"{self.id}: {{{', '.join(syntenies)}}}"

    def serialize(self) -> SerializedSpecies:
        """Convert the species to a JSON-serializable structure."""
        return {
            key: synteny.serialize()
            for key, synteny in self.syntenies.items()
        }

    @classmethod
    def unserialize(self, data: SerializedSpecies) -> Species:
        """Convert back a species from its serialized form."""
        return Species.of(*(
            replace(
                Synteny.unserialize(synteny),
                id=key
            )
            for key, synteny in data.items()
        ))
