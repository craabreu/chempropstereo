import enum
import typing as t

from rdkit import Chem


class SpatialArrangement(enum.IntEnum):
    """Base class for enumerating spatial arrangements in stereogenic groups."""

    @classmethod
    def get_from(cls, entity: Chem.Atom | Chem.Bond) -> t.Self:
        """
        Classify the spatial arrangement of a stereogenic group based on an atom's or
        bond's canonical stereo tag.

        Parameters
        ----------
        entity
            The atom or bond to get the spatial arrangement from.

        Returns
        -------
        SpatialArrangement
            The spatial arrangement obtained from the atom or bond.
        """
        if entity.HasProp(cls.tag):
            return cls(int(entity.GetProp(cls.tag)[0]))
        return cls.NONE


class Rank(enum.IntEnum):
    """Base class for ranks in stereogenic groups."""

    @classmethod
    def _get_neighbors(cls, atom: Chem.Atom) -> tuple[Chem.Atom, ...]:
        neighbors = atom.GetNeighbors()
        order = map(int, atom.GetProp(cls.tag)[1:])
        return tuple(neighbors[i] for i in order)

    @classmethod
    def from_bond(cls, bond: Chem.Bond, end_is_center: bool = False) -> t.Self:
        """
        Get the rank of a bond in a stereogenic group from its center atom's
        canonical stereo tag.

        Parameters
        ----------
        bond
            The bond to get the rank from.
        end_is_center
            Whether to treat the end atom as the center of the stereogenic group.
            If False (the default), the begin atom is treated as the center.

        Returns
        -------
        Rank
            The rank of the bond in the stereogenic group.
        """

        if end_is_center:
            center_atom, edge_index = bond.GetEndAtom(), bond.GetBeginAtomIdx()
        else:
            center_atom, edge_index = bond.GetBeginAtom(), bond.GetEndAtomIdx()
        if not center_atom.HasProp(cls.tag):
            return cls.NONE
        neighbors = cls._get_neighbors(center_atom)
        for rank, neighbor in enumerate(neighbors, start=1):
            if neighbor.GetIdx() == edge_index:
                return cls(rank)
        return cls.NONE
