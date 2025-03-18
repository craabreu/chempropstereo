import typing as t
from enum import IntEnum

from rdkit import Chem

from . import utils


class SpatialArrangement(IntEnum):
    """Base class for spatial arrangements in stereogenic groups."""

    @classmethod
    def get_from(cls, entity: Chem.Atom | Chem.Bond) -> t.Self:
        """
        Get the spatial arrangement of a stereogenic group from an atom's or bond's
        canonical stereo tag.

        Parameters
        ----------
        entity : Chem.Atom | Chem.Bond
            The atom or bond to get the spatial arrangement from.

        Returns
        -------
        SpatialArrangement
            The spatial arrangement obtained from the atom or bond.
        """
        if entity.HasProp("canonicalStereoTag"):
            return cls(int(entity.GetProp("canonicalStereoTag")[0]))
        return cls.NONE


class Rank(IntEnum):
    """Base class for ranks in stereogenic groups."""

    @staticmethod
    def _get_neighbors(atom: Chem.Atom) -> tuple[Chem.Atom, ...]:
        neighbors = atom.GetNeighbors()
        order = map(int, atom.GetProp("canonicalStereoTag")[1:])
        return tuple(neighbors[i] for i in order)

    @classmethod
    def get_from(cls, bond: Chem.Bond, reverse: bool = False) -> t.Self:
        """
        Get the rank of a bond in a stereogenic group from its begin (or end) atom's
        canonical stereo tag.

        Parameters
        ----------
        bond : Chem.Bond
            The bond to get the rank from.
        reverse : bool, optional
            Whether to reverse the direction of the bond (default is False).

        Returns
        -------
        Rank
            The rank obtained from the bond.
        """
        begin_atom, end_atom = utils.get_bond_ends(bond, reverse)
        if not begin_atom.HasProp("canonicalStereoTag"):
            return cls.NONE
        neighbors = cls._get_neighbors(begin_atom)
        for rank, neighbor in enumerate(neighbors, start=1):
            if neighbor.GetIdx() == end_atom.GetIdx():
                return cls(rank)
        return cls.NONE
