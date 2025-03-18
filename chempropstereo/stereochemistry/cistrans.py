import enum

import numpy as np
from rdkit import Chem

from . import base, utils


class StemArrangement(base.SpatialArrangement):
    """
    Enumeration for cis/trans arrangements in double bonds.

    Attributes
    ----------
    NONE : int
        Not a stereobond.
    CIS : int
        The substituents are on the same side.
    TRANS : int
        The substituents are on opposite sides.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("N/C(O)=C(S)/C")
    >>> stereochemistry.tag_cis_trans_stereobonds(mol)
    >>> bond = mol.GetBondWithIdx(2)
    >>> StemArrangement.get_from(bond)
    <StemArrangement.TRANS: 2>
    >>> bond = mol.GetBondWithIdx(0)
    >>> StemArrangement.get_from(bond)
    <StemArrangement.NONE: 0>
    """

    NONE = 0
    CIS = 1
    TRANS = 2


class BranchRank(enum.IntEnum):
    NONE = 0
    MAJOR = 1
    MINOR = 2


def get_cis_trans_neighbors(atom: Chem.Atom) -> tuple[int, ...]:
    neighbors = atom.GetNeighbors()
    return tuple(
        neighbors[i].GetIdx() for i in map(int, atom.GetProp("canonicalStereoTag"))
    )


def tag_cis_trans_stereobonds(mol: Chem.Mol, force: bool = False) -> None:
    r"""
    Tag tetrahedral stereocenters in a molecule as clockwise or counterclockwise based
    on their neighbors arranged in a descending order of their canonical ranks.

    Parameters
    ----------
    mol
        The molecule whose tetrahedral stereocenters are to be tagged.
    force
        Whether to overwrite existing chiral tags (default is False).

    Examples
    --------
    >>> from chempropstereo import stereochemistry
    >>> from rdkit import Chem
    >>> def desc(atom):
    ...     return f"{atom.GetSymbol()}{atom.GetIdx()}"
    >>> for smi in ["N/C(O)=C(S)/C", "N/C(O)=C(C)\\S", "O\\C(N)=C(S)/C"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     stereochemistry.tag_cis_trans_stereobonds(mol)
    ...     for bond in mol.GetBonds():
    ...       arrangement = stereochemistry.StemArrangement.get_from(bond).name
    ...       if arrangement != "NONE":
    ...          neighbors = []
    ...          for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
    ...             indices = stereochemistry.get_cis_trans_neighbors(atom)
    ...             neighbors.extend(map(desc, map(mol.GetAtomWithIdx, indices)))
    ...          print(arrangement, *neighbors)
    TRANS N0 O2 C5 S4
    TRANS N0 O2 C4 S5
    TRANS N2 O0 C5 S4
    """
    if mol.HasProp("hasCanonicalStereobonds") and not force:
        return
    Chem.SetBondStereoFromDirections(mol)
    hasStereobonds = False
    for bond in mol.GetBonds():
        tag = bond.GetStereo()
        if tag in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            hasStereobonds = True
            connected_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            indices = [atom.GetIdx() for atom in connected_atoms]
            neighbors = [
                [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
                for atom in connected_atoms
            ]
            ranks = np.fromiter(
                Chem.CanonicalRankAtomsInFragment(mol, sum(neighbors, [])), dtype=int
            )
            ranked_neighbor_indices = [
                [i for i in np.argsort(ranks[atoms]) if atoms[i] not in indices]
                for atoms in neighbors
            ]
            stereo_atoms = list(bond.GetStereoAtoms())
            flip = False
            for atoms, indices in zip(neighbors, ranked_neighbor_indices):
                if atoms[indices[0]] not in stereo_atoms:
                    flip = not flip
            if (tag == Chem.BondStereo.STEREOTRANS) == flip:
                arrangement = StemArrangement.CIS
            else:
                arrangement = StemArrangement.TRANS
            bond.SetIntProp("canonicalStereoTag", arrangement)
            for atom, indices in zip(connected_atoms, ranked_neighbor_indices):
                atom.SetProp("canonicalStereoTag", utils.concat(*indices))
    mol.SetBoolProp("hasCanonicalStereobonds", hasStereobonds)
