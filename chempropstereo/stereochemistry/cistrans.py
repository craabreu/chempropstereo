import numpy as np
from rdkit import Chem

from . import utils

_CIS = Chem.BondStereo.STEREOCIS
_TRANS = Chem.BondStereo.STEREOTRANS


def get_cis_trans_neighbors(atom: Chem.Atom) -> tuple[int, ...]:
    neighbors = atom.GetNeighbors()
    return tuple(
        neighbors[i].GetIdx() for i in map(int, atom.GetProp("canonicalCisTransTag"))
    )


def tag_cis_trans_stereobonds(mol: Chem.Mol) -> None:
    r"""
    Tag tetrahedral stereocenters in a molecule as clockwise or counterclockwise based
    on their neighbors arranged in a descending order of their canonical ranks.

    Parameters
    ----------
    mol
        The molecule whose tetrahedral stereocenters are to be tagged.

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
    ...       tag = bond.GetIntProp("canonicalCisTransTag")
    ...       if tag != 0:
    ...          neighbors = []
    ...          for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
    ...             indices = stereochemistry.get_cis_trans_neighbors(atom)
    ...             neighbors.extend(map(desc, map(mol.GetAtomWithIdx, indices)))
    ...          print("CIS" if tag == 1 else "TRANS", *neighbors)
    TRANS N0 O2 C5 S4
    TRANS N0 O2 C4 S5
    TRANS N2 O0 C5 S4
    """
    Chem.SetBondStereoFromDirections(mol)
    for bond in mol.GetBonds():
        tag = bond.GetStereo()
        if tag in (_CIS, _TRANS):
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
            for atom, indices in zip(connected_atoms, ranked_neighbor_indices):
                atom.SetProp("canonicalCisTransTag", utils.concat(*indices))
            stereo_atoms = list(bond.GetStereoAtoms())
            flip = False
            for atoms, indices in zip(neighbors, ranked_neighbor_indices):
                if atoms[indices[0]] not in stereo_atoms:
                    flip = not flip
            alignment = 1 if (tag == _TRANS) == flip else 2
            bond.SetIntProp("canonicalCisTransTag", alignment)
        else:
            bond.SetIntProp("canonicalCisTransTag", 0)
    mol.SetBoolProp("hasCanonicalCisTransTags", True)
