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


class BranchRank(base.Rank):
    NONE = 0
    MAJOR = 1
    MINOR = 2


def describe_stereobond(bond: Chem.Bond) -> str:
    """
    Describe a cis/trans stereobond.

    Parameters
    ----------
    bond : Chem.Bond
        The bond to describe.

    Returns
    -------
    str
        A string description of the cis/trans stereobond.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("N/C(O)=C(S)/C")
    >>> stereochemistry.tag_cis_trans_stereobonds(mol)
    >>> stereochemistry.describe_stereobond(mol.GetBondWithIdx(2))
    'N0 O2 C1 (TRANS) C3 C5 S4'
    """
    arrangement = StemArrangement.get_from(bond)
    if arrangement == StemArrangement.NONE:
        return "Not a stereobond"
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    return (
        " ".join(map(utils.describe_atom, BranchRank._get_neighbors(begin)))
        + " "
        + f" ({arrangement.name}) ".join(map(utils.describe_atom, (begin, end)))
        + " "
        + " ".join(map(utils.describe_atom, BranchRank._get_neighbors(end)))
    )


def tag_cis_trans_stereobonds(mol: Chem.Mol, force: bool = False) -> None:
    r"""
    Tag cis/trans stereobonds in a molecule based on their spatial arrangement.

    Parameters
    ----------
    mol
        The molecule whose cis/trans stereobonds are to be tagged.
    force
        Whether to overwrite existing stereobond tags (default is False).

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
    ...       arrangement = stereochemistry.StemArrangement.get_from(bond)
    ...       if arrangement != stereochemistry.StemArrangement.NONE:
    ...          print(stereochemistry.describe_stereobond(bond))
    N0 O2 C1 (TRANS) C3 C5 S4
    N0 O2 C1 (TRANS) C3 C4 S5
    N2 O0 C1 (TRANS) C3 C5 S4
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
                atom.SetProp("canonicalStereoTag", utils.concat(arrangement, *indices))
        elif bond.HasProp("canonicalStereoTag"):
            bond.ClearProp("canonicalStereoTag")
    mol.SetBoolProp("hasCanonicalStereobonds", hasStereobonds)
