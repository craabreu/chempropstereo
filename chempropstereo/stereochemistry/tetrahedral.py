
from rdkit import Chem

from . import base, utils

_CIP_CODES = {False: 0, "R": 1, "S": 2}


class ScanDirection(base.SpatialArrangement):
    """
    Enumeration for scan directions in canonicalized tetrahedral stereocenters.

    Attributes
    ----------
    NONE : int
        Not a stereocenter.
    CW : int
        Second, third, and fourth vertices must be scanned clockwise.
    CCW : int
        Second, third, and fourth vertices must be scanned counterclockwise.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(mol)
    >>> ScanDirection.get_from(mol.GetAtomWithIdx(1))
    <ScanDirection.CW: 1>
    >>> ScanDirection.get_from(mol.GetAtomWithIdx(2))
    <ScanDirection.NONE: 0>
    """

    NONE = 0
    CW = 1
    CCW = 2


class VertexRank(base.Rank):
    """
    Enumeration of vertex ranks for tetrahedral stereochemistry.

    Attributes
    ----------
    NONE : int
        Not a tetrahedral vertex.
    FIRST : int
        The first vertex in canonical order.
    SECOND : int
        The second vertex in canonical order.
    THIRD : int
        The third vertex in canonical order.
    FOURTH : int
        The fourth vertex in canonical order.


    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(mol)
    >>> bond = mol.GetBondWithIdx(1)
    >>> VertexRank.get_from(bond)
    <VertexRank.SECOND: 2>
    >>> VertexRank.get_from(bond, reverse=True)
    <VertexRank.NONE: 0>
    """

    NONE = 0
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


def get_cip_code(atom: Chem.Atom) -> int:
    """
    Get the CIP code of an atom as an integer.

    Parameters
    ----------
    atom
        The atom to get the CIP code of.

    Returns
    -------
    int
        The CIP code of the atom: 0 for no CIP code, 1 for R, 2 for S.

    Examples
    --------
    >>> from chempropstereo import stereochemistry
    >>> from rdkit import Chem
    >>> mol1 = Chem.MolFromSmiles("C[C@H](N)O")
    >>> [stereochemistry.get_cip_code(atom) for atom in mol1.GetAtoms()]
    [0, 1, 0, 0]
    >>> mol2 = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> [stereochemistry.get_cip_code(atom) for atom in mol2.GetAtoms()]
    [0, 2, 0, 0]
    """
    return _CIP_CODES[atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode")]


def get_stereocenter_neighbors(atom: Chem.Atom) -> tuple[int, ...]:
    """
    Extract the indices of neighbor atoms in canonical order from an atom with a
    canonical chiral tag.

    Parameters
    ----------
    atom : Chem.Atom
        The atom whose neighbor information is to be extracted.

    Returns
    -------
    tuple[int, ...]
        The indices of the ordered neighbor atoms, or an empty tuple if the atom has
        no chirality information.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(mol)
    >>> chiral_atom = mol.GetAtomWithIdx(1)  # The chiral carbon
    >>> stereochemistry.get_stereocenter_neighbors(chiral_atom)
    (3, 2, 0)
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> stereochemistry.get_stereocenter_neighbors(non_chiral_atom)
    ()
    """
    if not atom.HasProp("canonicalStereoTag"):
        return ()
    neighbors = atom.GetNeighbors()
    return tuple(
        neighbors[i].GetIdx() for i in map(int, atom.GetProp("canonicalStereoTag")[1:])
    )


def tag_tetrahedral_stereocenters(mol: Chem.Mol, force: bool = False) -> None:
    """
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
    >>> for smi in ["C[C@H](N)O", "C[C@@H](O)N"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     stereochemistry.tag_tetrahedral_stereocenters(mol)
    ...     for atom in mol.GetAtoms():
    ...         direction = stereochemistry.ScanDirection.get_from(atom).name
    ...         if direction != "NONE":
    ...             neighbors = [
    ...                 desc(mol.GetAtomWithIdx(idx))
    ...                 for idx in stereochemistry.get_stereocenter_neighbors(atom)
    ...             ]
    ...             print(desc(atom), f"({direction})", *neighbors)
    C1 (CW) O3 N2 C0
    C1 (CW) O2 N3 C0
    """
    if mol.HasProp("hasCanonicalStereocenters") and not force:
        return
    hasStereocenters = False
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            hasStereocenters = True
            neighbors = [atom.GetIdx() for atom in atom.GetNeighbors()]
            all_ranks = list(Chem.CanonicalRankAtomsInFragment(mol, neighbors))
            neighbor_ranks = [all_ranks[idx] for idx in neighbors]
            # Sorting ranks in descending order keeps explicit hydrogens at the end
            order, flip = utils.argsort_descending_with_parity(*neighbor_ranks)
            if (tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW) == flip:
                direction = ScanDirection.CW
            else:
                direction = ScanDirection.CCW
            atom.SetProp("canonicalStereoTag", utils.concat(direction, *order))
    mol.SetBoolProp("hasCanonicalStereocenters", hasStereocenters)
