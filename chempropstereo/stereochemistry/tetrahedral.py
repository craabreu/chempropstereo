from rdkit import Chem

from . import utils

_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW

_CIP_CODES = {False: 0, "R": 1, "S": 2}


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


def get_scan_direction(atom: Chem.Atom) -> int:
    """
    Extract the scan direction of the neighbors from an atom with a canonical
    chiral tag.

    Parameters
    ----------
    atom : Chem.Atom
        The atom whose chiral tag is to be extracted.

    Returns
    -------
    int
        The scan direction as an integer (1 for 'CW', 2 for 'CCW'), or 0 if the atom
        has no chirality property.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(mol)
    >>> chiral_atom = mol.GetAtomWithIdx(1)  # The chiral carbon
    >>> stereochemistry.get_scan_direction(chiral_atom)
    1
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> stereochemistry.get_scan_direction(non_chiral_atom)
    0
    """
    return int(atom.GetProp("canonicalChiralTag")[0])


def get_neighbors(atom: Chem.Atom) -> tuple[int, ...]:
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
    >>> stereochemistry.get_neighbors(chiral_atom)
    (3, 2, 0)
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> stereochemistry.get_neighbors(non_chiral_atom)
    ()
    """
    neighbors = atom.GetNeighbors()
    return tuple(
        neighbors[i].GetIdx() for i in map(int, atom.GetProp("canonicalChiralTag")[1:])
    )


def tag_tetrahedral_stereocenters(mol: Chem.Mol) -> None:
    """
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
    >>> for smi in ["C[C@H](N)O", "C[C@@H](O)N"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     stereochemistry.tag_tetrahedral_stereocenters(mol)
    ...     for atom in mol.GetAtoms():
    ...         direction = stereochemistry.get_scan_direction(atom)
    ...         if direction != 0:
    ...             neighbors = [
    ...                 desc(mol.GetAtomWithIdx(idx))
    ...                 for idx in stereochemistry.get_neighbors(atom)
    ...             ]
    ...             rotation = "CW" if direction == 1 else "CCW"
    ...             print(desc(atom), f"({rotation})", *neighbors)
    C1 (CW) O3 N2 C0
    C1 (CW) O2 N3 C0
    """
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag in (_CW, _CCW):
            neighbors = [atom.GetIdx() for atom in atom.GetNeighbors()]
            all_ranks = list(Chem.CanonicalRankAtomsInFragment(mol, neighbors))
            neighbor_ranks = [all_ranks[idx] for idx in neighbors]
            # Sorting ranks in descending order keeps explicit hydrogens at the end
            order, flip = utils.argsort_descending_with_parity(*neighbor_ranks)
            direction = 1 if (tag == _CCW) == flip else 2
            atom.SetProp("canonicalChiralTag", utils.concat(direction, *order))
        else:
            atom.SetProp("canonicalChiralTag", "0")
    mol.SetBoolProp("hasCanonicalChiralTags", True)
