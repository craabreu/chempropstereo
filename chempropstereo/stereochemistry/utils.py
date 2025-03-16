import typing as t

from rdkit import Chem

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
    >>> from chempropstereo.stereochemistry.utils import get_cip_code
    >>> from rdkit import Chem
    >>> mol1 = Chem.MolFromSmiles("C[C@H](N)O")
    >>> [get_cip_code(atom) for atom in mol1.GetAtoms()]
    [0, 1, 0, 0]
    >>> mol2 = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> [get_cip_code(atom) for atom in mol2.GetAtoms()]
    [0, 2, 0, 0]
    """
    return _CIP_CODES[atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode")]


def _swap_if_ascending(
    i: int, j: int, a: int, b: int, odd: bool
) -> tuple[int, int, int, int, bool]:
    """
    Helper function for :func:`utils.argsort_descending_with_parity` that conditionally
    swaps two indices and their corresponding values, and flips the parity of the
    permutation if a swap is performed.

    Parameters
    ----------
    i
        The first index.
    j
        The second index.
    a
        The value at the first index.
    b
        The value at the second index.
    odd
        Whether the permutation is currently odd rather than even.

    Returns
    -------
    tuple[int, int, int, int, bool]
        A tuple containing:

        - The potentially swapped indices i and j.
        - The potentially swapped values a and b.
        - A boolean indicating whether the permutation is odd after having or not
          performed the swap.
    """
    if a < b:
        return j, i, b, a, not odd
    return i, j, a, b, odd


def argsort_descending_with_parity(
    a: int, b: int, c: int, d: t.Optional[int] = None
) -> tuple[tuple[int, ...], bool]:
    """
    Perform an indirect sort on three or four integers and returns the indices that
    would arrange them in descending order, as well as a boolean indicating whether
    the resulting permutation is odd rather than even.

    Parameters
    ----------
    a
        The first integer.
    b
        The second integer.
    c
        The third integer.
    d
        The fourth integer, optional.

    Returns
    -------
    tuple[tuple[int, ...], bool]
        A tuple containing:

        - A tuple of indices that sort the integers in descending order.
        - A boolean indicating whether the required permutation is odd.

    Examples
    --------
    >>> from chempropstereo.stereochemistry.utils import argsort_descending_with_parity
    >>> argsort_descending_with_parity(9, 2, 1)
    ((0, 1, 2), False)
    >>> argsort_descending_with_parity(3, 6, 1)
    ((1, 0, 2), True)
    >>> argsort_descending_with_parity(3, 1, 2)
    ((0, 2, 1), True)
    >>> argsort_descending_with_parity(3, 6, 1, 8)
    ((3, 1, 0, 2), False)
    """
    i, j, a, b, odd = _swap_if_ascending(0, 1, a, b, False)
    j, k, b, c, odd = _swap_if_ascending(j, 2, b, c, odd)
    if d is None:
        i, j, a, b, odd = _swap_if_ascending(i, j, a, b, odd)
        return (i, j, k), odd
    k, m, c, d, odd = _swap_if_ascending(k, 3, c, d, odd)
    i, j, a, b, odd = _swap_if_ascending(i, j, a, b, odd)
    j, k, b, c, odd = _swap_if_ascending(j, k, b, c, odd)
    i, j, a, b, odd = _swap_if_ascending(i, j, a, b, odd)
    return (i, j, k, m), odd


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
    >>> from chempropstereo.stereochemistry import utils
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> utils.tag_tetrahedral_stereocenters(mol)
    >>> chiral_atom = mol.GetAtomWithIdx(1)  # The chiral carbon
    >>> utils.get_scan_direction(chiral_atom)
    1
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> utils.get_scan_direction(non_chiral_atom)
    0
    """
    return int(atom.GetProp("canonicalChiralTag")[0])


def get_neighbors_in_canonical_order(atom: Chem.Atom) -> tuple[int, ...]:
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
    >>> from chempropstereo.stereochemistry import utils
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> utils.tag_tetrahedral_stereocenters(mol)
    >>> chiral_atom = mol.GetAtomWithIdx(1)  # The chiral carbon
    >>> utils.get_neighbors_in_canonical_order(chiral_atom)
    (3, 2, 0)
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> utils.get_neighbors_in_canonical_order(non_chiral_atom)
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
    >>> from chempropstereo.stereochemistry import utils
    >>> from rdkit import Chem
    >>> def desc(atom):
    ...     return f"{atom.GetSymbol()}{atom.GetIdx()}"
    >>> for smi in ["C[C@H](N)O", "C[C@@H](O)N"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     utils.tag_tetrahedral_stereocenters(mol)
    ...     for atom in mol.GetAtoms():
    ...         direction = [None, "CW", "CCW"][utils.get_scan_direction(atom)]
    ...         if direction:
    ...             neighbors = [
    ...                 desc(mol.GetAtomWithIdx(idx))
    ...                 for idx in utils.get_neighbors_in_canonical_order(atom)
    ...             ]
    ...             print(desc(atom), f"({direction})", *neighbors)
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
            order, flip = argsort_descending_with_parity(*neighbor_ranks)
            direction = 1 if (tag == _CCW) == flip else 2
            atom.SetProp("canonicalChiralTag", "".join(map(str, [direction, *order])))
        else:
            atom.SetProp("canonicalChiralTag", "0")
