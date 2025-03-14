import typing as t

from rdkit import Chem

_CIP_CODES = {False: 0, "R": 1, "S": 2}
_CHIRAL_TAGS = {None: 0, "CW": 1, "CCW": 2}
_SCAN_DIRECTIONS = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: "CW",
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "CCW",
}


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
    >>> from chempropstereo.featurizers.utils import get_cip_code
    >>> from rdkit import Chem
    >>> mol1 = Chem.MolFromSmiles("C[C@H](N)O")
    >>> [get_cip_code(atom) for atom in mol1.GetAtoms()]
    [0, 1, 0, 0]
    >>> mol2 = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> [get_cip_code(atom) for atom in mol2.GetAtoms()]
    [0, 2, 0, 0]
    """
    return _CIP_CODES[atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode")]


def _swap_or_not(
    i: int, j: int, a: int, b: int, odd: bool
) -> tuple[int, int, int, int, bool]:
    """
    Helper function for argsort_descending_with_parity that conditionally swaps two
    indices and values, and returns whether the permutation is odd after the swap.

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
        Whether the permutation is currently odd.

    Returns
    -------
    tuple[int, int, int, int, bool]
        A tuple containing:
        - The potentially swapped indices (i, j).
        - The potentially swapped values (a, b).
        - A boolean indicating whether the permutation is odd after the potential swap.
    """
    if a < b:
        return j, i, b, a, not odd
    return i, j, a, b, odd


def argsort_descending_with_parity(
    a: int, b: int, c: int, d: int | None = None
) -> tuple[bool, tuple[int, ...]]:
    """
    Perform an indirect sort on three or four integers and returns the indices that
    sort them in descending order, as well as whether the required permutation is odd.

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
    tuple[bool, tuple[int, ...]]
        A tuple containing:
        - A boolean indicating whether the required permutation is odd.
        - A tuple of indices that sort the integers in descending order.

    Examples
    --------
    >>> from chempropstereo.featurizers.utils import argsort_descending_with_parity
    >>> argsort_descending_with_parity(9, 2, 1)
    (False, (0, 1, 2))
    >>> argsort_descending_with_parity(3, 6, 1)
    (True, (1, 0, 2))
    >>> argsort_descending_with_parity(3, 1, 2)
    (True, (0, 2, 1))
    >>> argsort_descending_with_parity(3, 6, 1, 8)
    (False, (3, 1, 0, 2))
    """
    i, j, a, b, odd = _swap_or_not(0, 1, a, b, False)
    j, k, b, c, odd = _swap_or_not(j, 2, b, c, odd)
    if d is None:
        i, j, a, b, odd = _swap_or_not(i, j, a, b, odd)
        return odd, (i, j, k)
    k, m, c, d, odd = _swap_or_not(k, 3, c, d, odd)
    i, j, a, b, odd = _swap_or_not(i, j, a, b, odd)
    j, k, b, c, odd = _swap_or_not(j, k, b, c, odd)
    i, j, a, b, odd = _swap_or_not(i, j, a, b, odd)
    return odd, (i, j, k, m)


def get_chiral_tag(direction: str, order: t.Sequence[int]) -> str:
    """
    Construct a canonical chiral tag representing the stereochemistry configuration.

    Parameters
    ----------
    direction
        The scan direction of the neighbors, either 'CW' for clockwise or 'CCW' for
        counterclockwise.
    order
        A sequence of integers representing the order of neighbors in the configuration.

    Returns
    -------
    str
        A string combining the scan direction and the ordered indices of the
        neighbors, separated by a colon.

    Examples
    --------
    >>> from chempropstereo.featurizers.utils import get_chiral_tag
    >>> get_chiral_tag("CW", [0, 2, 1])
    'CW:021'
    >>> get_chiral_tag("CCW", [1, 2, 0])
    'CCW:120'
    >>> get_chiral_tag("CW", [2, 0, 1, 3])
    'CW:2013'
    """
    return direction + ":" + "".join(map(str, order))


def get_scan_direction(atom: Chem.Atom, numeric: bool = False) -> str | None:
    """
    Extract the scan direction of the neighbors from an atom with a canonical
    chiral tag.

    Parameters
    ----------
    atom
        The atom whose chiral tag is to be extracted.
    numeric
        Whether to return the scan direction as an integer.

    Returns
    -------
    str | None
        The scan direction ('CW' or 'CCW'), or None if the atom has no chirality
        property.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo.featurizers import utils
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> utils.tag_tetrahedral_stereocenters(mol)
    >>> chiral_atom = mol.GetAtomWithIdx(1)  # The chiral carbon
    >>> utils.get_scan_direction(chiral_atom)
    'CW'
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> utils.get_scan_direction(non_chiral_atom) is None
    True
    """
    if numeric:
        return _CHIRAL_TAGS[get_scan_direction(atom)]
    if not atom.HasProp("chiral_tag"):
        return None
    direction, _ = atom.GetProp("chiral_tag").split(":")
    return direction


def get_direction_and_neighbors(
    atom: Chem.Atom,
) -> tuple[str, tuple[Chem.Atom, ...]] | tuple[None, None]:
    """
    Extract the scan direction and ordered neighbor atoms from an chiral atom.

    Parameters
    ----------
    atom
        The atom whose chiral information is to be extracted.

    Returns
    -------
    tuple[str, tuple[Chem.Atom, ...]] | tuple[None, None]
        A tuple containing the scan direction ('CW' or 'CCW') and the ordered
        neighbors, or (None, None) if the atom has no chirality information.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo.featurizers import utils
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> utils.tag_tetrahedral_stereocenters(mol)
    >>> chiral_atom = mol.GetAtomWithIdx(1)  # The chiral carbon
    >>> direction, neighbors = utils.get_direction_and_neighbors(chiral_atom)
    >>> direction
    'CW'
    >>> [n.GetSymbol() for n in neighbors]
    ['O', 'N', 'C']
    >>> non_chiral_atom = mol.GetAtomWithIdx(0)  # A non-chiral atom
    >>> utils.get_direction_and_neighbors(non_chiral_atom)
    (None, None)
    """
    if not atom.HasProp("chiral_tag"):
        return None, None
    direction, order_str = atom.GetProp("chiral_tag").split(":")
    order = tuple(map(int, order_str))
    neighbors = atom.GetNeighbors()
    return direction, tuple(neighbors[i] for i in order)


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
    >>> from chempropstereo.featurizers import utils
    >>> from rdkit import Chem
    >>> def desc(atom):
    ...     return f"{atom.GetSymbol()}{atom.GetIdx()}"
    >>> for smi in ["C[C@H](N)O", "C[C@@H](O)N"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     utils.tag_tetrahedral_stereocenters(mol)
    ...     for atom in mol.GetAtoms():
    ...         direction, neighbors = utils.get_direction_and_neighbors(atom)
    ...         if direction is not None:
    ...             print(desc(atom), f"({direction})", *map(desc, neighbors))
    C1 (CW) O3 N2 C0
    C1 (CW) O2 N3 C0
    """
    for atom in mol.GetAtoms():
        direction = _SCAN_DIRECTIONS.get(atom.GetChiralTag(), None)
        if direction is not None:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            all_ranks = list(Chem.CanonicalRankAtomsInFragment(mol, neighbors))
            neighbor_ranks = [all_ranks[n.GetIdx()] for n in atom.GetNeighbors()]
            # Sorting ranks in descending order keeps explicit hydrogens at the end
            flip, order = argsort_descending_with_parity(*neighbor_ranks)
            if flip:
                direction = "CCW" if direction == "CW" else "CW"
            atom.SetProp("chiral_tag", get_chiral_tag(direction, order))
