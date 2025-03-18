import typing as t

from rdkit import Chem


def _swap_if_ascending(
    i: int, j: int, a: int, b: int, odd: bool
) -> tuple[int, int, int, int, bool]:
    """
    Helper function for :func:`~stereochemistry._argsort_descending_with_parity` that
    conditionally swaps two indices and their corresponding values, and flips the
    parity of the permutation if a swap is performed.

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
    >>> from chempropstereo.stereochemistry import utils
    >>> utils.argsort_descending_with_parity(9, 2, 1)
    ((0, 1, 2), False)
    >>> utils.argsort_descending_with_parity(3, 6, 1)
    ((1, 0, 2), True)
    >>> utils.argsort_descending_with_parity(3, 1, 2)
    ((0, 2, 1), True)
    >>> utils.argsort_descending_with_parity(3, 6, 1, 8)
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


def concat(*args: t.Any) -> str:
    """
    Concatenate the string representations of the given arguments.

    Parameters
    ----------
    *args
        The arguments to be concatenated.

    Returns
    -------
    str
        The concatenated string.

    Examples
    --------
    >>> from chempropstereo.stereochemistry import utils
    >>> utils.concat(1, 2, 3)
    '123'
    >>> utils.concat("a", "b", "c")
    'abc'
    >>> utils.concat(1, "a", 2, "b", 3, "c")
    '1a2b3c'
    """
    return "".join(map(str, args))


def get_bond_ends(
    bond: Chem.Bond, reverse: bool = False
) -> tuple[Chem.Atom, Chem.Atom]:
    """
    Get the atoms at the ends of a bond.

    Parameters
    ----------
    bond : Chem.Bond
        The bond to get the end atoms from.
    reverse : bool, optional
        Whether to reverse the direction of the bond (default is False).

    Returns
    -------
    tuple[Chem.Atom, Chem.Atom]
        The atoms at the ends of the bond.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo.stereochemistry import utils
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> bond = mol.GetBondWithIdx(0)
    >>> [atom.GetIdx() for atom in utils.get_bond_ends(bond)]
    [0, 1]
    >>> [atom.GetIdx() for atom in utils.get_bond_ends(bond, reverse=True)]
    [1, 0]
    """
    if reverse:
        return bond.GetEndAtom(), bond.GetBeginAtom()
    return bond.GetBeginAtom(), bond.GetEndAtom()
