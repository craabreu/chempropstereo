"""Utilities for printing featurized molecules."""

import numpy as np


def describe_atom_features(
    index: int, features: np.ndarray, sizes: tuple[int, ...]
) -> str:
    """Format an atom feature array into a string.

    Parameters
    ----------
    index : int
        The index of the atom.
    features : np.ndarray
        The feature array of the atom.
    sizes : tuple[int, ...]
        The size of each feature type.

    Returns
    -------
    str
        A formatted string representing the atom features, including the
        atom index, bit features, and mass feature.

    Examples
    --------
    >>> import numpy as np
    >>> features = np.array([1, 0, 1, 0, 1, 0.120])
    >>> sizes = (2, 2, 1, 1)
    >>> describe_atom_features(1, features, sizes)
    '  1: 10 10 1 0.120'

    """
    atom_desc = str(index).rjust(3)
    s = "".join(map(str, map(int, features[:-1])))
    cuts = list(np.cumsum(sizes[:-1]))
    bits_desc = " ".join(s[a:b] for a, b in zip([0] + cuts, cuts))
    mass_desc = f"{features[-1]:.3f}"
    return f"{atom_desc}: {bits_desc} {mass_desc}"


def describe_bond_features(
    atoms: tuple[int, int], features: np.ndarray, sizes: tuple[int, ...]
) -> str:
    """Format a bond feature array into a string.

    Parameters
    ----------
    atoms : tuple[int, int]
        The indices of the atoms connected by the bond.
    features : np.ndarray
        The feature array of the bond.
    sizes : tuple[int, ...]
        The size of each feature type.

    Returns
    -------
    str
        A formatted string representing the bond features.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import featurizers
    >>> mol = Chem.MolFromSmiles('CC')
    >>> atoms = (0, 1)
    >>> features = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> sizes = (6, 2, 1, 3, 4, 2)
    >>> describe_bond_features(atoms, features, sizes)
    '    0→1: 010000 00 0 000 0111 11'

    """
    bond_desc = "\u2192".join(map(str, atoms)).rjust(7)
    s = "".join(map(str, map(int, features)))
    cuts = list(np.cumsum(sizes))
    return f"{bond_desc}: " + " ".join(s[a:b] for a, b in zip([0] + cuts, cuts))
