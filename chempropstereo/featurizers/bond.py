"""Bond featurization.

.. module:: featurizers.bond
.. moduleauthor:: Charlles Abreu <craabreu@mit.edu>
"""

import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry

_BOND_TYPES: tuple[Chem.BondType, ...] = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
_VERTEX_RANKS: tuple[stereochemistry.VertexRank, ...] = (
    stereochemistry.VertexRank.FIRST,
    stereochemistry.VertexRank.SECOND,
    stereochemistry.VertexRank.THIRD,
    stereochemistry.VertexRank.FOURTH,
)
_STEM_ARRANGEMENTS: tuple[stereochemistry.StemArrangement, ...] = (
    stereochemistry.StemArrangement.CIS,
    stereochemistry.StemArrangement.TRANS,
)
_BRANCH_RANKS: tuple[stereochemistry.BranchRank, ...] = (
    stereochemistry.BranchRank.MAJOR,
    stereochemistry.BranchRank.MINOR,
)


class BondStereoFeaturizer(chemprop.featurizers.base.VectorFeaturizer[Chem.Bond]):
    r"""Multi-hot bond featurizer that includes canonical stereochemistry information.

    The featurizer encodes the position of the end atom in the canonical order of
    neighbors when the begin atom has a canonical chiral tag.

    The featurized bonds are expected to be part of an RDKit molecule with canonical
    chiral tags assigned via :func:`tetrahedral.tag_tetrahedral_stereocenters`.

    Attributes
    ----------
    sizes: tuple[int]
        A tuple of integers representing the sizes of each bond subfeature.

    Examples
    --------
    >>> from chempropstereo import featurizers, stereochemistry
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> mol = Chem.MolFromSmiles("C\C(=C(O)/C=C(/N)O)[C@@H]([C@H](N)O)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(mol)
    >>> featurizer = featurizers.BondStereoFeaturizer()
    >>> def describe_bonds_from_atom(index):
    ...     for bond in mol.GetAtomWithIdx(index).GetBonds():
    ...         atom_is_begin = bond.GetBeginAtomIdx() == index
    ...         for reverse in (not atom_is_begin, atom_is_begin):
    ...             print(featurizer.pretty_print(bond, reverse))
    >>> stereochemistry.tag_stereogroups(mol)
    >>> stereochemistry.describe_stereobond(mol.GetBondBetweenAtoms(1, 2))
    'C0 C8 C1 (CIS) C2 C4 O3'
    >>> describe_bonds_from_atom(1) # doctest: +NORMALIZE_WHITESPACE
        1→0: 0 1000 0 0 0000 00 10
        0→1: 0 1000 0 0 0000 00 00
        1→2: 0 0100 1 0 0000 10 00
        2→1: 0 0100 1 0 0000 10 00
        1→8: 0 1000 0 0 0000 00 01
        8→1: 0 1000 0 0 0010 00 00
    >>> describe_bonds_from_atom(2) # doctest: +NORMALIZE_WHITESPACE
        2→1: 0 0100 1 0 0000 10 00
        1→2: 0 0100 1 0 0000 10 00
        2→3: 0 1000 1 0 0000 00 01
        3→2: 0 1000 1 0 0000 00 00
        2→4: 0 1000 1 0 0000 00 10
        4→2: 0 1000 1 0 0000 00 10
    >>> stereochemistry.describe_stereobond(mol.GetBondBetweenAtoms(4, 5))
    'C2 C4 (TRANS) C5 N6 O7'
    >>> describe_bonds_from_atom(4) # doctest: +NORMALIZE_WHITESPACE
        4→2: 0 1000 1 0 0000 00 10
        2→4: 0 1000 1 0 0000 00 10
        4→5: 0 0100 1 0 0000 01 00
        5→4: 0 0100 1 0 0000 01 00
    >>> describe_bonds_from_atom(5) # doctest: +NORMALIZE_WHITESPACE
        5→4: 0 0100 1 0 0000 01 00
        4→5: 0 0100 1 0 0000 01 00
        5→6: 0 1000 1 0 0000 00 10
        6→5: 0 1000 1 0 0000 00 00
        5→7: 0 1000 1 0 0000 00 01
        7→5: 0 1000 1 0 0000 00 00
    >>> stereochemistry.describe_stereocenter(mol.GetAtomWithIdx(8))
    'C8 (CCW) O12 C9 C1'
    >>> describe_bonds_from_atom(8) # doctest: +NORMALIZE_WHITESPACE
        8→1: 0 1000 0 0 0010 00 00
        1→8: 0 1000 0 0 0000 00 01
        8→9: 0 1000 0 0 0100 00 00
        9→8: 0 1000 0 0 0010 00 00
       8→12: 0 1000 0 0 1000 00 00
       12→8: 0 1000 0 0 0000 00 00
    >>> stereochemistry.describe_stereocenter(mol.GetAtomWithIdx(9))
    'C9 (CW) O11 N10 C8'
    >>> describe_bonds_from_atom(9) # doctest: +NORMALIZE_WHITESPACE
        9→8: 0 1000 0 0 0010 00 00
        8→9: 0 1000 0 0 0100 00 00
       9→10: 0 1000 0 0 0100 00 00
       10→9: 0 1000 0 0 0000 00 00
       9→11: 0 1000 0 0 1000 00 00
       11→9: 0 1000 0 0 0000 00 00

    """

    def __init__(self):
        self.bond_types = _BOND_TYPES
        self.vertex_ranks = _VERTEX_RANKS
        self.stem_arrangements = _STEM_ARRANGEMENTS
        self.branch_ranks = _BRANCH_RANKS
        self._len = sum(self.sizes)

    def __len__(self) -> int:
        return self._len

    def __call__(self, b: Chem.Bond | None, flip_direction: bool = False) -> np.ndarray:
        """Encode a bond in a molecule with canonical stereochemistry information.

        Parameters
        ----------
        b : Chem.Bond | None
            The bond to be encoded.
        flip_direction : bool, optional
            Whether to reverse the direction of the bond (default is False).

        Returns
        -------
        np.ndarray
            A vector encoding the bond.

        Notes
        -----
        The vector includes the following information:
        - Null bond indicator
        - Bond types
        - Conjugation indicator
        - Ring indicator
        - Canonical vertex rank
        - Canonical stem arrangement
        - Canonical branch rank

        """
        if b is None:
            x = np.zeros(len(self), int)
            x[0] = 1
            return x

        bond_type = b.GetBondType()
        vertex_rank = stereochemistry.VertexRank.from_bond(b, flip_direction)
        arrangement = stereochemistry.StemArrangement.get_from(b)
        branch_rank = stereochemistry.BranchRank.from_bond(b, flip_direction)

        return np.array(
            [
                b is None,
                *(bond_type == item for item in _BOND_TYPES),
                b.GetIsConjugated(),
                b.IsInRing(),
                *(vertex_rank == item for item in _VERTEX_RANKS),
                *(arrangement == item for item in _STEM_ARRANGEMENTS),
                *(branch_rank == item for item in _BRANCH_RANKS),
            ],
            dtype=int,
        )

    @property
    def sizes(self) -> list[int]:
        """Get a list of sizes corresponding to different bond features.

        The list contains the sizes for:
        - Null bond indicator
        - Bond types
        - Conjugation indicator
        - Ring indicator
        - Tetrahedral vertex ranks
        - Cis/trans stem arrangements
        - Cis/trans branch ranks

        Returns
        -------
        list[int]
            A list of integers representing the sizes of each bond feature.

        Examples
        --------
        >>> from chempropstereo import featurizers
        >>> featurizer = featurizers.BondStereoFeaturizer()
        >>> featurizer.sizes
        (1, 4, 1, 1, 4, 2, 2)

        """
        return (
            1,
            len(self.bond_types),
            1,
            1,
            len(self.vertex_ranks),
            len(self.stem_arrangements),
            len(self.branch_ranks),
        )

    def pretty_print(self, b: Chem.Bond | None, flip_direction: bool = False) -> str:
        """Get a formatted string representation of the bond features.

        Parameters
        ----------
        b : Chem.Bond or None
            The bond to be described. If None, a null bond is assumed.
        flip_direction : bool, optional
            Whether to reverse the direction of the bond (default is False).

        Returns
        -------
        str
            A formatted string representing the bond features.

        Examples
        --------
        >>> from rdkit import Chem
        >>> from chempropstereo import featurizers
        >>> mol = Chem.MolFromSmiles('CC')
        >>> bond = mol.GetBondWithIdx(0)
        >>> featurizer = featurizers.BondStereoFeaturizer()
        >>> featurizer.pretty_print(bond)
        '    0→1: 0 1000 0 0 0000 00 00'

        """
        atoms = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        if flip_direction:
            atoms.reverse()
        bond_desc = "\u2192".join(map(str, atoms)).rjust(7)
        s = "".join(map(str, self(b, flip_direction)))
        cuts = list(np.cumsum(self.sizes))
        return f"{bond_desc}: " + " ".join(s[a:b] for a, b in zip([0] + cuts, cuts))
