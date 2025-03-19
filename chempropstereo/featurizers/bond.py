import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry

_BOND_TYPES: tuple[Chem.BondType] = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)


class BondStereoFeaturizer(chemprop.featurizers.base.VectorFeaturizer[Chem.Bond]):
    """
    Multi-hot bond featurizer that includes the position of the end atom in the
    canonical order of neighbors when the begin atom has a canonical chiral tag.

    The featurized bonds are expected to be part of an RDKit molecule with canonical
    chiral tags assigned via :func:`tetrahedral.tag_tetrahedral_stereocenters`.

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
    >>> stereochemistry.tag_steregroups(mol)
    >>> stereochemistry.describe_stereobond(mol.GetBondBetweenAtoms(1, 2))
    'C0 C8 C1 (CIS) C2 C4 O3'
    >>> describe_bonds_from_atom(1) # doctest: +NORMALIZE_WHITESPACE
      1→0: 0 1000 0 0 10000 100 010
      0→1: 0 1000 0 0 10000 100 100
      1→2: 0 0100 1 0 10000 010 100
      2→1: 0 0100 1 0 10000 010 100
      1→8: 0 1000 0 0 10000 100 001
      8→1: 0 1000 0 0 00010 100 100
    >>> describe_bonds_from_atom(2) # doctest: +NORMALIZE_WHITESPACE
      2→1: 0 0100 1 0 10000 010 100
      1→2: 0 0100 1 0 10000 010 100
      2→3: 0 1000 1 0 10000 100 001
      3→2: 0 1000 1 0 10000 100 100
      2→4: 0 1000 1 0 10000 100 010
      4→2: 0 1000 1 0 10000 100 010
    >>> stereochemistry.describe_stereobond(mol.GetBondBetweenAtoms(4, 5))
    'C2 C4 (TRANS) C5 N6 O7'
    >>> describe_bonds_from_atom(4) # doctest: +NORMALIZE_WHITESPACE
      4→2: 0 1000 1 0 10000 100 010
      2→4: 0 1000 1 0 10000 100 010
      4→5: 0 0100 1 0 10000 001 100
      5→4: 0 0100 1 0 10000 001 100
    >>> describe_bonds_from_atom(5) # doctest: +NORMALIZE_WHITESPACE
      5→4: 0 0100 1 0 10000 001 100
      4→5: 0 0100 1 0 10000 001 100
      5→6: 0 1000 1 0 10000 100 010
      6→5: 0 1000 1 0 10000 100 100
      5→7: 0 1000 1 0 10000 100 001
      7→5: 0 1000 1 0 10000 100 100
    >>> stereochemistry.describe_stereocenter(mol.GetAtomWithIdx(8))
    'C8 (CCW) O12 C9 C1'
    >>> describe_bonds_from_atom(8) # doctest: +NORMALIZE_WHITESPACE
      8→1: 0 1000 0 0 00010 100 100
      1→8: 0 1000 0 0 10000 100 001
      8→9: 0 1000 0 0 00100 100 100
      9→8: 0 1000 0 0 00010 100 100
     8→12: 0 1000 0 0 01000 100 100
     12→8: 0 1000 0 0 10000 100 100
    >>> stereochemistry.describe_stereocenter(mol.GetAtomWithIdx(9))
    'C9 (CW) O11 N10 C8'
    >>> describe_bonds_from_atom(9) # doctest: +NORMALIZE_WHITESPACE
      9→8: 0 1000 0 0 00010 100 100
      8→9: 0 1000 0 0 00100 100 100
     9→10: 0 1000 0 0 00100 100 100
     10→9: 0 1000 0 0 10000 100 100
     9→11: 0 1000 0 0 01000 100 100
     11→9: 0 1000 0 0 10000 100 100
    """

    def __len__(self) -> int:
        return sum(self.sizes)

    def __call__(self, b: Chem.Bond | None, flip_direction: bool = False) -> np.ndarray:
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
                *(vertex_rank == item for item in stereochemistry.VertexRank),
                *(arrangement == item for item in stereochemistry.StemArrangement),
                *(branch_rank == item for item in stereochemistry.BranchRank),
            ],
            dtype=int,
        )

    @property
    def sizes(self) -> list[int]:
        """
        Returns a list of sizes corresponding to different bond features.

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
        [1, 4, 1, 1, 5, 3, 3]
        """
        return [
            1,
            len(_BOND_TYPES),
            1,
            1,
            len(stereochemistry.VertexRank),
            len(stereochemistry.StemArrangement),
            len(stereochemistry.BranchRank),
        ]

    def pretty_print(self, b: Chem.Bond | None, flip_direction: bool = False) -> str:
        """
        Returns a formatted string representation of the bond features.

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
        '  0→1: 0 1000 0 0 10000 100 100'
        """
        atoms = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        if flip_direction:
            atoms.reverse()
        bond_desc = "\u2192".join(map(str, atoms)).rjust(5)
        s = "".join(map(str, self(b, flip_direction)))
        cuts = list(np.cumsum(self.sizes))
        return f"{bond_desc}: " + " ".join(s[a:b] for a, b in zip([0] + cuts, cuts))
