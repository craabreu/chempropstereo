import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry


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
    >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(mol)
    >>> featurizer = featurizers.BondStereoFeaturizer()
    >>> for bond in mol.GetAtomWithIdx(1).GetBonds():
    ...     one_is_begin = bond.GetBeginAtomIdx() == 1
    ...     for reverse in (not one_is_begin, one_is_begin):
    ...         features = featurizer(bond, reverse)
    ...         assert len(features) == len(featurizer)
    ...         print("".join(map(str, features)))
    010000000010100100
    010000010000100100
    010000000100100100
    010000010000100100
    010000001000100100
    010000010000100100
    """

    def __init__(self) -> None:
        self.bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ]

    def __len__(self) -> int:
        return (
            3  # null bond?, is conjugated?, is in ring?
            + len(self.bond_types)  # bond types
            + len(stereochemistry.VertexRank)  # tetrahedral vertex ranks
            + len(stereochemistry.StemArrangement)  # cis/trans stem arrangements
            + len(stereochemistry.BranchRank)  # cis/trans branch ranks
        )

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
                *(bond_type == item for item in self.bond_types),
                b.GetIsConjugated(),
                b.IsInRing(),
                *(vertex_rank == item for item in stereochemistry.VertexRank),
                *(arrangement == item for item in stereochemistry.StemArrangement),
                *(branch_rank == item for item in stereochemistry.BranchRank),
            ],
            dtype=int,
        )
