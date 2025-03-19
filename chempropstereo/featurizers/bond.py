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
    ...         print("".join(map(str, featurizer(bond, reverse))))
    0100000100000000010
    0100000100000010000
    0100000100000000100
    0100000100000010000
    0100000100000001000
    0100000100000010000
    """

    def __init__(self) -> None:
        self.bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ]
        self.stereo = [
            Chem.BondStereo.STEREONONE,
            Chem.BondStereo.STEREOANY,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOCIS,
            Chem.BondStereo.STEREOTRANS,
        ]

    def __len__(self) -> int:
        return (
            1
            + len(self.bond_types)
            + 2
            + (len(self.stereo) + 1)
            + len(stereochemistry.VertexRank)
        )

    def __call__(self, b: Chem.Bond | None, flip_direction: bool = False) -> np.ndarray:
        if b is None:
            x = np.zeros(len(self), int)
            x[0] = 1
            return x

        bond_type = b.GetBondType()
        stereo_type = b.GetStereo()
        vertex_rank = stereochemistry.VertexRank.from_bond(b, flip_direction)

        return np.array(
            [
                b is None,
                *(bond_type == item for item in self.bond_types),
                b.GetIsConjugated(),
                b.IsInRing(),
                *(stereo_type == item for item in self.stereo),
                stereo_type not in self.stereo,
                *(vertex_rank == item for item in stereochemistry.VertexRank),
            ],
            dtype=int,
        )
