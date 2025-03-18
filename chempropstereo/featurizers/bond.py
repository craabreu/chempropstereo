import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry


class BondStereoFeaturizer(chemprop.featurizers.MultiHotBondFeaturizer):
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
        super().__init__(
            bond_types=[
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ],
            stereos=[
                Chem.BondStereo.STEREONONE,
                Chem.BondStereo.STEREOANY,
                Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOE,
                Chem.BondStereo.STEREOCIS,
                Chem.BondStereo.STEREOTRANS,
            ],
        )

    def __len__(self) -> int:
        return super().__len__() + len(stereochemistry.VertexRank)

    def __call__(self, b: Chem.Bond | None, reverse: bool = False) -> np.ndarray:
        x = super().__call__(b)
        vertex_rank = stereochemistry.VertexRank.get_from(b, reverse)
        x[-len(stereochemistry.VertexRank) + vertex_rank] = 1
        return x
