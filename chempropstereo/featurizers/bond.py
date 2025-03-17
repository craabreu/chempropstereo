import chemprop
import numpy as np
from rdkit.Chem.rdchem import Bond

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
    >>> neighbors = stereochemistry.get_neighbors(
    ...     mol.GetAtomWithIdx(1)
    ... )
    >>> for neighbor in neighbors:
    ...     bond = mol.GetBondBetweenAtoms(1, neighbor)
    ...     one_is_begin = bond.GetBeginAtomIdx() == 1
    ...     for reverse in (not one_is_begin, one_is_begin):
    ...         print("".join(map(str, featurizer(bond, reverse))))
    0100000100000010000
    0100000100000001000
    0100000100000010000
    0100000100000000100
    0100000100000010000
    0100000100000000010
    """

    def __len__(self):
        return super().__len__() + 5

    def __call__(self, b: Bond | None, reverse: bool = False) -> np.ndarray:
        if reverse:
            begin_atom = b.GetBeginAtom()
            end_index = b.GetEndAtomIdx()
        else:
            begin_atom = b.GetEndAtom()
            end_index = b.GetBeginAtomIdx()
        neighbors = stereochemistry.get_neighbors(begin_atom)
        x = super().__call__(b)
        x[-4 + neighbors.index(end_index) if neighbors else -5] = 1
        return x
