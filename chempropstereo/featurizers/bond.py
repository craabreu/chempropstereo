import chemprop
import numpy as np
from rdkit.Chem.rdchem import Bond

from . import utils


class BondStereoFeaturizer(chemprop.featurizers.MultiHotBondFeaturizer):
    def __len__(self):
        return super().__len__() + 5

    def __call__(self, b: Bond | None, reverse: bool = False) -> np.ndarray:
        if reverse:
            begin_atom = b.GetBeginAtom()
            end_index = b.GetEndAtomIdx()
        else:
            begin_atom = b.GetEndAtom()
            end_index = b.GetBeginAtomIdx()
        neighbors = utils.get_neighbors_in_canonical_order(begin_atom, numeric=True)
        x = super().__call__(b)
        x[-4 + neighbors.index(end_index) if neighbors else -5] = 1
        return x
