import chemprop
import numpy as np
from rdkit.Chem.rdchem import Bond

# from chemprop.featurizers.base import VectorFeaturizer


class BondStereoFeaturizer(chemprop.featurizers.MultiHotBondFeaturizer):
    def __len__(self):
        return super().__len__()

    def __call__(self, b: Bond | None) -> np.ndarray:
        x = np.zeros(len(self), int)

        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1

        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2

        stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
        x[i + stereo_bit] = 1

        return x
