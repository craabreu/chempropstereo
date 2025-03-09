import numpy as np
from chemprop import featurizers
from rdkit import Chem

from .utils import get_cip_code


class AtomFeaturizer(featurizers.MultiHotAtomFeaturizer):
    def __init__(
        self, mode: featurizers.AtomFeatureMode = featurizers.AtomFeatureMode.V2
    ) -> None:
        featurizer = featurizers.get_multi_hot_atom_featurizer(mode)
        super().__init__(
            atomic_nums=featurizer.atomic_nums,
            degrees=featurizer.degrees,
            formal_charges=featurizer.formal_charges,
            chiral_tags=list(range(3)),
            num_Hs=featurizer.num_Hs,
            hybridizations=featurizer.hybridizations,
        )

    def __call__(self, a: Chem.Atom | None) -> np.ndarray:
        x = np.zeros(len(self), int)

        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            get_cip_code(a),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]

        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        return x
