import numpy as np
from chemprop import featurizers
from rdkit import Chem

from . import utils


class AtomFeaturizer(featurizers.MultiHotAtomFeaturizer):
    """
    Multi-hot atom featurizer that includes CIP codes for stereocenters.

    Parameters
    ----------
    mode : featurizers.AtomFeatureMode
        The mode to use for the featurizer. Available modes are `V1`, `V2`, and
        `ORGANIC`.

    Examples
    --------
    >>> from chempropstereo.featurizers.atom import AtomFeaturizer
    >>> r_atom = Chem.MolFromSmiles("C[C@H](N)O").GetAtomWithIdx(1)
    >>> s_atom = Chem.MolFromSmiles("C[C@@H](N)O").GetAtomWithIdx(1)
    >>> v2_featurizer = AtomFeaturizer()
    >>> for featurizer in [AtomFeaturizer("V2"), AtomFeaturizer("ORGANIC")]:
    ...     for atom in [r_atom, s_atom]:
    ...         print("".join(map(str, featurizer(atom))))
    00000100000000000000000000000000000000000010000001001000100000000100000
    00000100000000000000000000000000000000000010000001000100100000000100000
    0010000000000000010000001001000100000001000
    0010000000000000010000001000100100000001000
    """

    def __init__(self, mode: str | featurizers.AtomFeatureMode = "V2") -> None:
        featurizer = featurizers.get_multi_hot_atom_featurizer(
            featurizers.AtomFeatureMode.get(mode)
        )
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
            utils.get_cip_code(a),
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
