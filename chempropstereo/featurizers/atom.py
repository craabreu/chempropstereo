import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry


class AtomCIPFeaturizer(chemprop.featurizers.MultiHotAtomFeaturizer):
    """
    Multi-hot atom featurizer that includes a CIP code if the atom is a stereocenter.

    The featurized atoms are expected to be part of an RDKit molecule with CIP labels
    assigned via the `AssignCIPLabels`_ function.

    .. _AssignCIPLabels: https://www.rdkit.org/docs/source/\
rdkit.Chem.rdCIPLabeler.html#rdkit.Chem.rdCIPLabeler.AssignCIPLabels

    Parameters
    ----------
    mode : featurizers.AtomFeatureMode
        The mode to use for the featurizer. Available modes are `V1`_, `V2`_, and
        `ORGANIC`_.

        .. _V1: https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/\
featurizers/atom/index.html#chemprop.featurizers.atom.MultiHotAtomFeaturizer.v1
        .. _V2: https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/\
featurizers/atom/index.html#chemprop.featurizers.atom.MultiHotAtomFeaturizer.v2
        .. _ORGANIC: https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/\
featurizers/atom/index.html#chemprop.featurizers.atom.MultiHotAtomFeaturizer.organic

    Examples
    --------
    >>> from chempropstereo import AtomCIPFeaturizer
    >>> from rdkit import Chem
    >>> r_mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> s_mol = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> for mol in [r_mol, s_mol]:
    ...     Chem.AssignCIPLabels(mol)
    >>> r_atom = r_mol.GetAtomWithIdx(1)
    >>> s_atom = s_mol.GetAtomWithIdx(1)
    >>> for featurizer in [AtomCIPFeaturizer("V2"), AtomCIPFeaturizer("ORGANIC")]:
    ...     for atom in [r_atom, s_atom]:
    ...         print("".join(map(str, featurizer(atom))))
    00000100000000000000000000000000000000000010000001001000100000000100000
    00000100000000000000000000000000000000000010000001000100100000000100000
    0010000000000000010000001001000100000001000
    0010000000000000010000001000100100000001000
    """

    def __init__(self, mode: str | chemprop.featurizers.AtomFeatureMode = "V2") -> None:
        featurizer = chemprop.featurizers.get_multi_hot_atom_featurizer(
            chemprop.featurizers.AtomFeatureMode.get(mode)
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
            stereochemistry.get_cip_code(a),
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


class AtomStereoFeaturizer(chemprop.featurizers.MultiHotAtomFeaturizer):
    """
    Multi-hot atom featurizer that includes a canonical chiral tag for each atom.

    The featurized atoms are expected to be part of an RDKit molecule with canonical
    chiral tags assigned via :func:`~stereochemistry.tag_tetrahedral_stereocenters`.

    Parameters
    ----------
    mode
        The mode to use for the featurizer. Available modes are `V1`_, `V2`_, and
        `ORGANIC`_.

        .. _V1: https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/\
featurizers/atom/index.html#chemprop.featurizers.atom.MultiHotAtomFeaturizer.v1
        .. _V2: https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/\
featurizers/atom/index.html#chemprop.featurizers.atom.MultiHotAtomFeaturizer.v2
        .. _ORGANIC: https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/\
featurizers/atom/index.html#chemprop.featurizers.atom.MultiHotAtomFeaturizer.organic

    Examples
    --------
    >>> from chempropstereo import featurizers, stereochemistry
    >>> from rdkit import Chem
    >>> cw_mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> ccw_mol = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> stereochemistry.tag_tetrahedral_stereocenters(cw_mol)
    >>> stereochemistry.tag_tetrahedral_stereocenters(ccw_mol)
    >>> cw_atom = cw_mol.GetAtomWithIdx(1)
    >>> ccw_atom = ccw_mol.GetAtomWithIdx(1)
    >>> non_chiral_atom = cw_mol.GetAtomWithIdx(0)
    >>> for featurizer in [
    ...     featurizers.AtomStereoFeaturizer("V2"),
    ...     featurizers.AtomStereoFeaturizer("ORGANIC")
    ... ]:
    ...     for atom in [non_chiral_atom, cw_atom, ccw_atom]:
    ...         print("".join(map(str, map(int, featurizer(atom)))))
    0000010000000000000000000000000000000000001000000101000001000000100000
    0000010000000000000000000000000000000000001000000100100100000000100000
    0000010000000000000000000000000000000000001000000100010100000000100000
    001000000000000001000000101000001000001000
    001000000000000001000000100100100000001000
    001000000000000001000000100010100000001000
    """

    def __init__(self, mode: str | chemprop.featurizers.AtomFeatureMode = "V2") -> None:
        featurizer = chemprop.featurizers.get_multi_hot_atom_featurizer(
            chemprop.featurizers.AtomFeatureMode.get(mode)
        )
        super().__init__(
            atomic_nums=featurizer.atomic_nums,
            degrees=featurizer.degrees,
            formal_charges=featurizer.formal_charges,
            chiral_tags=range(2),
            num_Hs=featurizer.num_Hs,
            hybridizations=featurizer.hybridizations,
        )
        self._lower_bound = sum(map(len, self._subfeats[:3]))

    def __call__(self, a: Chem.Atom | None) -> np.ndarray:
        x = np.zeros(len(self))

        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            stereochemistry.get_scan_direction(a),
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
