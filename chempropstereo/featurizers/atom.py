"""Atom featurization.

.. module:: featurizers.atom
.. moduleauthor:: Charlles Abreu <craabreu@mit.edu>
"""

import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry


class AtomCIPFeaturizer(chemprop.featurizers.MultiHotAtomFeaturizer):
    """Multi-hot atom featurizer that includes a CIP code if the atom is a stereocenter.

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
    ...         features = featurizer(atom)
    ...         assert len(features) == len(featurizer)
    ...         print("".join(map(str, features)))
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
        """Featurize an RDKit atom with stereochemical information.

        Parameters
        ----------
        a : Chem.Atom | None
            The atom to featurize. If None, returns a zero array.

        Returns
        -------
        np.ndarray
            A 1D array of shape `(len(self),)` containing the following features:
            - One-hot encoding of the atomic number
            - One-hot encoding of the total degree
            - One-hot encoding of the formal charge
            - One-hot encoding of the CIP code
            - One-hot encoding of the total number of hydrogens
            - One-hot encoding of the hybridization
            - Boolean indicating whether the atom is aromatic
            - Mass of the atom divided by 100

        """
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
    """Multi-hot atom featurizer that includes a canonical chiral tag for each atom.

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
    ...     featurizers.AtomStereoFeaturizer("ORGANIC"),
    ... ]:
    ...     for atom in [non_chiral_atom, cw_atom, ccw_atom]:
    ...         features = featurizer(atom)
    ...         assert len(features) == len(featurizer)
    ...         print("".join(map(str, map(int, features))))
    0000010000000000000000000000000000000000001000000101000001000000100000
    0000010000000000000000000000000000000000001000000100100100000000100000
    0000010000000000000000000000000000000000001000000100010100000000100000
    001000000000000001000000101000001000001000
    001000000000000001000000100100100000001000
    001000000000000001000000100010100000001000

    """

    def __init__(self, mode: str | chemprop.featurizers.AtomFeatureMode) -> None:
        featurizer = chemprop.featurizers.get_multi_hot_atom_featurizer(
            chemprop.featurizers.AtomFeatureMode.get(mode)
        )
        super().__init__(
            atomic_nums=featurizer.atomic_nums,
            degrees=featurizer.degrees,
            formal_charges=featurizer.formal_charges,
            chiral_tags=[],
            num_Hs=featurizer.num_Hs,
            hybridizations=featurizer.hybridizations,
        )

    def __len__(self):
        return (
            super().__len__()  # Original subfeatures
            - 1  # minus chiral tag padding
            + len(stereochemistry.ScanDirection)  # plus scan direction slots
        )

    def __call__(self, a: Chem.Atom | None) -> np.ndarray:
        """Featurize an RDKit atom with stereochemical information.

        Parameters
        ----------
        a
            The atom to featurize.

        Returns
        -------
        np.ndarray
            A 1D array of shape `(len(self),)` containing the following features:
            - `atomic_num`: one-hot encoding of the atomic number
            - `total_degree`: one-hot encoding of the total degree
            - `formal_charge`: one-hot encoding of the formal charge
            - `scan_direction`: one-hot encoding of the scan direction
            - `total_num_hs`: one-hot encoding of the total number of Hs
            - `hybridization`: one-hot encoding of the hybridization
            - `is_aromatic`: boolean indicating whether the atom is aromatic
            - `mass`: mass of the atom divided by 100

        """
        if a is None:
            return np.zeros(len(self))

        atomic_num = a.GetAtomicNum()
        total_degree = a.GetTotalDegree()
        formal_charge = a.GetFormalCharge()
        total_num_hs = a.GetTotalNumHs()
        hybridization = a.GetHybridization()
        scan_direction = stereochemistry.ScanDirection.get_from(a)

        return np.array(
            [
                *(atomic_num == item for item in self.atomic_nums),
                atomic_num not in self.atomic_nums,
                *(total_degree == item for item in self.degrees),
                total_degree not in self.degrees,
                *(formal_charge == item for item in self.formal_charges),
                formal_charge not in self.formal_charges,
                *(scan_direction == item for item in stereochemistry.ScanDirection),
                *(total_num_hs == item for item in self.num_Hs),
                total_num_hs not in self.num_Hs,
                *(hybridization == item for item in self.hybridizations),
                hybridization not in self.hybridizations,
                a.GetIsAromatic(),
                0.01 * a.GetMass(),
            ],
            dtype=float,
        )
