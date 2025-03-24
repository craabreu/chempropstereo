"""Atom featurization.

.. module:: featurizers.atom
.. moduleauthor:: Charlles Abreu <craabreu@mit.edu>
"""

import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry
from . import utils

_SCAN_DIRECTIONS: tuple[stereochemistry.ScanDirection, ...] = (
    stereochemistry.ScanDirection.CW,
    stereochemistry.ScanDirection.CCW,
)


class AtomStereoFeaturizer(chemprop.featurizers.base.VectorFeaturizer[Chem.Atom]):
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
    >>> featurizer = featurizers.AtomStereoFeaturizer("ORGANIC")
    >>> for smi in ["C[C@H](N)O", "C[C@@H](N)O"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     stereochemistry.tag_stereogroups(mol)
    ...     print(f"Molecule: {smi}")
    ...     for atom in mol.GetAtoms():
    ...         print(featurizer.pretty_print(atom))
    Molecule: C[C@H](N)O
      0: 0010000000000 0000100 000010 001 000100 00010 0 0.120
      1: 0010000000000 0000100 000010 100 010000 00010 0 0.120
      2: 0001000000000 0001000 000010 001 001000 00010 0 0.140
      3: 0000100000000 0010000 000010 001 010000 00010 0 0.160
    Molecule: C[C@@H](N)O
      0: 0010000000000 0000100 000010 001 000100 00010 0 0.120
      1: 0010000000000 0000100 000010 010 010000 00010 0 0.120
      2: 0001000000000 0001000 000010 001 001000 00010 0 0.140
      3: 0000100000000 0010000 000010 001 010000 00010 0 0.160

    """

    def __init__(self, mode: str | chemprop.featurizers.AtomFeatureMode) -> None:
        featurizer = chemprop.featurizers.get_multi_hot_atom_featurizer(
            chemprop.featurizers.AtomFeatureMode.get(mode)
        )
        self.atomic_nums = featurizer.atomic_nums
        self.degrees = featurizer.degrees
        self.formal_charges = featurizer.formal_charges
        self.scan_directions = _SCAN_DIRECTIONS
        self.num_Hs = featurizer.num_Hs
        self.hybridizations = featurizer.hybridizations
        self._len = sum(self.sizes)

    def __len__(self):
        return self._len

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
                *(scan_direction == item for item in self.scan_directions),
                scan_direction not in self.scan_directions,
                *(total_num_hs == item for item in self.num_Hs),
                total_num_hs not in self.num_Hs,
                *(hybridization == item for item in self.hybridizations),
                hybridization not in self.hybridizations,
                a.GetIsAromatic(),
                0.01 * a.GetMass(),
            ],
            dtype=float,
        )

    @property
    def sizes(self) -> tuple[int, ...]:
        """Get a tuple of sizes corresponding to different atom features.

        The tuple contains the sizes for:
        - Atomic numbers
        - Total degrees
        - Formal charges
        - Scan directions
        - Total numbers of Hs
        - Hybridizations
        - Aromatic indicator
        - Mass

        Returns
        -------
        tuple[int, ...]
            A tuple of integers representing the sizes of each atom feature.

        Examples
        --------
        >>> from chempropstereo import featurizers
        >>> featurizer = featurizers.AtomStereoFeaturizer("ORGANIC")
        >>> featurizer.sizes
        (13, 7, 6, 3, 6, 5, 1, 1)

        """
        return (
            len(self.atomic_nums) + 1,
            len(self.degrees) + 1,
            len(self.formal_charges) + 1,
            len(self.scan_directions) + 1,
            len(self.num_Hs) + 1,
            len(self.hybridizations) + 1,
            1,
            1,
        )

    def pretty_print(self, a: Chem.Atom | None) -> str:
        """Get a formatted string representation of the atom features.

        Parameters
        ----------
        a : Chem.Atom or None
            The atom to be described. If None, a null atom is assumed.

        Returns
        -------
        str
            A formatted string representing the atom features.

        Examples
        --------
        >>> from rdkit import Chem
        >>> from chempropstereo import featurizers
        >>> mol = Chem.MolFromSmiles('CC')
        >>> atom = mol.GetAtomWithIdx(0)
        >>> featurizer = featurizers.AtomStereoFeaturizer("ORGANIC")
        >>> featurizer.pretty_print(atom)
        '  0: 0010000000000 0000100 000010 001 000100 00010 0 0.120'

        """
        return utils.describe_atom_features(a.GetIdx(), self(a), self.sizes)


class AtomAchiralFeaturizer(AtomStereoFeaturizer):
    """Multi-hot atom featurizer that excludes canonical chiral tags.

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
    >>> featurizer = featurizers.AtomAchiralFeaturizer(mode="ORGANIC")
    >>> for smi in ["C[C@H](N)O", "C[C@@H](N)O"]:
    ...     mol = Chem.MolFromSmiles(smi)
    ...     stereochemistry.tag_stereogroups(mol)
    ...     print(f"Molecule: {smi}")
    ...     for atom in mol.GetAtoms():
    ...         print(featurizer.pretty_print(atom))
    Molecule: C[C@H](N)O
      0: 0010000000000 0000100 000010 000100 00010 0 0.120
      1: 0010000000000 0000100 000010 010000 00010 0 0.120
      2: 0001000000000 0001000 000010 001000 00010 0 0.140
      3: 0000100000000 0010000 000010 010000 00010 0 0.160
    Molecule: C[C@@H](N)O
      0: 0010000000000 0000100 000010 000100 00010 0 0.120
      1: 0010000000000 0000100 000010 010000 00010 0 0.120
      2: 0001000000000 0001000 000010 001000 00010 0 0.140
      3: 0000100000000 0010000 000010 010000 00010 0 0.160

    """

    def __init__(
        self,
        mode: str | chemprop.featurizers.AtomFeatureMode,
    ) -> None:
        featurizer = chemprop.featurizers.get_multi_hot_atom_featurizer(
            chemprop.featurizers.AtomFeatureMode.get(mode)
        )
        self.atomic_nums = featurizer.atomic_nums
        self.degrees = featurizer.degrees
        self.formal_charges = featurizer.formal_charges
        self.num_Hs = featurizer.num_Hs
        self.hybridizations = featurizer.hybridizations
        self._len = sum(self.sizes)

    def __len__(self):
        return self._len

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

        return np.array(
            [
                *(atomic_num == item for item in self.atomic_nums),
                atomic_num not in self.atomic_nums,
                *(total_degree == item for item in self.degrees),
                total_degree not in self.degrees,
                *(formal_charge == item for item in self.formal_charges),
                formal_charge not in self.formal_charges,
                *(total_num_hs == item for item in self.num_Hs),
                total_num_hs not in self.num_Hs,
                *(hybridization == item for item in self.hybridizations),
                hybridization not in self.hybridizations,
                a.GetIsAromatic(),
                0.01 * a.GetMass(),
            ],
            dtype=float,
        )

    @property
    def sizes(self) -> tuple[int, ...]:
        """Get a tuple of sizes corresponding to different atom features.

        The tuple contains the sizes for:
        - Atomic numbers
        - Total degrees
        - Formal charges
        - Scan directions
        - Total numbers of Hs
        - Hybridizations
        - Aromatic indicator
        - Mass

        Returns
        -------
        tuple[int, ...]
            A tuple of integers representing the sizes of each atom feature.

        Examples
        --------
        >>> from chempropstereo import featurizers
        >>> featurizer = featurizers.AtomAchiralFeaturizer("ORGANIC")
        >>> featurizer.sizes
        (13, 7, 6, 6, 5, 1, 1)

        """
        return (
            len(self.atomic_nums) + 1,
            len(self.degrees) + 1,
            len(self.formal_charges) + 1,
            len(self.num_Hs) + 1,
            len(self.hybridizations) + 1,
            1,
            1,
        )
