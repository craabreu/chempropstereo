import numpy as np
from chemprop import data, featurizers
from rdkit import Chem

from .atom import AtomFeaturizer
from .bond import BondFeaturizer


class MoleculeFeaturizer(featurizers.SimpleMoleculeMolGraphFeaturizer):
    """
    Molecule featurizer that includes CIP codes for stereocenters.

    Examples
    --------
    >>> from chempropstereo.featurizers.molecule import MoleculeFeaturizer
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> r_mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> s_mol = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> featurizer = MoleculeFeaturizer()
    >>> r_molgraph = featurizer(r_mol)
    >>> s_molgraph = featurizer(s_mol)
    >>> assert not np.array_equal(r_molgraph.V, s_molgraph.V)
    >>> assert np.array_equal(r_molgraph.E, s_molgraph.E)
    """

    def __init__(self):
        super().__init__(
            atom_featurizer=AtomFeaturizer(),
            bond_featurizer=BondFeaturizer(),
        )

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> data.MolGraph:
        mol = Chem.Mol(mol)
        Chem.AssignCIPLabels(mol)
        return super().__call__(mol, atom_features_extra, bond_features_extra)
