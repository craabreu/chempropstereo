import numpy as np
from chemprop import data, featurizers
from rdkit import Chem

from .atom import AtomFeaturizer
from .bond import BondFeaturizer


class MoleculeFeaturizer(featurizers.SimpleMoleculeMolGraphFeaturizer):
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
