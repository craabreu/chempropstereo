import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry
from .atom import AtomCIPFeaturizer, AtomStereoFeaturizer
from .bond import BondStereoFeaturizer


class MoleculeCIPFeaturizer(chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer):
    """
    Molecule featurizer that includes CIP codes for stereocenters.

    Examples
    --------
    >>> from chempropstereo import MoleculeCIPFeaturizer
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> r_mol = Chem.MolFromSmiles("C[C@H](N)O")
    >>> s_mol = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> featurizer = MoleculeCIPFeaturizer()
    >>> r_molgraph = featurizer(r_mol)
    >>> s_molgraph = featurizer(s_mol)
    >>> assert not np.array_equal(r_molgraph.V, s_molgraph.V)
    >>> assert np.array_equal(r_molgraph.E, s_molgraph.E)
    """

    def __init__(self):
        super().__init__(
            atom_featurizer=AtomCIPFeaturizer(),
            bond_featurizer=chemprop.featurizers.MultiHotBondFeaturizer(),
        )

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> chemprop.data.MolGraph:
        mol = Chem.Mol(mol)
        Chem.AssignCIPLabels(mol)
        return super().__call__(mol, atom_features_extra, bond_features_extra)


class MoleculeStereoFeaturizer(chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer):
    """
    Molecule featurizer that includes canonicalized chiral tags for tetrahedral
    stereocenters and the order of bonds stemming from them.

    Examples
    --------
    >>> from chempropstereo import featurizers
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> r_mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)O"))
    >>> s_mol = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](N)O"))
    >>> featurizer = featurizers.MoleculeStereoFeaturizer()
    >>> r_molgraph = featurizer(r_mol)
    >>> s_molgraph = featurizer(s_mol)
    >>> assert not np.array_equal(r_molgraph.V, s_molgraph.V)
    >>> assert np.array_equal(r_molgraph.E, s_molgraph.E)
    """

    def __init__(self):
        super().__init__(
            atom_featurizer=AtomStereoFeaturizer(),
            bond_featurizer=BondStereoFeaturizer(),
        )

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> chemprop.data.MolGraph:
        if not mol.HasProp("hasCanonicalChiralTags"):
            stereochemistry.tag_tetrahedral_stereocenters(mol)

        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as "
                "`len(atom_features_extra)`! "
                f"Got: {n_atoms} and {len(atom_features_extra)}, respectively."
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as "
                "`len(bond_features_extra)`! "
                f"Got: {n_bonds} and {len(bond_features_extra)}, respectively."
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array(
                [self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single
            )
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for j, reverse in enumerate((False, True)):
                x_e = self.bond_featurizer(bond, reverse)
                if bond_features_extra is not None:
                    x_e = np.concatenate(
                        (x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single
                    )
                E[i + j] = x_e
                edge_index[j].extend([v, u] if reverse else [u, v])
            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return chemprop.data.MolGraph(V, E, edge_index, rev_edge_index)
