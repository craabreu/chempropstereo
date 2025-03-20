"""Molecule featurization.

.. module:: featurizers.molecule
.. moduleauthor:: Charlles Abreu <craabreu@mit.edu>
"""

import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry
from .atom import AtomCIPFeaturizer, AtomStereoFeaturizer
from .bond import BondStereoFeaturizer


class MoleculeCIPFeaturizer(chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer):
    """Molecule featurizer that includes CIP codes for stereocenters.

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
        """Featurize a molecule with canonical stereochemical information.

        Parameters
        ----------
        mol
            Molecule to be featurized.
        atom_features_extra
            Extra features to be added to the atoms.
        bond_features_extra
            Extra features to be added to the bonds.

        Returns
        -------
        chemprop.data.MolGraph
            Featurized molecule with canonical stereochemical information.

        """
        mol = Chem.Mol(mol)
        Chem.AssignCIPLabels(mol)
        return super().__call__(mol, atom_features_extra, bond_features_extra)


class MoleculeStereoFeaturizer(chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer):
    """Molecule featurizer that includes canonical stereochemical information.

    This featurizer includes canonicalized tetrahedral stereocenters and
    cis/trans stereobonds.

    Parameters
    ----------
    divergent_bonds : bool
        Whether to add stereochemical features to the directed bonds that diverge from
        stereocenters and stereobonds, as opposed to those that converge to them.

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

    def __init__(self, divergent_bonds: bool = True):
        super().__init__(
            atom_featurizer=AtomStereoFeaturizer(),
            bond_featurizer=BondStereoFeaturizer(),
        )
        self.divergent_bonds = divergent_bonds

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> chemprop.data.MolGraph:
        """Featurize a molecule with canonical stereochemical information.

        Parameters
        ----------
        mol
            Molecule to be featurized.
        atom_features_extra
            Extra features to be added to the atoms.
        bond_features_extra
            Extra features to be added to the bonds.

        Returns
        -------
        chemprop.data.MolGraph
            Featurized molecule with canonical stereochemical information.

        """
        stereochemistry.tag_tetrahedral_stereocenters(mol, force=False)
        stereochemistry.tag_cis_trans_stereobonds(mol, force=False)

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
            vertices = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            vertices = np.array(
                [self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single
            )
        edges = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            vertices = np.hstack((vertices, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            u, v = (begin, end) if self.divergent_bonds else (end, begin)
            for j, flip_direction in enumerate((False, True)):
                x_e = self.bond_featurizer(bond, flip_direction)
                if bond_features_extra is not None:
                    x_e = np.concatenate(
                        (x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single
                    )
                edges[i + j] = x_e
                edge_index[j].extend([v, u] if flip_direction else [u, v])
            i += 2

        rev_edge_index = np.arange(len(edges)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return chemprop.data.MolGraph(vertices, edges, edge_index, rev_edge_index)
