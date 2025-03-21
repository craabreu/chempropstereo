"""Molecule featurization.

.. module:: featurizers.molecule
.. moduleauthor:: Charlles Abreu <craabreu@mit.edu>
"""

import chemprop
import numpy as np
from rdkit import Chem

from .. import stereochemistry
from . import utils
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
    r"""Molecule featurizer that includes canonical stereochemical information.

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
    >>> mol = Chem.MolFromSmiles("C[C@@H](N)/C=C(O)/N")
    >>> for divergent in (True, False):
    ...     print(f"\nWith {'di' if divergent else 'con'}vergent bonds:\n")
    ...     featurizer = featurizers.MoleculeStereoFeaturizer(
    ...         mode="ORGANIC",
    ...         divergent_bonds=divergent,
    ...     )
    ...     print(featurizer.pretty_print(mol))
    <BLANKLINE>
    With divergent bonds:
    <BLANKLINE>
    Vertices:
      0: 0010000000000 0000100 000010 001 000100 00010 0 0.120
      1: 0010000000000 0000100 000010 010 010000 00010 0 0.120
      2: 0001000000000 0001000 000010 001 001000 00010 0 0.140
      3: 0010000000000 0001000 000010 001 010000 00100 0 0.120
      4: 0010000000000 0001000 000010 001 100000 00100 0 0.120
      5: 0000100000000 0010000 000010 001 010000 00100 0 0.160
      6: 0001000000000 0001000 000010 001 001000 00100 0 0.140
    Edges:
        0→1: 0 1000 0 0 0000 00 00
        1→0: 0 1000 0 0 0100 00 00
        1→2: 0 1000 0 0 1000 00 00
        2→1: 0 1000 0 0 0000 00 00
        1→3: 0 1000 0 0 0010 00 00
        3→1: 0 1000 0 0 0000 00 10
        3→4: 0 0100 1 0 0000 01 00
        4→3: 0 0100 1 0 0000 01 00
        4→5: 0 1000 1 0 0000 00 01
        5→4: 0 1000 1 0 0000 00 00
        4→6: 0 1000 1 0 0000 00 10
        6→4: 0 1000 1 0 0000 00 00
    <BLANKLINE>
    With convergent bonds:
    <BLANKLINE>
    Vertices:
      0: 0010000000000 0000100 000010 001 000100 00010 0 0.120
      1: 0010000000000 0000100 000010 010 010000 00010 0 0.120
      2: 0001000000000 0001000 000010 001 001000 00010 0 0.140
      3: 0010000000000 0001000 000010 001 010000 00100 0 0.120
      4: 0010000000000 0001000 000010 001 100000 00100 0 0.120
      5: 0000100000000 0010000 000010 001 010000 00100 0 0.160
      6: 0001000000000 0001000 000010 001 001000 00100 0 0.140
    Edges:
        1→0: 0 1000 0 0 0000 00 00
        0→1: 0 1000 0 0 0100 00 00
        2→1: 0 1000 0 0 1000 00 00
        1→2: 0 1000 0 0 0000 00 00
        3→1: 0 1000 0 0 0010 00 00
        1→3: 0 1000 0 0 0000 00 10
        4→3: 0 0100 1 0 0000 01 00
        3→4: 0 0100 1 0 0000 01 00
        5→4: 0 1000 1 0 0000 00 01
        4→5: 0 1000 1 0 0000 00 00
        6→4: 0 1000 1 0 0000 00 10
        4→6: 0 1000 1 0 0000 00 00

    """

    def __init__(
        self,
        mode: str | chemprop.featurizers.AtomFeatureMode,
        divergent_bonds: bool,
    ) -> None:
        super().__init__(
            atom_featurizer=AtomStereoFeaturizer(mode),
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
        stereochemistry.tag_stereogroups(mol, force=False)

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

    def pretty_print(self, mol: Chem.Mol) -> None:
        """Print a formatted string representation of the featurized molecule.

        Parameters
        ----------
        mol
            The molecule to be featurized.

        Returns
        -------
        str
            A string with the following format:
            .. code-block:: text

                Vertices:
                <atom1 features>
                <atom2 features>
                ...
                Edges:
                <bond1 features>
                <bond2 features>
                ...

            The features for each atom and bond are described in terms of the
            one-hot encodings of the properties and the floats for the masses.

        Example
        -------
        >>> from rdkit import Chem
        >>> from chempropstereo import MoleculeStereoFeaturizer
        >>> mol = Chem.MolFromSmiles("C[C@H](N)O")
        >>> featurizer = MoleculeStereoFeaturizer("ORGANIC", divergent_bonds=True)
        >>> print(featurizer.pretty_print(mol))  # doctest: +NORMALIZE_WHITESPACE
        Vertices:
        0: 0010000000000 0000100 000010 001 000100 00010 0 0.120
        1: 0010000000000 0000100 000010 100 010000 00010 0 0.120
        2: 0001000000000 0001000 000010 001 001000 00010 0 0.140
        3: 0000100000000 0010000 000010 001 010000 00010 0 0.160
        Edges:
            0→1: 0 1000 0 0 0000 00 00
            1→0: 0 1000 0 0 0010 00 00
            1→2: 0 1000 0 0 0100 00 00
            2→1: 0 1000 0 0 0000 00 00
            1→3: 0 1000 0 0 1000 00 00
            3→1: 0 1000 0 0 0000 00 00

        """
        molgraph = self(mol)
        vertices = "\n".join(
            utils.describe_atom_features(vertex, features, self.atom_featurizer.sizes)
            for vertex, features in enumerate(molgraph.V)
        )
        edges = "\n".join(
            utils.describe_bond_features(edge, features, self.bond_featurizer.sizes)
            for edge, features in zip(molgraph.edge_index.T, molgraph.E)
        )
        return f"Vertices:\n{vertices}\nEdges:\n{edges}"
