"""Module for tagging stereogenic groups in molecules.

.. module:: stereochemistry.all
.. moduleauthor:: Charlles Abreu <craabreu@mit.edu>
"""

import numpy as np
from rdkit import Chem

from .cistrans import tag_cis_trans_stereobonds
from .tetrahedral import tag_tetrahedral_stereocenters


def tag_stereogroups(mol: Chem.Mol, force: bool = False) -> None:
    r"""Add canonical stereochemistry information to stereogenic groups in a molecule.

    The currently supported stereogenic groups are:

    - tetrahedral stereocenters
    - cis/trans stereobonds

    Parameters
    ----------
    mol
        The molecule whose stereogenic groups are to be tagged.
    force
        Whether to overwrite existing chiral tags (default is False).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from chempropstereo import stereochemistry
    >>> mol = Chem.MolFromSmiles("C\C(=C(O)/C=C(/N)O)[C@@H]([C@H](N)O)O")
    >>> stereochemistry.tag_stereogroups(mol)
    >>> for atom in mol.GetAtoms():
    ...     direction = stereochemistry.ScanDirection.get_from(atom)
    ...     if direction != stereochemistry.ScanDirection.NONE:
    ...         print(stereochemistry.describe_stereocenter(atom))
    C8 (CCW) O12 C9 C1
    C9 (CW) O11 N10 C8
    >>> for bond in mol.GetBonds():
    ...     arrangement = stereochemistry.StemArrangement.get_from(bond)
    ...     if arrangement != stereochemistry.StemArrangement.NONE:
    ...         print(stereochemistry.describe_stereobond(bond))
    C0 C8 C1 (CIS) C2 C4 O3
    C2 C4 (TRANS) C5 N6 O7

    """
    tag_tetrahedral_stereocenters(mol, force)
    tag_cis_trans_stereobonds(mol, force)


def set_relative_neighbor_ranking(
    mol: Chem.Mol, break_ties: bool = False, force: bool = False
) -> None:
    r"""Add neighbor ranking information to the bonds of a molecule.

    Neighbors of each atom are sorted in descending order based on their RDKit canonical
    ranks (see :rdmolfiles:`CanonicalRankAtoms`). This sorting keeps hydrogens at the
    end of the list.

    For each bond in the molecule, two integer properties are added:

    - `endRankFromBegin`: The relative rank of the bond's end atom with respect to all
        neighbors of the bond's begin atom.
    - `beginRankFromEnd`: The relative rank of the bond's begin atom with respect to all
        neighbors of the bond's end atom.

    A relative rank indicates the position of an atom in the sorted list of another
    atom's neighbors. For example, if an atom has three neighbors with RDKit ranks
    `(3, 1, 7)`, their relative ranks will be `(1, 2, 0)`.

    By default, rank assignments can result in ties. For instance, if the neighbors
    have RDKit ranks `(3, 1, 1)`, their resulting relative ranks will be `(1, 0, 0)`.

    If the optional parameter `break_ties` is set to `True`, ties will be resolved by
    considering the arbitrary order of the atoms in the molecule.

    The molecule is tagged with a boolean property `neighborRankTiesAreBroken` to
    indicate whether ties have been broken in neighbor-rank assignments.

    Parameters
    ----------
    mol
        The molecule to add neighbor ranking information to.
    break_ties
        Whether to break ties in neighbor rank assignments (default is False).
    force
        Whether to add neighbor ranking information even if it has already been added
        with the same tie-breaking choice (default is False).

    Examples
    --------
    >>> from chempropstereo import stereochemistry
    >>> from chempropstereo.stereochemistry.utils import describe_atom
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> mol = Chem.MolFromSmiles("NC(O)=C(O)O")
    >>> for break_ties in [False, True]:
    ...     stereochemistry.set_relative_neighbor_ranking(mol, break_ties, force=True)
    ...     print("\nBreak ties:", ["No", "Yes"][break_ties])
    ...     ranks = {describe_atom(atom): [] for atom in mol.GetAtoms()}
    ...     for bond in mol.GetBonds():
    ...         begin = describe_atom(bond.GetBeginAtom())
    ...         end = describe_atom(bond.GetEndAtom())
    ...         ranks[begin].append((end, bond.GetIntProp("endRankFromBegin")))
    ...         ranks[end].append((begin, bond.GetIntProp("beginRankFromEnd")))
    ...     for atom in ranks:
    ...         print(atom, *sorted(ranks[atom], key=lambda x: x[1]))
    <BLANKLINE>
    Break ties: No
    N0 ('C1', 0)
    C1 ('C3', 0) ('O2', 1) ('N0', 2)
    O2 ('C1', 0)
    C3 ('C1', 0) ('O4', 1) ('O5', 1)
    O4 ('C3', 0)
    O5 ('C3', 0)
    <BLANKLINE>
    Break ties: Yes
    N0 ('C1', 0)
    C1 ('C3', 0) ('O2', 1) ('N0', 2)
    O2 ('C1', 0)
    C3 ('C1', 0) ('O5', 1) ('O4', 2)
    O4 ('C3', 0)
    O5 ('C3', 0)

    """
    if (
        mol.HasProp("neighborRankTiesAreBroken")
        and mol.GetBoolProp("neighborRankTiesAreBroken") == break_ties
        and not force
    ):
        return
    # Negative ranks from Chem.CanonicalRankAtoms to ensure hydrogen atoms are last
    all_priorities = -np.fromiter(
        Chem.CanonicalRankAtoms(
            mol, breakTies=break_ties, includeChirality=False, includeAtomMaps=False
        ),
        dtype=int,
    )
    sorted_neighbors = []
    for atom in mol.GetAtoms():
        neighbors = np.fromiter(
            (atom.GetIdx() for atom in atom.GetNeighbors()), dtype=int
        )
        neighbor_priorities = all_priorities[neighbors]
        ranks = np.searchsorted(np.unique(neighbor_priorities), neighbor_priorities)
        sorted_neighbors.append(dict(zip(neighbors, ranks)))
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endRankFromBegin", int(sorted_neighbors[begin][end]))
        bond.SetIntProp("beginRankFromEnd", int(sorted_neighbors[end][begin]))
    mol.SetBoolProp("neighborRankTiesAreBroken", break_ties)
