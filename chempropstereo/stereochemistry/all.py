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


def add_neighbor_rank_tags(
    mol: Chem.Mol, break_ties: bool = False, force: bool = False
) -> None:
    """Add neighbor-rank tags to the bonds of a molecule.

    Neighbors of each atom are sorted in descending order of their RDKit canonical
    ranks. Then, each bond is given two integer properties:

    - `endRankFromBegin`: the index of its end atom in the sorted list of its begin
      atom's neighbors.
    - `beginRankFromEnd`: the index of its begin atom in the sorted list of its end
      atom's neighbors.

    The molecule is tagged with a boolean property `hasNeighborRanks` to indicate
    that neighbor-rank tags have been added.

    Parameters
    ----------
    mol
        The molecule to add neighbor-rank tags to.
    break_ties
        Whether to break ties in neighbor-rank assignments.
    force
        Whether to add neighbor-rank tags even if they have already been added.

    Examples
    --------
    >>> from chempropstereo import stereochemistry
    >>> from rdkit import Chem
    >>> import numpy as np
    >>> mol = Chem.MolFromSmiles("NC(O)=C(S)C")
    >>> ranks = np.fromiter(Chem.CanonicalRankAtoms(mol, includeChirality=False), int)
    >>> ranked_atoms = map(mol.GetAtomWithIdx, map(int, np.argsort(-ranks)))
    >>> print(" ".join(map(stereochemistry.utils.describe_atom, ranked_atoms)))
    C3 C1 S4 O2 N0 C5
    >>> stereochemistry.add_neighbor_rank_tags(mol)
    >>> for bond in mol.GetBonds():
    ...     print(
    ...         stereochemistry.utils.describe_atom(bond.GetBeginAtom()),
    ...         stereochemistry.utils.describe_atom(bond.GetEndAtom()),
    ...         bond.GetIntProp("endRankFromBegin"),
    ...         bond.GetIntProp("beginRankFromEnd"),
    ...     )
    N0 C1 0 2
    C1 O2 1 0
    C1 C3 0 0
    C3 S4 1 0
    C3 C5 2 0

    """
    if mol.HasProp("hasNeighborRanks") and not force:
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
        ranks = np.searchsorted(
            neighbor_priorities,
            neighbor_priorities,
            sorter=np.argsort(neighbor_priorities),
        )
        sorted_neighbors.append(dict(zip(neighbors, ranks)))
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond.SetIntProp("endRankFromBegin", int(sorted_neighbors[begin][end]))
        bond.SetIntProp("beginRankFromEnd", int(sorted_neighbors[end][begin]))
    mol.SetBoolProp("hasNeighborRanks", True)
