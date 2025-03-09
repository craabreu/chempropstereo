from rdkit import Chem

_CIP_CODE_MAP = {False: 0, "R": 1, "S": 2}


def get_cip_code(atom: Chem.Atom) -> int:
    """
    Get the CIP code of an atom as an integer.

    Parameters
    ----------
    atom : Chem.Atom
        The atom to get the CIP code of.

    Returns
    -------
    int
        The CIP code of the atom: 0 for no CIP code, 1 for R, 2 for S.

    Examples
    --------
    >>> from chempropstereo.featurizers.utils import get_cip_code
    >>> from rdkit import Chem
    >>> mol1 = Chem.MolFromSmiles("C[C@H](N)O")
    >>> [get_cip_code(atom) for atom in mol1.GetAtoms()]
    [0, 1, 0, 0]
    >>> mol2 = Chem.MolFromSmiles("C[C@@H](N)O")
    >>> [get_cip_code(atom) for atom in mol2.GetAtoms()]
    [0, 2, 0, 0]
    """
    return _CIP_CODE_MAP[atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode")]
