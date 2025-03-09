from rdkit import Chem

_CIP_CODE_MAP = {False: 0, "R": 1, "S": 2}


def get_cip_code(atom: Chem.Atom) -> int:
    """
    Get the CIP code of an atom.

    Parameters
    ----------
    atom : Chem.Atom
        The atom to get the CIP code of.

    Returns
    -------
    int
        The CIP code of the atom.
    """
    return _CIP_CODE_MAP[atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode")]
