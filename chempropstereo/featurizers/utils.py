from rdkit import Chem


def get_cip_code(atom: Chem.Atom) -> str | None:
    return atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
