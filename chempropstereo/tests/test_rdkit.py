import numpy as np
from rdkit import Chem


def test_chiral_center_detection():
    def check_molecule(smiles, descriptions, use_legacy):
        Chem.SetUseLegacyStereoPerception(use_legacy)
        mol = Chem.MolFromSmiles(smiles)
        info_list = Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=False)
        assert len(info_list) == len(descriptions)
        for center, desc in zip(info_list, descriptions):
            assert center.type == Chem.StereoType.Atom_Tetrahedral
            assert center.specified == Chem.StereoSpecified.Specified
            atom = mol.GetAtomWithIdx(center.centeredOn)
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            assert atom.GetChiralTag() == getattr(
                Chem.ChiralType, f"CHI_TETRAHEDRAL_{desc}"
            )
            assert center.descriptor == getattr(Chem.StereoDescriptor, f"Tet_{desc}")
            assert all(i == j for i, j in zip(center.controllingAtoms, neighbors))
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in [center.centeredOn for center in info_list]:
                assert atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    original_use_legacy = Chem.GetUseLegacyStereoPerception()
    for use_legacy in [True, False]:
        check_molecule("C[C@@H](O)N", ["CW"], use_legacy)
        check_molecule("C[C@H](O)N", ["CCW"], use_legacy)
        check_molecule("C[C@H](N)O", ["CCW"], use_legacy)
        check_molecule("C[C@@H](O)[C@@H](C)N", ["CW", "CW"], use_legacy)
        check_molecule("C[C@@H](O)[C@H](C)N", ["CW", "CCW"], use_legacy)
    Chem.SetUseLegacyStereoPerception(original_use_legacy)


def test_canonical_ranking():
    mol = Chem.MolFromSmiles("C[C@@H]([C@@H](C)N)O")
    order = np.argsort(Chem.CanonicalRankAtoms(mol)).tolist()
    assert order == [0, 3, 4, 5, 1, 2]
    mol = Chem.AddHs(mol)
    order = np.argsort(Chem.CanonicalRankAtoms(mol)).tolist()
    assert order == [16, 14, 15, 6, 7, 8, 11, 12, 13, 9, 10, 5, 4, 0, 3, 1, 2]
    ranked_elements = "".join([mol.GetAtomWithIdx(i).GetSymbol() for i in order])
    assert ranked_elements == "HHHHHHHHHHHONCCCC"  # Hydrogens first
