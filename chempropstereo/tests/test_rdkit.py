from rdkit import Chem


def test_stereoperception_setting():
    """
    Test that the default stereo perception is set to True and can be set to False
    """
    assert Chem.GetUseLegacyStereoPerception() is True
    Chem.SetUseLegacyStereoPerception(False)
    assert Chem.GetUseLegacyStereoPerception() is False
    Chem.SetUseLegacyStereoPerception(True)


def test_find_potential_stereo():
    mol = Chem.MolFromSmiles("C[C@@H]([C@@H](C)N)O")
    info_list = Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=False)
    assert len(info_list) == 2
    c1, c2 = info_list

    assert c1.centeredOn == 1
    assert list(c1.controllingAtoms) == [0, 2, 5]
    assert c1.descriptor == Chem.StereoDescriptor.Tet_CW
    assert c1.specified == Chem.StereoSpecified.Specified
    assert c1.type == Chem.StereoType.Atom_Tetrahedral

    assert c2.centeredOn == 2
    assert list(c2.controllingAtoms) == [1, 3, 4]
    assert c2.descriptor == Chem.StereoDescriptor.Tet_CW
    assert c2.specified == Chem.StereoSpecified.Specified
    assert c2.type == Chem.StereoType.Atom_Tetrahedral

    mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](C)N")
    info_list = Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=False)
    assert len(info_list) == 2
    c1, c2 = info_list

    assert c1.centeredOn == 1
    assert list(c1.controllingAtoms) == [0, 2, 3]
    assert c1.descriptor == Chem.StereoDescriptor.Tet_CCW
    assert c1.specified == Chem.StereoSpecified.Specified
    assert c1.type == Chem.StereoType.Atom_Tetrahedral

    assert c2.centeredOn == 3
    assert list(c2.controllingAtoms) == [1, 4, 5]
    assert c2.descriptor == Chem.StereoDescriptor.Tet_CW
    assert c2.specified == Chem.StereoSpecified.Specified
    assert c2.type == Chem.StereoType.Atom_Tetrahedral
