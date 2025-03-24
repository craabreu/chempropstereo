"""Chemprop Stereochemistry Extension.

This module provides functionalities and features for handling stereochemistry
in molecular structures. It includes various featurizers for atoms and bonds,
as well as utilities for describing and tagging stereochemical elements in
molecules.

Key Features:
- Atom and bond featurizers that incorporate stereochemical information.
- Functions for describing stereobonds and stereocenters.
- Utilities for tagging cis/trans stereochemistry and tetrahedral stereocenters.

This extension is designed to enhance the capabilities of the Chemprop framework
by integrating stereochemical considerations into molecular representations.
"""

from ._version import __version__  # noqa: F401
from .featurizers import (
    AtomAchiralFeaturizer,
    AtomStereoFeaturizer,
    BondNeighborRankingFeaturizer,
    BondStereoFeaturizer,
    MoleculeStereoFeaturizer,
)
from .stereochemistry import (
    BranchRank,
    ScanDirection,
    StemArrangement,
    VertexRank,
    describe_stereobond,
    describe_stereocenter,
    get_cip_code,
    set_relative_neighbor_ranking,
    tag_cis_trans_stereobonds,
    tag_tetrahedral_stereocenters,
)

__all__ = [
    "AtomAchiralFeaturizer",
    "AtomStereoFeaturizer",
    "BondNeighborRankingFeaturizer",
    "BondStereoFeaturizer",
    "BranchRank",
    "ScanDirection",
    "StemArrangement",
    "VertexRank",
    "MoleculeStereoFeaturizer",
    "describe_stereobond",
    "describe_stereocenter",
    "get_cip_code",
    "set_relative_neighbor_ranking",
    "tag_cis_trans_stereobonds",
    "tag_tetrahedral_stereocenters",
]
