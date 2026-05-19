from pathlib import Path
import sys

import numpy as np
import pytest
from pymatgen.io.cif import CifParser


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root for data-backed integration tests."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def cau10_structure(repo_root: Path):
    """Return the CAU-10 structure used in defect-generation regression tests."""
    return CifParser(
        str(repo_root / "CAU10H_hydr.cif"),
        check_cif=False,
        site_tolerance=1e-3,
    ).parse_structures(primitive=False)[0]


@pytest.fixture(scope="session")
def cau10_full_mask(cau10_structure):
    """Return a mask that keeps all CAU-10 sites available for replacement."""
    return np.ones(len(cau10_structure), dtype=bool)