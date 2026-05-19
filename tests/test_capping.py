import numpy as np
import pytest
from pymatgen.core import Lattice

from porran.capping import cap_with_H2O, cap_with_OH


def test_cap_with_oh_uses_lattice_aware_direction() -> None:
    lattice = Lattice.orthorhombic(10, 20, 30)
    sites = [
        ("M", np.array([0.1, 0.1, 0.1])),
        ("O", np.array([0.2, 0.2, 0.1])),
    ]

    capped = cap_with_OH(lattice, sites, 0, 1)

    open_cart = lattice.get_cartesian_coords(sites[0][1])
    expected = lattice.get_cartesian_coords(sites[1][1] - sites[0][1])
    expected /= np.linalg.norm(expected)

    oxygen_cart = capped[1][1]
    actual = oxygen_cart - open_cart
    actual /= np.linalg.norm(actual)

    assert np.allclose(actual, expected)


def test_cap_with_h2o_returns_finite_coordinates() -> None:
    lattice = Lattice.orthorhombic(10, 20, 30)
    sites = [
        ("M", np.array([0.1, 0.1, 0.1])),
        ("O", np.array([0.2, 0.2, 0.15])),
    ]

    capped = cap_with_H2O(lattice, sites, 0, 1)
    coords = np.array([coord for _, coord in capped])

    assert np.isfinite(coords).all()


def test_capping_rejects_collapsed_bond_direction() -> None:
    lattice = Lattice.cubic(10)
    sites = [
        ("M", np.array([0.1, 0.1, 0.1])),
        ("O", np.array([0.1, 0.1, 0.1])),
    ]

    with pytest.raises(ValueError, match="must not coincide"):
        cap_with_OH(lattice, sites, 0, 1)