"""Geometry helpers for adding charge-balancing cap groups to open metal sites."""

from pymatgen.core.structure import Lattice
import numpy as np
from typing import List, Tuple

BOND_LEN_MO = 1.8
BOND_LEN_OH = 0.96
HOH_ANGLE = 104.5
CAP_CHARGE = {"OH": -1.0, "H2O": 0.0} 




# TODO: Generalize capping functions for different metals and cap groups
def _bond_direction(
    lattice: Lattice,
    sites: List[Tuple[str, np.ndarray]],
    open_ind: int,
    bonded_ind: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cartesian metal position and unit vector toward the removed linker atom.

    Parameters
    ----------
    lattice : Lattice
        Lattice of the parent MOF structure.
    sites : List[Tuple[str, np.ndarray]]
        Atom symbols with fractional coordinates.
    open_ind : int
        Index of the open metal site.
    bonded_ind : int
        Index of the removed linker atom that was bonded to the metal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Metal cartesian coordinates and the unit cartesian bond direction.

    Raises
    ------
    ValueError
        If the bonded and open sites collapse to the same periodic position.
    """
    bonded_frac = np.asarray(sites[bonded_ind][1], dtype=float)
    open_frac = np.asarray(sites[open_ind][1], dtype=float)

    dfrac = bonded_frac - open_frac
    dfrac -= np.round(dfrac)

    vec_bo = lattice.get_cartesian_coords(dfrac)
    norm = np.linalg.norm(vec_bo)
    if norm == 0:
        raise ValueError("Bonded and open sites must not coincide")

    open_cart = lattice.get_cartesian_coords(open_frac)
    return open_cart, vec_bo / norm


def cap_with_OH(lattice: Lattice, sites: List[Tuple[str, np.ndarray]], open_ind: int, bonded_ind: int):
    """
    Caps a metal center with open coordination sites with OH.
    Parameters
    ----------
    lattice: Lattice
        Lattice of the MOF structure without defects
    sites: List[Tuple[str, np.ndarray]]
        List of tuples containing atom types and their fractional coordinates
    open_ind: int
        Index of the metal atom with open coordination site
    bonded_ind: int
        Index of the atom in the linker to be removed which was bonded to the metal
    """

    open_cart, vec_bo = _bond_direction(lattice, sites, open_ind, bonded_ind)

    O_coords = open_cart + BOND_LEN_MO * vec_bo
    h_coords = O_coords + BOND_LEN_OH * vec_bo

    return [('H', h_coords), ('O', O_coords)]


def cap_with_H2O(lattice: Lattice, sites: List[Tuple[str, np.ndarray]], open_ind: int, bonded_ind: int):
    """
    Caps a metal center with an H2O group.
    Parameters
    ----------
    lattice: Lattice
        Lattice of the MOF structure without defects
    sites: List[Tuple[str, np.ndarray]]
        List of tuples containing atom types and their fractional coordinates
    open_ind: int
        Index of the metal atom with open coordination site
    bonded_ind: int
        Index of the atom in the linker to be removed which was bonded to the metal
    """
    open_cart, vec_bo = _bond_direction(lattice, sites, open_ind, bonded_ind)

    O_coords = open_cart + BOND_LEN_MO * vec_bo


    if abs(vec_bo[0]) < 0.9:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = np.array([0.0, 1.0, 0.0])
    # Make it perpendicular
    v_perp = perp - np.dot(perp, vec_bo) * vec_bo
    v_perp /= np.linalg.norm(v_perp)

    # Rotate perpendicular vector to set HOH angle
    angle_rad = np.radians(HOH_ANGLE / 2)
    h1_coords = O_coords + BOND_LEN_OH * (
        np.cos(angle_rad) * vec_bo + np.sin(angle_rad) * v_perp
    )
    h2_coords = O_coords + BOND_LEN_OH * (
        np.cos(angle_rad) * vec_bo - np.sin(angle_rad) * v_perp
    )

    return [('H', h1_coords), ('H', h2_coords), ('O', O_coords)]


capping_functions = {
    'OH': cap_with_OH,
    'H2O': cap_with_H2O
}