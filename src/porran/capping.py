from pymatgen.core.structure import Structure, Lattice
import numpy as np
from typing import List, Tuple

BOND_LEN_MO = 1.8
BOND_LEN_OH = 0.96
HOH_ANGLE = 104.5
CAP_CHARGE = {"OH": -1.0, "H2O": 0.0} 




# TODO: Generalize capping functions for different metals and cap groups
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

    # find the nearest Al atom to the oxygen
    bonded_site = sites[bonded_ind]
    open_site = sites[open_ind]
    # mask Al atoms in dists (distance matrix)
    
    frac_bonded = lattice.get_fractional_coords(bonded_site[1])
    frac_open = lattice.get_fractional_coords(open_site[1])
    
    dfrac = frac_bonded - frac_open # vector pointing from open metal site to the bonded linker atom
    dfrac -= np.round(dfrac)      # wrap to [-0.5, 0.5] along each axis
    vec_BO = lattice.get_cartesian_coords(dfrac)
    vec_BO /= np.linalg.norm(vec_BO)

    O_coords = lattice.get_cartesian_coords(open_site[1]) + BOND_LEN_MO * vec_BO
    h_coords = O_coords + BOND_LEN_OH * vec_BO

    return [('H', h_coords), ('O', O_coords)]


def cap_with_H2O(lattice: Lattice, sites: List[Tuple[str, np.ndarray]], open_ind: int, bonded_ind: int):
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
    # find the nearest Al atom to the oxygen
    bonded_site = sites[bonded_ind]
    open_site = sites[open_ind]
    # mask Al atoms in dists (distance matrix)
    
    frac_bonded = lattice.get_fractional_coords(bonded_site[1])
    frac_open = lattice.get_fractional_coords(open_site[1])
    
    dfrac = frac_bonded - frac_open # vector pointing from open metal site to the bonded linker atom
    dfrac -= np.round(dfrac)      # wrap to [-0.5, 0.5] along each axis
    vec_BO = lattice.get_cartesian_coords(dfrac)
    vec_BO /= np.linalg.norm(vec_BO)

    O_coords = lattice.get_cartesian_coords(open_site[1]) + BOND_LEN_MO * vec_BO


    if abs(vec_BO[0]) < 0.9:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = np.array([0.0, 1.0, 0.0])
    # Make it perpendicular
    v_perp = perp - np.dot(perp, vec_BO) * vec_BO
    v_perp /= np.linalg.norm(v_perp)

    # Rotate perpendicular vector to set HOH angle
    angle_rad = np.radians(HOH_ANGLE / 2)
    h1_coords = O_coords + BOND_LEN_OH * (
        np.cos(angle_rad) * vec_BO + np.sin(angle_rad) * v_perp
    )
    h2_coords = O_coords + BOND_LEN_OH * (
        np.cos(angle_rad) * vec_BO - np.sin(angle_rad) * v_perp
    )

    return [('H', h1_coords), ('H', h2_coords), ('O', O_coords)]


capping_functions = {
    'OH': cap_with_OH,
    'H2O': cap_with_H2O
}