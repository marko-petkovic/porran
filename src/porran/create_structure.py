import logging
from typing import List, Optional

import numpy as np
from pymatgen.core import Molecule, Structure
from scipy import spatial

from .transformations import rotation_axis_angle
from .utils import extract_linkers

logger = logging.getLogger(__name__)

BOND_LEN_MO = 1.8
BOND_LEN_OH = 0.96
HOH_ANGLE = 104.5

def create_zeo(structure: Structure, mask, replacement_inds, *args, **kwargs):
    """
    Creates a structure with Si atoms replaced by Al atoms

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    mask : np.array
        Mask to select atoms to be replaced
    replacement_inds : np.array
        Indices of Si atoms to replace with Al atoms

    Returns
    -------
    List[Structure]
        List with a single Structure with Si atoms replaced by Al atoms
    """

    # select indices of Si atoms to replace
    inds = np.where(mask)[0]
    inds = inds[replacement_inds]

    structure_copy = structure.copy()
    structure_copy[inds] = "Al"

    return [structure_copy]

# TODO: Generalize capping functions for different metals and cap groups
def cap_with_OH(structure: Structure, ox_ind: int, dists: np.array):
    # find the nearest Al atom to the oxygen
    ox_site = structure[ox_ind]
    # mask Al atoms in dists (distance matrix)
    al_inds = [i for i, site in enumerate(structure) if site.species_string == "Al"]
  
    al_dists = dists[ox_ind][al_inds]

    nearest_al_ind = al_inds[np.argmin(al_dists)]
    al_site = structure[nearest_al_ind]
    lattice = structure.lattice
    frac_O = lattice.get_fractional_coords(ox_site.coords)
    frac_Al = lattice.get_fractional_coords(al_site.coords)
    dfrac = frac_O - frac_Al
    dfrac -= np.round(dfrac)      # wrap to [-0.5, 0.5] along each axis
    vec_OA = lattice.get_cartesian_coords(dfrac)
    vec_OA /= np.linalg.norm(vec_OA)
    h_coords = ox_site.coords + BOND_LEN_OH * vec_OA

    return h_coords

def cap_with_H2O(structure: Structure, ox_ind: int, dists: np.array):
    # find the nearest Al atom to the oxygen
    ox_site = structure[ox_ind]
    # mask Al atoms in dists (distance matrix)
    al_inds = [i for i, site in enumerate(structure) if site.species_string == "Al"]

    al_dists = dists[ox_ind][al_inds]

    nearest_al_ind = al_inds[np.argmin(al_dists)]
    al_site = structure[nearest_al_ind]
    lattice = structure.lattice
    frac_O = lattice.get_fractional_coords(ox_site.coords)
    frac_Al = lattice.get_fractional_coords(al_site.coords)
    dfrac = frac_O - frac_Al
    dfrac -= np.round(dfrac)      # wrap to [-0.5, 0.5] along each axis
    vec_OA = lattice.get_cartesian_coords(dfrac)
    vec_OA /= np.linalg.norm(vec_OA)
    
    if abs(vec_OA[0]) < 0.9:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = np.array([0.0, 1.0, 0.0])
    # Make it perpendicular
    v_perp = perp - np.dot(perp, vec_OA) * vec_OA
    v_perp /= np.linalg.norm(v_perp)

    # Rotate perpendicular vector to set HOH angle
    angle_rad = np.radians(HOH_ANGLE / 2)
    h1_coords = ox_site.coords + BOND_LEN_OH * (
        np.cos(angle_rad) * vec_OA + np.sin(angle_rad) * v_perp
    )
    h2_coords = ox_site.coords + BOND_LEN_OH * (
        np.cos(angle_rad) * vec_OA - np.sin(angle_rad) * v_perp
    )

    return h1_coords, h2_coords



def create_defect_mof(
    structure: Structure,
    mask: np.ndarray,
    replacement_inds: np.ndarray,
    cap_group: List[str] = ["OH","H2O"], # currently useless
    *args,
    **kwargs,
):
    """Create a MOF with missing linkers and capped metallic centers.
    Parameters
    ----------
    structure : Structure
        The MOF to create defects in.
    mask : np.ndarray
        Mask to select atoms to be replaced
    replacement_inds : np.ndarray
        Indices of linkers to remove within the structure graph.
    cap_group : List[str]
        The groups to use for capping metallic centers.
    """

    dist_maxtrix_struct = structure.distance_matrix
    
    replacement_inds = np.where(replacement_inds)[0].tolist()

    linkers, _ = extract_linkers(structure)
    atoms_to_remove = [atom for linker_inds in replacement_inds for atom in linkers[linker_inds]]
    new_sites = []
    for i, site in enumerate(structure.sites):
        if i not in atoms_to_remove:
            new_sites.append(site)

    structure_copy = Structure(structure.lattice, [site.specie for site in new_sites],
                               [site.frac_coords for site in new_sites], coords_are_cartesian=False)
    
    # TODO: The code below works only for CAU-10. It should be generalized.
    # loop throug all removed linkers, and identify their Oxygen atoms

    for linker_inds in replacement_inds:
        linker = linkers[linker_inds]
        o_inds = [ind for ind in linker if structure[ind].species_string == "O"]

        # find the two oxygens that are furthest apart
        # these two oxygens will be capped with H20, the rest with OH

        coords = structure.frac_coords[o_inds]
        dist_maxtrix = structure.lattice.get_all_distances(coords, coords)
        i, j = np.unravel_index(np.argmax(dist_maxtrix), dist_maxtrix.shape)
        k, l = set(range(len(o_inds))) - {i, j} # get the other indices

        # add oxygens in the structure copy
        for o_ind in o_inds:
            structure_copy.append(species="O", coords=structure[o_ind].coords, coords_are_cartesian=True)

        # cap oxygens (h2o)
        h2o_h1, h2o_h2 = cap_with_H2O(structure, o_inds[i], dist_maxtrix_struct)
        h2o_h3, h2o_h4 = cap_with_H2O(structure, o_inds[j], dist_maxtrix_struct)

        # cap oxygens (oh)
        oh_h1 = cap_with_OH(structure, o_inds[k], dist_maxtrix_struct)
        oh_h2 = cap_with_OH(structure, o_inds[l], dist_maxtrix_struct)

        for h_coord in [oh_h1, oh_h2, h2o_h1, h2o_h2, h2o_h3, h2o_h4]:
            structure_copy.append(species="H", coords=h_coord, coords_are_cartesian=True)
    
    return [structure_copy]


def create_dmof(
    structure: Structure,
    mask: np.ndarray,
    replacement_inds: np.ndarray,
    dopants: Molecule | List[Molecule],
    max_attempts: int = 100,
    rng: Optional[np.random.Generator] = None,
    *args,
    **kwargs,
) -> List[Structure]:
    """Create a MOF with added functional groups.

    Parameters
    ----------
    structure : Structure
        The MOF to add functional groups to.
    mask : np.ndarray
        Mask to select atoms to be replaced
    replacement_inds : np.ndarray
        Indices to replace within the structure graph.
    dopants: Molecule | List[Molecule]
        Molecule(s) representing the functional groups to add. If a
        single Molecule is provided, then that is used for all
        replacements. If a List, it must have the same length as
        replacement_inds.
    max_attempts: int
        A random rotation is applied to the dopant to avoid overlap with
        existing sites at most max_attempts times. If there is still
        overlap after that, the dopant is not placed and a warning is
        logged.
    rng: np.random.Generator, optional
        Generator for random rotations.
    """
    if rng is None:
        rng = np.random.default_rng()

    max_ch_bond_length: float = 1.15  # Angstrom

    # if dopants is a single Molecule, copy it replacement_inds times
    if isinstance(dopants, Molecule):
        dopants = [dopants] * len(replacement_inds)

    # get indices in the structure of the H atoms to replace, instead of
    # the indices in the graph
    h_indices = np.where(mask)[0][replacement_inds]

    # find the C atoms that the H atoms are bonded to
    dm = structure.distance_matrix[h_indices, :]
    c_indices = np.argwhere(np.logical_and(dm < max_ch_bond_length, dm > 0))[:, 1]

    structure_copy = structure.copy()
    for i, (c_i, h_i) in enumerate(zip(c_indices, h_indices)):
        # get the location and direction for the dopant
        location = structure.cart_coords[c_i]
        direction = structure.cart_coords[h_i] - structure.cart_coords[c_i]

        # rotate the dopant to align with the C-H bond, dopants are
        # assumed to be aligned with the x-axis
        dopant = dopants[i].copy()
        v, a = rotation_axis_angle(np.array([1.0, 0.0, 0.0]), direction)
        dopant.rotate_sites(theta=a, axis=v)

        # move the dopant to the correct location, the origin of the
        # dopant reference frame is assumed to be at the C-atom
        dopant.translate_sites(vector=location)

        # remove H from the structure
        structure_copy.remove_sites([h_i])

        # try to add the dopant to the structure
        for _ in range(max_attempts):
            dopant.rotate_sites(
                theta=rng.uniform(0, 2 * np.pi), axis=direction, anchor=location
            )

            # check for overlap with existing atoms
            # get the fractional coordinates of the dopants.
            d_frac = structure_copy.lattice.get_fractional_coords(
                cart_coords=dopant.cart_coords
            )
            # calculate the distances between dopant and structure atoms
            # along the lattice dimensions, taking periodic boundaries
            # into account
            frac_dists = np.abs(structure_copy.frac_coords[:, None] - d_frac)
            frac_dists = np.where(frac_dists > 0.5, np.abs(1 - frac_dists), frac_dists)
            # convert to cartesian distances
            cart_dists = structure_copy.lattice.get_cartesian_coords(
                fractional_coords=frac_dists
            )
            # calculate square of norm and compare to tolerance
            if np.any(
                np.sum(np.square(cart_dists), -1) < structure_copy.DISTANCE_TOLERANCE**2
            ):
                continue

            # no overlap, add the dopant
            for site in dopant:
                structure_copy.append(
                    species=site.species,
                    coords=site.coords,
                    coords_are_cartesian=True,
                    validate_proximity=False,
                    properties=site.properties,
                )
            break
        else:
            logger.warning(
                "Could not add dopant %s to the structure at index %d",
                dopant.reduced_formula,
                c_i,
            )

        # insert a dummy site back at h_i to keep the indices correct
        structure_copy.insert(idx=h_i, species="X", coords=structure.frac_coords[h_i])

    # remove the dummy sites
    structure_copy.remove_species("X")

    return [structure_copy]
