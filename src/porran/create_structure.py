import logging
from typing import List, Optional

import numpy as np
from pymatgen.core import Molecule, Structure
from scipy import spatial

from .transformations import rotation_axis_angle

logger = logging.getLogger(__name__)


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
        # create a KDTree for overlap check
        point_tree = spatial.KDTree(data=structure_copy.cart_coords)

        # try to add the dopant to the structure
        for _ in range(max_attempts):
            dopant.rotate_sites(
                theta=rng.uniform(0, 2 * np.pi), axis=direction, anchor=location
            )

            # check for overlap with existing atoms
            overlap = point_tree.query_ball_point(
                x=dopant.cart_coords, r=structure_copy.DISTANCE_TOLERANCE, p=2
            )
            overlap = [
                (i, existing_atom)
                for i, atom_to_add in enumerate(overlap)
                for existing_atom in atom_to_add
            ]
            if len(overlap) > 0:
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
            # try:
            #     for site in dopant:
            #         structure_copy.append(
            #             species=site.species,
            #             coords=site.coords,
            #             coords_are_cartesian=True,
            #             validate_proximity=True,
            #             properties=site.properties,
            #         )
            #     break
            # except ValueError as e:
            #     # only catch the error if it is about proximity
            #     if e.args[0] == "New site is too close to an existing site!":
            #         continue
            #     raise e
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
