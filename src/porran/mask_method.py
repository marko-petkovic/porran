from typing import Callable, List

import numpy as np
from pymatgen.core import Structure


def mask_zeo(structure: Structure, *args, **kwargs):
    """
    Calculate a mask to select Si atoms in a zeolite

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite

    Returns
    -------
    np.array
        Mask to select Si atoms in the structure
    """
    return np.array([site.species_string == "Si" for site in structure])


def mask_species(structure: Structure, species: List[str], *args, **kwargs):
    """
    Calculate a mask to select atoms of certain species in a structure

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    species : List[str]
        List of species to select

    Returns
    -------
    np.array
        Mask to select atoms of certain species in the structure
    """
    return np.array([site.species_string in species for site in structure], dtype=bool)


def mask_all(structure: Structure, *args, **kwargs):
    """
    Calculate a mask to select all atoms in a structure

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite

    Returns
    -------
    np.array
        Mask to select all atoms in the structure
    """
    return np.ones(len(structure), dtype=bool)


def mask_array(structure: Structure, mask: np.array, *args, **kwargs):
    """
    Calculate a mask to select atoms in a structure based on a mask array

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    mask : np.array
        Mask array

    Returns
    -------
    np.array
        Mask to select atoms in the structure based on the mask array
    """
    if len(mask) != len(structure):
        raise ValueError("Mask array must be the same length as the structure")
    return mask.astype(bool)


def mask_box(structure: Structure, box: np.array, *args, **kwargs):
    """
    Calculate a mask to select atoms in a structure based on a box

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    box : np.array
        Box to select atoms (fractional coordinates)
        Shape: (3,2)

    Returns
    -------
    np.array
        Mask to select atoms in the structure based on the box
    """
    mask = np.ones(len(structure), dtype=bool)
    for i in range(3):
        # if the box crosses the boundary
        if box[i, 0] > box[i, 1]:
            mask = mask & (
                (structure.frac_coords[:, i] >= box[i, 0])
                | (structure.frac_coords[:, i] <= box[i, 1])
            )
        else:
            mask = (
                mask
                & (structure.frac_coords[:, i] >= box[i, 0])
                & (structure.frac_coords[:, i] <= box[i, 1])
            )
    return mask


def mask_h_on_c(structure: Structure, *args, **kwargs) -> np.ndarray:
    """Create a mask which has True for all hydrogen atoms bonded to carbon atoms.

    Parameters
    ----------
    structure : Structure
        Structure object of a MOF.

    Returns
    -------
    np.ndarray
        Mask with True for hydrogen atoms bonded to carbon atoms.
    """
    # define the maximum distance to detect a C-H bond
    max_ch_bond_length: float = 1.15  # Angstrom

    # get the indices of all H and C atoms in the structure
    h_indices = np.array(structure.indices_from_symbol("H"))
    c_indices = np.array(structure.indices_from_symbol("C"))

    # get a distance matrix between all H and C atoms in the structure
    dm = structure.distance_matrix[h_indices, :][:, c_indices]  # shape: (n_H, n_C)

    # find the indices of H atoms that are closer than
    # max_ch_bond_length to one and only one C atom
    h_indices_mask = np.sum(dm < max_ch_bond_length, axis=1) == 1

    # create a mask for all atoms in the structure
    mask = np.zeros(len(structure), dtype=bool)
    mask[h_indices] = h_indices_mask

    return mask


def mask_combination(masks: List[Callable], *args, **kwargs):
    """
    Creates a function that combines multiple masks

    Parameters
    ----------
    masks : List[Callable]
        List of mask functions

    Returns
    -------
    Callable
        Function that combines multiple masks
    """

    def mask_comb(structure: Structure, mask_method, *args, **kwargs):
        mask = np.ones(len(structure), dtype=bool)
        x = 0
        for m in masks:
            mask = mask & m(structure, mask_method[x], *args, **kwargs)
            x += 1
        return mask

    return mask_comb
