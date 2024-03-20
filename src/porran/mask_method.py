from pymatgen.core import Structure
from typing import List
import numpy as np


def mask_zeo(structure : Structure, *args, **kwargs):
    '''
    Calculate a mask to select Si atoms in a zeolite

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    
    Returns
    -------
    np.array
        Mask to select Si atoms in the structure
    '''
    return np.array([site.species_string == 'Si' for site in structure])


def mask_species(structure : Structure, species : List[str], *args, **kwargs):
    '''
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
    '''
    return np.array([site.species_string in species for site in structure], dtype=bool)


def mask_all(structure : Structure, *args, **kwargs):
    '''
    Calculate a mask to select all atoms in a structure

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    
    Returns
    -------
    np.array
        Mask to select all atoms in the structure
    '''
    return np.ones(len(structure), dtype=bool)

def mask_array(structure : Structure, mask : np.array, *args, **kwargs):
    '''
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
    '''
    if len(mask) != len(structure):
        raise ValueError('Mask array must be the same length as the structure')
    return mask.astype(bool)
