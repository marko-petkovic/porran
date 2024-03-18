from pymatgen.core import Structure

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
