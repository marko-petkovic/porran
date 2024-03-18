from pymatgen.core import Structure

import numpy as np


def create_zeo(structure : Structure, mask, replacement_inds, *args, **kwargs):
    '''
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
    Structure
        Structure with Si atoms replaced by Al atoms
    '''

    
    # select indices of Si atoms to replace
    inds = np.where(mask)[0]
    inds = inds[replacement_inds]

    structure_copy = structure.copy()
    structure_copy[inds] = 'Al'

    return structure_copy
    