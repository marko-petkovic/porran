import logging
from typing import List, Optional

import numpy as np
from pymatgen.core import Molecule, Structure, Lattice
from pymatgen.io.cif import CifParser

from itertools import permutations
from scipy import spatial

from .transformations import rotation_axis_angle
from .utils import extract_linkers, read_cif_bonds, readcif, number_to_atom, mean_frac_pbc
from .capping import capping_functions, CAP_CHARGE


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




def random_sample_only_replace_if_needed(choices: List, num_samples:int) -> List:
    """Randomly sample from choices, only replacing if num_samples > len(choices)"""

    if num_samples <= len(choices):
        return np.random.choice(choices, size=num_samples, replace=False).tolist()
    else:
        n_full, rem = divmod(num_samples, len(choices))
        samples = choices * n_full
        samples += np.random.choice(choices, size=rem, replace=False).tolist()
        np.random.shuffle(samples)
        return samples


def create_defect_mof(
    structure: Structure,
    mask: np.ndarray,
    replacement_inds: np.ndarray,
    cap_group: List[str] = ["OH","H2O"], # currently useless
    download_path: str = "downloads",
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
    download_path: str
        Path in which files processed by mofid are downloaded.
    """

    dist_maxtrix_struct = structure.distance_matrix
    
    replacement_inds = np.where(replacement_inds)[0].tolist()

    linkers, _ = extract_linkers(download_path=download_path)
    _, _, frac_pos = readcif(f'{download_path}/linkers.cif')

    _, atomtypes_mof, frac_pos_mof = readcif(f'{download_path}/mof_asr.cif')
    bond_i, bond_j, _ = read_cif_bonds(f'{download_path}/mof_asr.cif')





    # mark atoms to remove & associate each removed atom with a linker ID
    atom_to_linker = {}  # maps atom index → linker id
    atoms_to_remove = []

    # TODO: calculate charges of missing linkers

    for linker_id, linker_inds in enumerate(replacement_inds):
        for atom in linkers[linker_inds]:
            atoms_to_remove.append(atom)
            atom_to_linker[atom] = linker_id   # assign ID per linker
    frac_pos_to_remove = frac_pos[atoms_to_remove]


    # TODO: count removed charges
    # find atom index in mof
    idxes_to_remove_in_mof = []
    atom_to_linker_mof = {}  # linker id mapped to MOF index

    for frac_pos_r, original_atom_idx in zip(frac_pos_to_remove, atoms_to_remove):
        for idx_mof, frac_pos_m in enumerate(frac_pos_mof):
            if np.allclose(frac_pos_r, frac_pos_m, atol=1e-3):
                idxes_to_remove_in_mof.append(idx_mof)
                atom_to_linker_mof[idx_mof] = atom_to_linker[original_atom_idx]
                break

    
    # identify bonds between atoms to remove and atoms to keep
    # these bonds will be used to identify metal centers that need capping
    bonds_to_replace = []
    for i, j in zip(bond_i, bond_j):
        if (i in idxes_to_remove_in_mof and j not in idxes_to_remove_in_mof) or \
           (j in idxes_to_remove_in_mof and i not in idxes_to_remove_in_mof):
            linker_id = atom_to_linker_mof[i] if i in atom_to_linker_mof else atom_to_linker_mof[j]

            bonds_to_replace.append((i, j, linker_id))


    ### ------------------------------------------------------------------------ ###
    ### The following logic is specific to capping CAU-10 with OH and H2O groups ###
    ### ------------------------------------------------------------------------ ###
    
    metals_to_cap = {}  # {linker_id: {metal_idx: [bonded_atom_idxs]}}

    

    for i, j, linker_id in bonds_to_replace:

        # determine which side is metal vs removed
        if i in idxes_to_remove_in_mof:
            metal_ind  = j
            bonded_ind = i
        else:
            metal_ind  = i
            bonded_ind = j

        # initialize linker if not present
        if linker_id not in metals_to_cap:
            metals_to_cap[linker_id] = {}

        # initialize metal site list
        if metal_ind not in metals_to_cap[linker_id]:
            metals_to_cap[linker_id][metal_ind] = []

        # append removed atom index
        metals_to_cap[linker_id][metal_ind].append(bonded_ind)

    old_sites = list([(number_to_atom[atomnumber], frac_coords) for atomnumber, frac_coords in zip(atomtypes_mof, frac_pos_mof)])

    new_sites = []
    # for i, site in enumerate(structure.sites):
    for i, (atomtype, frac_coords) in enumerate(old_sites):
        if i not in idxes_to_remove_in_mof:
            new_sites.append((atomtype, frac_coords))

    

    structure_copy = Structure(structure.lattice, [atomtype for atomtype, _ in new_sites],
                               [frac_coords for _, frac_coords in new_sites], coords_are_cartesian=False)
    
    # TODO: The code below works only for CAU-10. It should be generalized. Using the cap_group argument.
    # TODO: Account for charges when capping. Currently, it's done manually

    # loop throug all removed linkers, and identify their Oxygen atoms


    for linker_id, metal_dict in metals_to_cap.items():

        metals = list(metal_dict.keys())
        n_vac = len(metals)

        cap_groups = (n_vac // 2) * ["OH", "H2O"]
        if n_vac % 2 == 1:
            cap_groups.append("OH")

        cap_charges = [CAP_CHARGE[cap] for cap in cap_groups]

        metal_frac_positions = [old_sites[m][1] for m in metals]
        balanced_caps = balance_charges(structure.lattice, metal_frac_positions, cap_groups, cap_charges)
        
        # assign caps to metals according to balanced result
        for metal_idx, cap in zip(metals, balanced_caps):

            if cap not in capping_functions:
                raise ValueError(f"Unknown cap group: {cap}")

            for bonded_ind in metal_dict[metal_idx]:

                new_atoms = capping_functions[cap](
                    structure.lattice, old_sites, metal_idx, bonded_ind
                )

                for symbol, coords in new_atoms:
                    structure_copy.append(
                        species=symbol,
                        coords=coords,
                        coords_are_cartesian=True
                    )

    # n_vacancies = sum([len(metals_to_cap[metal]) for metal in metals_to_cap])
    # cap_groups = (n_vacancies // 2 )* ["OH", "H2O"]
    # if n_vacancies % 2 == 1:
    #     cap_groups.append("OH")
    # np.random.shuffle(cap_groups)

    # for open_metal in metals_to_cap:
        
    #     # TODO: add better logic for how to choose capping groups
    #     # for now, just cap with H2O and OH. Randomly assign
        
        
        
    #     for idx, bonded_ind in enumerate(metals_to_cap[open_metal]):
    #         cap = cap_groups.pop(0)
    #         if cap not in capping_functions:
    #             raise ValueError(f"Unknown cap group: {cap}")
    #         new_atoms = capping_functions[cap](structure.lattice, old_sites, open_metal, bonded_ind)
    #         for symbol, coords in new_atoms:
    #             structure_copy.append(species=symbol, coords=coords, coords_are_cartesian=True)
                
            # if cap == "OH":
            #     h_coord, o_coord = capping_functions["OH"](structure, open_metal, bonded_ind)
            #     structure_copy.append(species="H", coords=h_coord, coords_are_cartesian=True)
            #     structure_copy.append(species="O", coords=o_coord, coords_are_cartesian=True)
            # elif cap == "H2O":
            #     h1_coord, h2_coord, o_coord = capping_functions["H2O"](structure, open_metal, bonded_ind)
            #     structure_copy.append(species="H", coords=h1_coord, coords_are_cartesian=True)
            #     structure_copy.append(species="H", coords=h2_coord, coords_are_cartesian=True)
            #     structure_copy.append(species="O", coords=o_coord, coords_are_cartesian=True)
            # else:
            #     raise ValueError(f"Unknown cap group: {cap}")
    
    return [structure_copy]


def balance_charges(lattice: Lattice, positions: List[List[float]], caps: List[str], cap_charges: List[float]) -> List[str]:
    """
    Finds the best permutation of the caps to minimize the dipole moment created by the capping groups.
    """


    assert len(positions) == len(cap_charges), "Positions and cap_charges must have the same length"
    positions = np.array(positions)
    # get geometric center of positions
    center_frac = mean_frac_pbc(positions)

    # get vectors from center to each position
    vecs = []
    for pos in positions:
        dfrac = pos - center_frac
        dfrac -= np.round(dfrac)  # wrap to [-0.5, 0.5]
        vec = lattice.get_cartesian_coords(dfrac)
        vecs.append(vec)
    vecs = np.array(vecs)
    best_score = float('inf')
    best_permutation = None
    for perm in permutations(range(len(caps))):
        dipole = np.zeros(3)
        for i, cap_idx in enumerate(perm):
            dipole += vecs[i] * cap_charges[cap_idx]
        score = np.linalg.norm(dipole)
        if score < best_score:
            best_score = score
            best_permutation = perm

    return [caps[i] for i in best_permutation]

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
