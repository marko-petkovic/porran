import logging
from typing import List, Optional

import numpy as np
from pymatgen.core import Molecule, Structure, Lattice
from itertools import permutations

from .transformations import rotation_axis_angle
from .utils import (
    expand_frac_positions,
    extract_linkers,
    mean_frac_pbc,
    normalize_supercell,
    number_to_atom,
    read_cif_bonds,
    readcif,
)
from .capping import capping_functions, CAP_CHARGE


logger = logging.getLogger(__name__)


def _selection_to_indices(selection):
    """Normalize replacement selection into an explicit list of indices."""
    sel = np.array(selection)

    if sel.dtype == bool:
        return np.where(sel)[0].tolist()

    if np.issubdtype(sel.dtype, np.integer):
        return sel.astype(int).tolist()

    raise ValueError("replacement_inds must be an integer index array or a boolean mask")


def _expand_linkers(linkers, n_atoms_unit, supercell):
    """Replicate unit-cell linker atom indices across a supercell."""
    sx, sy, sz = supercell
    n_cells = sx * sy * sz
    expanded = []
    for cell_idx in range(n_cells):
        offset = cell_idx * n_atoms_unit
        for linker in linkers:
            expanded.append([atom_idx + offset for atom_idx in linker])
    return expanded


def _expand_bonds(bond_i, bond_j, bond_jimage, n_atoms_unit, supercell):
    """Expand unit-cell bonds to a supercell, preserving boundary-crossing connectivity."""
    sx, sy, sz = supercell

    cell_coords = []
    for ix in range(sx):
        for iy in range(sy):
            for iz in range(sz):
                cell_coords.append((ix, iy, iz))

    cell_to_idx = {coord: idx for idx, coord in enumerate(cell_coords)}

    expanded_i = []
    expanded_j = []
    for source_cell in cell_coords:
        source_idx = cell_to_idx[source_cell]
        source_offset = source_idx * n_atoms_unit

        for i, j, jimg in zip(bond_i, bond_j, bond_jimage):
            target_cell = (
                (source_cell[0] + jimg[0]) % sx,
                (source_cell[1] + jimg[1]) % sy,
                (source_cell[2] + jimg[2]) % sz,
            )
            target_idx = cell_to_idx[target_cell]
            target_offset = target_idx * n_atoms_unit

            expanded_i.append(i + source_offset)
            expanded_j.append(j + target_offset)

    return expanded_i, expanded_j


def _build_frac_index_map(frac_positions: np.ndarray, decimals: int = 6, wrap: bool = True):
    """Build a map from rounded fractional coordinates to atom indices."""
    mapping = {}
    for idx, frac in enumerate(frac_positions):
        if wrap:
            key = tuple(np.round(np.mod(frac, 1.0), decimals=decimals))
        else:
            key = tuple(np.round(frac, decimals=decimals))
        mapping.setdefault(key, []).append(idx)
    return mapping


def _build_linker_to_mof_map(
    linker_frac_positions: np.ndarray,
    mof_frac_positions: np.ndarray,
    decimals: int = 6,
):
    """Map each linker atom index (unit cell) to an atom index in mof_asr (unit cell)."""
    frac_index_map = _build_frac_index_map(mof_frac_positions, decimals=decimals)
    linker_to_mof = {}
    for linker_idx, frac in enumerate(linker_frac_positions):
        key = tuple(np.round(np.mod(frac, 1.0), decimals=decimals))
        candidates = frac_index_map.get(key, [])
        if not candidates:
            continue
        linker_to_mof[linker_idx] = candidates.pop()
    return linker_to_mof


def _build_unit_cell_linker_bonds(
    linker_indices: List[int],
    linkers: List[List[int]],
    linker_to_mof_unit: dict,
    bond_i: List[int],
    bond_j: List[int],
    n_linker_atoms_unit: int,
    n_mof_atoms_unit: int,
):
    """
    Build a map: linker_id → list of (metal_mof_idx, linker_atom_mof_idx) pairs.
    This represents the unit-cell bond topology for removed linkers.
    """
    linker_bonds = {}  # linker_id → [(metal_idx, linker_atom_idx), ...]
    
    for linker_id in linker_indices:
        linker_bonds[linker_id] = []
    
    # Build set of removed linker atoms in MOF
    atoms_to_remove_in_mof = set()
    for linker_id in linker_indices:
        for atom_idx in linkers[linker_id]:
            mof_idx = linker_to_mof_unit.get(atom_idx)
            if mof_idx is not None:
                atoms_to_remove_in_mof.add(mof_idx)
    
    # Find bonds between removed atoms and kept atoms
    for i, j in zip(bond_i, bond_j):
        # Check if bond crosses removed/kept boundary
        if (i in atoms_to_remove_in_mof and j not in atoms_to_remove_in_mof) or \
           (j in atoms_to_remove_in_mof and i not in atoms_to_remove_in_mof):
            
            # Identify which side is linker vs metal
            if i in atoms_to_remove_in_mof:
                linker_mof_idx = i
                metal_mof_idx = j
            else:
                linker_mof_idx = j
                metal_mof_idx = i
            
            # Find which linker_id this linker atom belongs to
            for linker_id in linker_indices:
                if linker_mof_idx in [linker_to_mof_unit.get(a) for a in linkers[linker_id]]:
                    linker_bonds[linker_id].append((metal_mof_idx, linker_mof_idx))
                    break
    
    return linker_bonds




def create_zeo(structure: Structure, mask, replacement_inds, modify_O_connected_to_Al: bool = False, modify_O_connected_to_Al_Al: bool = False, *args, **kwargs):
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
    modify_O_connected_to_Al : bool
        Whether to modify O atoms connected to Al atoms
    modify_O_connected_to_Al_Al : bool
        Whether to modify O atoms connected to Al atoms that are connected to Al atoms

    Returns
    -------
    List[Structure]
        List with a single Structure with Si atoms replaced by Al atoms
    """

    # select indices of Si atoms to replace
    inds = np.where(mask)[0]
    inds = inds[replacement_inds]

    structure_copy = structure.copy()
    structure_copy[inds] = "Al" # type: ignore

    if modify_O_connected_to_Al:
        o_inds = np.where(np.array([site.species_string == "O" for site in structure_copy]))[0]
        dist_matrix = structure_copy.distance_matrix
        # set diagonal to inf to ignore self-distance
        np.fill_diagonal(dist_matrix, np.inf)

        # calculate closest 2 neighbours for each O atom
        closest_inds = np.argsort(dist_matrix[o_inds], axis=1)[:, :2]

        for i, o_ind in enumerate(o_inds):
            o_ind = int(o_ind)
            neighbours = closest_inds[i]
            
            if modify_O_connected_to_Al:
                if any(structure_copy[neighbour].species_string == "Al" for neighbour in neighbours): # type: ignore
                    structure_copy[o_ind].label = "Label: Oa" # type: ignore
            
            
            if modify_O_connected_to_Al_Al:
                if all(structure_copy[neighbour].species_string == "Al" for neighbour in neighbours): # type: ignore
                    structure_copy[o_ind].label = "Label: Oaa" # type: ignore
            
           
        

    return [structure_copy]




def random_sample_only_replace_if_needed(choices: List, num_samples:int) -> List:
    """Sample from choices without replacement when possible, with fallback repetition."""

    if num_samples <= len(choices):
        return np.random.choice(choices, size=num_samples, replace=False).tolist()
    else:
        perp = np.array([0.0, 1.0, 0.0])
    

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
    supercell=(1, 1, 1),
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

    supercell = normalize_supercell(supercell)

    replacement_inds = _selection_to_indices(replacement_inds)

    linkers, _ = extract_linkers(download_path=download_path)
    _, _, frac_pos = readcif(f'{download_path}/linkers.cif')

    _, atomtypes_mof, frac_pos_mof = readcif(f'{download_path}/mof_asr.cif')
    bond_i, bond_j, bond_jimage = read_cif_bonds(f'{download_path}/mof_asr.cif')

    frac_pos_unit = np.array(frac_pos)
    frac_pos_mof_unit = np.array(frac_pos_mof)
    atomtypes_mof_unit = np.array(atomtypes_mof)

    linker_to_mof_unit = _build_linker_to_mof_map(frac_pos_unit, frac_pos_mof_unit)

    n_linker_atoms_unit = len(frac_pos_unit)
    n_mof_atoms_unit = len(atomtypes_mof_unit)
    n_linkers_unit = len(linkers)

    # In supercell mode replacement_inds are indices in the expanded linker list.
    # Keep them as-is (sampled defects are independent of supercell size), but also
    # compute which unit-linker templates are needed for bond replication.
    if supercell != (1, 1, 1):
        unit_replacement_inds = sorted(set(linker_id % n_linkers_unit for linker_id in replacement_inds))
    else:
        unit_replacement_inds = replacement_inds

    # Build unit-cell bond template for the linkers to be removed
    linker_bonds_unit = _build_unit_cell_linker_bonds(
        unit_replacement_inds,
        linkers,
        linker_to_mof_unit,
        bond_i,
        bond_j,
        n_linker_atoms_unit,
        n_mof_atoms_unit,
    )

    if supercell != (1, 1, 1):
        linkers = _expand_linkers(linkers, n_linker_atoms_unit, supercell)
        frac_pos = expand_frac_positions(frac_pos_unit, supercell)
        frac_pos_mof = expand_frac_positions(frac_pos_mof_unit, supercell)
        atomtypes_mof = np.tile(atomtypes_mof, int(np.prod(supercell)))
        bond_i, bond_j = _expand_bonds(bond_i, bond_j, bond_jimage, n_mof_atoms_unit, supercell)

        expected_sites = len(atomtypes_mof)
        if len(structure) != expected_sites:
            structure = structure.copy()
            structure.make_supercell(supercell)
    else:
        frac_pos = frac_pos_unit
        frac_pos_mof = frac_pos_mof_unit
        atomtypes_mof = atomtypes_mof_unit





    # mark atoms to remove & associate each removed atom with a linker ID
    atom_to_linker = {}  # maps atom index → linker id
    atoms_to_remove = []

    # TODO: calculate charges of missing linkers


    # Build atoms_to_remove by expanding replacement_inds appropriately
    if supercell == (1, 1, 1):
        # Unit cell: direct mapping from replacement_inds to linker atoms
        for linker_id, linker_inds in enumerate(replacement_inds):
            for atom in linkers[linker_inds]:
                atoms_to_remove.append(atom)
                atom_to_linker[atom] = linker_inds  # Use actual linker ID
    else:
        # Supercell: replacement_inds already point to specific expanded linkers.
        # Remove only the sampled linker instances.
        n_cells = int(np.prod(supercell))
        n_linkers_expanded = n_linkers_unit * n_cells
        for linker_id in replacement_inds:
            if linker_id < 0 or linker_id >= n_linkers_expanded:
                raise ValueError(
                    f"replacement linker index {linker_id} out of bounds for expanded linker list of size {n_linkers_expanded}"
                )
            for atom in linkers[linker_id]:
                atoms_to_remove.append(atom)
                atom_to_linker[atom] = linker_id

    # Identify atoms removed in MOF and create idxes_to_remove_in_mof
    idxes_to_remove_in_mof = []
    atom_to_linker_mof = {}
    
    for atom_idx in atoms_to_remove:
        cell_idx_linker = atom_idx // n_linker_atoms_unit
        local_idx_linker = atom_idx % n_linker_atoms_unit
        
        mof_local_idx = linker_to_mof_unit.get(local_idx_linker)
        if mof_local_idx is None:
            continue
        
        # Map to MOF index in the same cell
        mof_idx = cell_idx_linker * n_mof_atoms_unit + mof_local_idx
        idxes_to_remove_in_mof.append(mof_idx)
        atom_to_linker_mof[mof_idx] = atom_to_linker[atom_idx]

    # Identify bonds between removed and kept atoms using the unit-cell template and replicate in supercell
    bonds_to_replace = []
    
    if supercell == (1, 1, 1):
        # Unit cell: use direct linker_bonds_unit
        for linker_id in unit_replacement_inds:
            for metal_mof_idx, linker_mof_idx in linker_bonds_unit.get(linker_id, []):
                bonds_to_replace.append((linker_mof_idx, metal_mof_idx, linker_id))
    else:
        # Supercell: build boundary bonds only for sampled expanded linker instances.
        for linker_id in replacement_inds:
            cell_idx = linker_id // n_linkers_unit
            unit_linker_id = linker_id % n_linkers_unit
            cell_offset_mof = cell_idx * n_mof_atoms_unit

            for metal_mof_idx, linker_mof_idx in linker_bonds_unit.get(unit_linker_id, []):
                linker_mof_idx_sc = linker_mof_idx + cell_offset_mof
                metal_mof_idx_sc = metal_mof_idx + cell_offset_mof
                bonds_to_replace.append((linker_mof_idx_sc, metal_mof_idx_sc, linker_id))


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
    
    return [structure_copy]


def balance_charges(lattice: Lattice, positions: List[List[float]], caps: List[str], cap_charges: List[float]) -> List[str]:
    """
    Assign capping groups to vacancy positions to minimize net dipole magnitude.
    """


    assert len(positions) == len(cap_charges), "Positions and cap_charges must have the same length"
    positions = np.array(positions) # type: ignore
    # get geometric center of positions
    center_frac = mean_frac_pbc(positions) # type: ignore

    # get vectors from center to each position
    vecs = []
    for pos in positions:
        dfrac = pos - center_frac
        dfrac -= np.round(dfrac)  # wrap to [-0.5, 0.5]
        vec = lattice.get_cartesian_coords(dfrac)
        vecs.append(vec)
    vecs = np.array(vecs)
    # Exact search scales as O(n!), so for larger vacancy sets use a greedy fallback.
    if len(caps) > 9:
        remaining = list(range(len(caps)))
        assigned = []
        dipole = np.zeros(3)
        for i in range(len(caps)):
            best_idx = None
            best_score = float("inf")
            for cap_idx in remaining:
                score = np.linalg.norm(dipole + vecs[i] * cap_charges[cap_idx])
                if score < best_score:
                    best_score = score
                    best_idx = cap_idx
            assigned.append(best_idx)
            remaining.remove(best_idx) # type: ignore[arg-type]
            dipole += vecs[i] * cap_charges[best_idx] # type: ignore[index]
        return [caps[i] for i in assigned]

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

    return [caps[i] for i in best_permutation] # type: ignore

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
