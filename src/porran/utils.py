from typing import Dict, Optional
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.graphs import StructureGraph

ATOMS = [        
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

atom_to_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,"Hg":80,
    "Tl":81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,"Pa":91,"U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,"Bk":97,"Cf":98,"Es":99,"Fm":100,
    "Md":101,"No":102,"Lr":103,"Rf":104,"Db":105,"Sg":106,"Bh":107,"Hs":108,"Mt":109,"Ds":110,
    "Rg":111,"Cn":112,"Nh":113,"Fl":114,"Mc":115,"Lv":116,"Ts":117,"Og":118
}

number_to_atom = {v: k for k, v in atom_to_number.items()}


def is_atom(element: str):
    '''
    Check if an element is an atom

    Parameters
    ----------
    element : str
        Element to check

    Returns
    -------
    bool
        True if element is an atom, False otherwise
    '''
    return element in ATOMS


def determine_crystal_system(a, b, c, alpha, beta, gamma, digits=3):

    a = round(a, digits)
    b = round(b, digits)
    c = round(c, digits)

    alpha = round(alpha, digits)
    beta = round(beta, digits)
    gamma = round(gamma, digits)

    if a == b == c:

        if alpha == beta == gamma == 90:
            return 'cubic'

        elif alpha == beta == gamma:
            return 'trigonal'
    
    elif (a == b != c) or (a == c != b) or (b == c != a):

        if alpha == beta == gamma == 90:
            return 'tetragonal'

        elif (alpha == beta == 90 and gamma == 120) or \
             (alpha == gamma == 90 and beta == 120) or \
             (beta == gamma == 90 and alpha == 120):
            return 'hexagonal'
    
    elif a != b != c:

        if alpha == beta == gamma == 90:
            return 'orthorhombic'

        elif (alpha == beta == 90 and gamma != 90) or \
            (alpha == gamma == 90 and beta != 90) or \
            (beta == gamma == 90 and alpha != 90):
            return 'monoclinic'

    
    return 'triclinic'

def mean_frac_pbc(frac_coords: np.ndarray) -> np.ndarray:
    """
    Compute the mean of fractional coordinates under periodic boundary conditions.
    Input array shape: (N, 3) with coordinates in [0, 1).

    Returns a single (3,) fractional coordinate in [0, 1).
    """
    # Convert fractional coords to angles on the unit circle
    angles = frac_coords * 2 * np.pi

    # Vector average
    sin_sum = np.sin(angles).mean(axis=0)
    cos_sum = np.cos(angles).mean(axis=0)

    # Angle of resulting mean vector
    mean_angles = np.arctan2(sin_sum, cos_sum)

    # Back to fractional coords in [0, 1)
    mean_frac = (mean_angles / (2 * np.pi)) % 1.0
    return mean_frac

def extract_linkers(structure: Structure):
    jmol = JmolNN()
    struct_graph = StructureGraph.from_local_env_strategy(structure, jmol)

    # G = struct_graph.graph.copy()
    visited = set()
    linkers = []
    linkers_pos = []

    def find_linker_group(atom_index):
        linker_group = set()
        atoms_to_visit = [atom_index]
        
        while atoms_to_visit:
            current_atom = atoms_to_visit.pop()
            if current_atom in visited:
                continue
            
            visited.add(current_atom)
            linker_group.add(current_atom)
            
            for neighbor in struct_graph.get_connected_sites(current_atom):
                neighbor_index = neighbor.index
                neighbor_element = structure[neighbor_index].species_string # type: ignore
                
                # TODO: generalize metal check + linker elements
                if neighbor_index not in visited and neighbor_element not in {"Al"}:
                    if neighbor_element in {"C", "O", "N", "H"}:
                        atoms_to_visit.append(neighbor_index)
                
        if any(element == "C" for element in [structure[index].species_string for index in linker_group]): # type: ignore
            return linker_group
        else:
            return None
        
    for i, site in enumerate(structure):
        if not site.specie.is_metal and i not in visited:
            linker = find_linker_group(i)
            if linker:
                # calculate positions of linker atoms
                linker_pos = [structure[index].frac_coords for index in linker] # type: ignore
                # calculate center of mass of linker (fractional coordinates, pbc considered)
                linkers_pos.append(mean_frac_pbc(np.array(linker_pos)))


                linkers.append(linker)

    return linkers, linkers_pos


def write_cif(structure: Structure, filename: str, decimals: int = 4, custom_charges: Optional[Dict[str, float]] = None, *args, **kwargs):
    '''
    Write a structure to a CIF file

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure to write
    filename : str
        Name of the CIF file
    '''
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c

    alpha = structure.lattice.alpha
    beta = structure.lattice.beta
    gamma = structure.lattice.gamma

    vol = structure.volume

    crystal_system = determine_crystal_system(a, b, c, alpha, beta, gamma)
    # sga = SpacegroupAnalyzer(structure, symprec=0.0001)
    # try:
    #     crystal_system = sga.get_crystal_system()
    # except:
    #     crystal_system = 'triclinic'


    
    with open(filename, 'w') as f:
        f.write("data_structure\n")
        f.write("\n")
        f.write(f"_cell_length_a {a:.{decimals}f}\n")
        f.write(f"_cell_length_b {b:.{decimals}f}\n")
        f.write(f"_cell_length_c {c:.{decimals}f}\n")
        f.write(f"_cell_angle_alpha {alpha:.{decimals}f}\n")
        f.write(f"_cell_angle_beta {beta:.{decimals}f}\n")
        f.write(f"_cell_angle_gamma {gamma:.{decimals}f}\n")
        f.write(f"_cell_volume {vol:.{decimals}f}\n")
        f.write("\n")
        f.write(f"_symmetry_cell_setting {crystal_system}\n")
        f.write(f"_symmetry_space_group_name_Hall 'P 1'\n")
        f.write(f"_symmetry_space_group_name_H-M 'P 1'\n")
        f.write("_symmetry_Int_Tables_number 1\n")
        f.write("_symmetry_equiv_pos_as_xyz 'x,y,z'\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_site_charge\n")

        for site in structure:
            
        

            if "Label: " in site.label:
                atom_label = site.label
                # remove "Label: " from the label
                atom_label = atom_label.replace("Label: ", "")
            else:
                atom_label = site.species_string


            if custom_charges is not None and atom_label in custom_charges:
                charge = custom_charges[atom_label]
            else:
                charge = 0.0

            f.write(f"{atom_label}\t{site.species_string}\t{site.frac_coords[0]:.{decimals}f}\t{site.frac_coords[1]:.{decimals}f}\t{site.frac_coords[2]:.{decimals}f}\t{charge:.{decimals}f}\n")

