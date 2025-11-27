import numpy as np
import networkx as nx
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

def extract_linkers(download_path: str):

    _, _, frac_pos = readcif(f'{download_path}/linkers.cif')
    i, j, _ = read_cif_bonds(f'{download_path}/linkers.cif')

    G = nx.Graph()
    for idx, pos in enumerate(frac_pos):
        G.add_node(idx, element=pos)
    for start, end in zip(i, j):
        G.add_edge(start, end)
    components = list(nx.connected_components(G))

    linkers = []
    linkers_pos = []

    for component in components:
        linker_group = list(component)
        linker_pos = [frac_pos[index] for index in linker_group]
        linkers_pos.append(mean_frac_pbc(np.array(linker_pos)))
        linkers.append(linker_group)

    return linkers, linkers_pos

    # jmol = JmolNN()
    # struct_graph = StructureGraph.from_local_env_strategy(structure, jmol)

    # # G = struct_graph.graph.copy()
    # visited = set()
    # linkers = []
    # linkers_pos = []

    # def find_linker_group(atom_index):
    #     linker_group = set()
    #     atoms_to_visit = [atom_index]
        
    #     while atoms_to_visit:
    #         current_atom = atoms_to_visit.pop()
    #         if current_atom in visited:
    #             continue
            
    #         visited.add(current_atom)
    #         linker_group.add(current_atom)
            
    #         for neighbor in struct_graph.get_connected_sites(current_atom):
    #             neighbor_index = neighbor.index
    #             neighbor_element = structure[neighbor_index].species_string
                
    #             # TODO: generalize metal check + linker elements
    #             if neighbor_index not in visited and neighbor_element not in {"Al"}:
    #                 if neighbor_element in {"C", "O", "N", "H"}:
    #                     atoms_to_visit.append(neighbor_index)
                
    #     if any(element == "C" for element in [structure[index].species_string for index in linker_group]):
    #         return linker_group
    #     else:
    #         return None
        
    # for i, site in enumerate(structure):
    #     if not site.specie.is_metal and i not in visited:
    #         linker = find_linker_group(i)
    #         if linker:
    #             # calculate positions of linker atoms
    #             linker_pos = [structure[index].frac_coords for index in linker]
    #             # calculate center of mass of linker (fractional coordinates, pbc considered)
    #             linkers_pos.append(mean_frac_pbc(np.array(linker_pos)))


    #             linkers.append(linker)

    # return linkers, linkers_pos


def readcif(name):
    with open(name, "r") as fi:
        EIF = fi.readlines()
        cond2 = False
        atom_props_count = 0
        atomlines = []
        counter = 0
        cell_parameter_boundary = [0.0, 0.0]
        for line in EIF:
            line_stripped = line.strip()
            if (not line) or line_stripped.startswith("#"):
                continue
            line_splitted = line.split()

            if line_stripped.startswith("_cell_length_a"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_a = float(temp)
                cell_parameter_boundary[0] = counter + 1
            elif line_stripped.startswith("_cell_length_b"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_b = float(temp)
            elif line_stripped.startswith("_cell_length_c"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_c = float(temp)
            elif line_stripped.startswith("_cell_angle_alpha"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_alpha = float(temp)
            elif line_stripped.startswith("_cell_angle_beta"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_beta = float(temp)
            elif line_stripped.startswith("_cell_angle_gamma"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_gamma = float(temp)
                cell_parameter_boundary[1] = counter + 1
            if cond2 and line_stripped.startswith("loop_"):
                break
            else:
                if line_stripped.startswith("_atom"):
                    atom_props_count += 1
                    if line_stripped == "_atom_site_label":
                        type_index = atom_props_count - 1
                    elif line_stripped == "_atom_site_fract_x":
                        fracx_index = atom_props_count - 1
                    elif line_stripped == "_atom_site_fract_y":
                        fracy_index = atom_props_count - 1
                    elif line_stripped == "_atom_site_fract_z":
                        fracz_index = atom_props_count - 1
                    cond2 = True
                elif cond2:
                    if len(line_splitted) == atom_props_count:
                        atomlines.append(line)
            counter += 1
        positions = []
        atomtypes = []
        for _, at in enumerate(atomlines):
            ln = at.strip().split()
            positions.append(
                [
                    float(ln[fracx_index].replace("(", "").replace(")", "")),
                    float(ln[fracy_index].replace("(", "").replace(")", "")),
                    float(ln[fracz_index].replace("(", "").replace(")", "")),
                ]
            )
            ln[type_index] = ln[type_index].strip("_")
            at_type = "".join([i for i in ln[type_index] if not i.isdigit()])
            atomtypes.append(atom_to_number[at_type])

        lattice_params = np.array(
            [cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma]
        )
        positions = np.array(positions)
        atomtypes = np.array(atomtypes)
        return lattice_params, atomtypes, positions

def process_cif_jimage(string):
    if string == ".":
        return [0, 0, 0]
    else:
        return [int(string[2]) - 5, int(string[3]) - 5, int(string[4]) - 5]

def get_digits(string):
    return int("".join([c for c in string if c.isdigit()]))

def read_cif_bonds(cif):
    with open(cif, "r") as f:
        lines = f.read().splitlines()

    # No bonds
    if sum(["_ccdc_geom_bond_type" in lin for lin in lines]) == 0:
        return [], [], []

    bond_start = (
        int(np.nonzero(["_ccdc_geom_bond_type" in lin for lin in lines])[0]) + 1
    )
    from_index = []
    to_index = []
    to_jimage = []
    for idx in range(bond_start, len(lines)):
        v1, v2, _, jimage, _ = [x for x in lines[idx].split(" ") if x]
        v1 = get_digits(v1)
        v2 = get_digits(v2)
        jimage = process_cif_jimage(jimage)
        from_index.append(v1)
        to_index.append(v2)
        to_jimage.append(jimage)
    return from_index, to_index, to_jimage

def write_cif(structure: Structure, filename: str, decimals: int = 3, *args, **kwargs):
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
            # for zeolites:
            if site.species_string == 'Si':
                f.write(f"{site.species_string} {site.species_string} {site.frac_coords[0]:.{decimals}f} {site.frac_coords[1]:.{decimals}f} {site.frac_coords[2]:.{decimals}f} -0.393\n")

            else:
                f.write(f"{site.species_string} {site.species_string} {site.frac_coords[0]:.{decimals}f} {site.frac_coords[1]:.{decimals}f} {site.frac_coords[2]:.{decimals}f} 0.000\n")

