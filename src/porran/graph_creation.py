from pymatgen.core import Structure

import networkx as nx
import numpy as np

import warnings

from .utils import extract_linkers



def mof_graph(structure : Structure, radius : float, *args, **kwargs):
    '''
    Create a graph from a MOF structure
    Edges in the graph are defined by bonds found using JmolNN

    Parameters
    ----------
    structure : Structure
        Structure object of the MOF

    Returns
    -------
    nx.Graph
        Graph of the MOF
    '''

    _, linkers_pos_frac = extract_linkers(structure)

    lattice = structure.lattice

    N = len(linkers_pos_frac)

    # 2. Compute the PBC distance matrix (NxN)
    dist_matrix = lattice.get_all_distances(
        linkers_pos_frac, linkers_pos_frac
    )

    # 3. Build adjacency mask (exclude self-edges)
    mask = (dist_matrix <= radius) & (dist_matrix > 1e-12)

    # 4. Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add edges (only i < j to avoid duplicates)
    rows, cols = np.where(mask)
    for i, j in zip(rows, cols):
        if i < j:
            G.add_edge(i, j)

    # 5. Optional sanity check
    check_graph(G)

    return G


def zeo_graph(structure : Structure, *args, **kwargs):
    '''
    Create a graph from a zeolite
    Edges in the graph are defined by T-O-T connections, where T is a tetrahedral atom

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite

    Returns
    -------
    nx.Graph
        Graph of the zeolite
    '''

    # Get indices of Si and O atoms
    si_inds = np.array([i for i, site in enumerate(structure) if site.species_string == 'Si'])
    o_inds = np.array([i for i, site in enumerate(structure) if site.species_string == 'O'])

    

    # Get all T-O-T connections
    d = structure.distance_matrix
    # get the Si-O distances
    si_o_dists = d[si_inds][:, o_inds]
    # get closest 2 Sis to each O
    edge_ind = np.argsort(si_o_dists, axis=0)[:2]

    # Create graph
    G = nx.Graph()

    G.add_nodes_from(np.arange(si_inds.shape[0]))

    for i in range(edge_ind.shape[1]):
        node1, node2 = edge_ind[:, i]
        G.add_edge(node1, node2)

    for i in range(len(G.nodes)):
        G.nodes[i]['value'] = 0

    check_graph(G)

    return G


def radius_graph(structure : Structure, radius : float, mask : np.array = None, *args, **kwargs):
    '''
    Create a graph from a structure
    Edges in the graph are defined by atoms within a certain radius of each other
    Mask can be used to only include certain atoms in the graph

    Parameters
    ----------
    structure : Structure
        Structure object of the all silica zeolite
    radius : float
        Radius to define edges
    mask : np.array, optional
        Array of bools to select atoms in the graph

    Returns
    -------
    nx.Graph
        Graph of the structure
    '''

    # get distance matrix and apply mask
    d = structure.distance_matrix

    # set all distance to self to inf
    np.fill_diagonal(d, np.inf)

    # apply mask
    if mask is not None:
        d = d[mask][:, mask]
        atom_inds = np.arange(len(structure))[mask]
    else:
        atom_inds = np.arange(len(structure))

    edge_ind = np.argwhere(d < radius).T

    # Create graph
    G = nx.Graph()

    G.add_nodes_from(np.arange(atom_inds.shape[0]))

    for i in range(edge_ind.shape[1]):
        node1, node2 = edge_ind[:, i]
        G.add_edge(node1, node2)

    for i in range(len(G.nodes)):
        G.nodes[i]['value'] = 0

    check_graph(G)

    return G


def check_graph(G : nx.Graph):
    '''
    Perform checks on a graph
    Some replacament algorithms might not work if a warning is given

    Parameters
    ----------
    G : nx.Graph
        Graph to check

    Returns
    -------
    None
    '''
    n_edges = len(G.edges)
    if n_edges == 0:
        warnings.warn('Graph has no edges')
    
    conn = nx.is_connected(G)
    if not conn:
        warnings.warn('Graph is not connected')

    