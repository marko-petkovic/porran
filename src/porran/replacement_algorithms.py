"""Node-selection algorithms used for substitutions on structure graphs."""

import numpy as np

import networkx as nx

from typing import List, Optional
from itertools import combinations


def _validate_n_subs(n_subs: int, n_nodes: int) -> None:
    """Validate the requested number of substitutions."""
    if n_subs < 0:
        raise ValueError('Number of substitutions must be non-negative')
    if n_subs > n_nodes:
        raise ValueError('Number of substitutions is too large for the structure')


def _distance_vector(G: nx.Graph, nodes: tuple, source) -> np.ndarray:
    """Return graph distances from one source to all nodes in graph order."""
    path_lengths = nx.single_source_shortest_path_length(G, source)
    if len(path_lengths) != len(nodes):
        raise ValueError('Graph must be connected to use maximize_entropy')

    return np.fromiter(
        (path_lengths[node] for node in nodes),
        dtype=float,
        count=len(nodes),
    )

def random(G : nx.Graph, n_subs : int, *args, **kwargs):
    '''
    Randomly select n_subs nodes from the graph  
    
    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select

    Returns
    -------
    np.array
        Array of selected nodes
    '''
    nodes = np.array(tuple(G.nodes()), dtype=object)
    _validate_n_subs(n_subs, len(nodes))

    if n_subs == 0:
        return np.array([], dtype=nodes.dtype)

    selected_indices = np.random.choice(len(nodes), n_subs, replace=False)
    return nodes[selected_indices]


def lowenstein(G : nx.Graph, n_subs : int, n_random : int = 1,*args, **kwargs):
    '''
    Generates all possible random configurations with n_subs Al atoms
    Loops through configurations and selects the first one that satisfies the Lowenstein constraint
    By setting n_random > 1, the function will keep looping until n_random valid configurations are found
    From these, it will select a random one

    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select
    n_random : int, optional
        Number of configurations obeying the Lowenstein constraint from which to sample, default is 1
    
    Returns
    -------
    np.array
        Array of selected nodes
    '''

    G = G.copy()
    nodes = tuple(G.nodes())
    node_array = np.array(nodes, dtype=object)
    _validate_n_subs(n_subs, len(nodes))
    if n_random <= 0:
        raise ValueError('n_random must be positive')
    if n_subs == 0:
        return np.array([], dtype=node_array.dtype)

    # get adjacency matrix in graph-node order
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

    # keep combinations lazy so large search spaces do not need to fit in memory
    combs = combinations(range(len(nodes)), n_subs)

    al_subs = []

    for comb in combs:
        # check if the combination is valid
        if np.sum(adj_matrix[np.ix_(comb, comb)]) == 0:
            al_subs.append(comb)
            if len(al_subs) == n_random:
                break
    
    if len(al_subs) > 0:
        return node_array[list(al_subs[np.random.choice(len(al_subs))])]
    
    raise ValueError('No valid combination found')



def random_lowenstein(G : nx.Graph, n_subs : int, *args, **kwargs):
    '''
    Randomly select n_subs nodes from the graph while obeying the Lowenstein constraint
    
    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select

    Returns
    -------
    np.array
        Array of selected nodes
    '''
    G = G.copy()
    _validate_n_subs(n_subs, G.number_of_nodes())

    if n_subs == 0:
        return np.array([], dtype=int)

    selected_nodes = set()
    for _ in range(n_subs):
        if G.number_of_nodes() == 0:
            raise ValueError('No valid Lowenstein configuration found')

        # Select a random node
        node = np.random.choice(list(G.nodes))
        selected_nodes.add(node)

        # Remove the node and its neighbours from the graph
        G.remove_nodes_from(list(G.neighbors(node)))
        G.remove_node(node)

    return np.array(list(selected_nodes))


def clusters(G : nx.Graph, n_subs : int, node_idx : Optional[int] = None, *args, **kwargs):
    '''
    Selects n_subs nodes around a random node

    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select
    node_idx : int, optional
        Index of the node to select neighbours from. If None, a random node is selected

    Returns
    -------
    np.array
        Array of selected nodes
    '''
    G = G.copy()
    _validate_n_subs(n_subs, G.number_of_nodes())

    if n_subs == 0:
        return np.array([], dtype=int)

    if node_idx is None:
        node_idx = np.random.choice(list(G.nodes))
    elif node_idx not in G:
        raise ValueError('node_idx must be present in the graph')

    target_neighbours = n_subs - 1

    if target_neighbours <= 0:
        return np.array([node_idx])

    # get first neighbours of node_idx
    neighbours = set(G.neighbors(node_idx))

    if len(neighbours) >= target_neighbours:
        # select the required number of neighbours
        neighbours = np.random.choice(list(neighbours), target_neighbours, replace=False)
        neighbours = np.concatenate(([node_idx], neighbours)) # type: ignore
        return neighbours

    while len(neighbours) < target_neighbours:
        # add next shell of neightbours
        added_neighbours = set()
        
        for idx in neighbours:
            new_neighbours = set(G.neighbors(idx))
            added_neighbours = added_neighbours.union(new_neighbours)

        # remove source node and already added neighbours
        added_neighbours = added_neighbours.difference([node_idx])
        added_neighbours = added_neighbours.difference(neighbours)

        # if the added neighbours are more than the required number of subs
        # select a random subset of them
        if len(neighbours) + len(added_neighbours) > target_neighbours:
            added_neighbours = np.random.choice(
                list(added_neighbours),
                target_neighbours - len(neighbours),
                replace=False,
            )
            neighbours = neighbours.union(added_neighbours)
            break
        elif len(added_neighbours) == 0:
            raise ValueError('No neighbours left!')
        else:
            neighbours = neighbours.union(added_neighbours)
    
    return np.array(list(neighbours)+[node_idx])



def chains(G : nx.Graph, n_subs : int, chain_lengths : List[int], *args, **kwargs):
    '''
    Selects nodes in chains of specified lengths

    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select
    chain_lengths : List[int]
        List of chain lengths to select
        Sum of chain lengths should be equal to n_subs
    
    Returns
    -------
    np.array
        Array of selected nodes
    '''
    G = G.copy()
    _validate_n_subs(n_subs, G.number_of_nodes())

    if any(chain <= 0 for chain in chain_lengths):
        raise ValueError('Chain lengths must be positive')
    if n_subs == 0:
        return np.array([], dtype=int)

    if n_subs != sum(chain_lengths):
        raise ValueError('Sum of chain lengths should be equal to n_subs')

    # sort chains from long to short
    sorted_chain_lengths = sorted(chain_lengths, reverse=True)

    al_subs = []

    for chain in sorted_chain_lengths:
        
        if G.number_of_nodes() == 0:
            raise ValueError('Graph is empty')

        # select random node
        node_idx = np.random.choice(list(G.nodes))
        al_subs.append(node_idx)
        chn_len = 1

        while chn_len < chain:
            
            
            neighbours = set(G.neighbors(node_idx))
            
            if len(neighbours) == 0:
                raise ValueError('No neighbours left')
            
            # delete node from graph
            G.remove_node(node_idx)
            
            # select random neighbour
            node_idx = np.random.choice(list(neighbours))
            al_subs.append(node_idx)
            chn_len += 1

            # remove remaining neighbours from the graph
            neighbours = neighbours.difference([node_idx])
            G.remove_nodes_from(neighbours)
        
        # remove neighbours of the last node from the graph
        
        neighbours = set(G.neighbors(node_idx))
        G.remove_nodes_from(neighbours)
        G.remove_node(node_idx)
    
    return np.array(al_subs)


def multi_clusters(G : nx.Graph, n_subs : int, cluster_sizes : List[int], make_space : bool = False, *args, **kwargs):
    '''
    Selects nodes in multiple clusters of specified sizes

    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select
    cluster_sizes : List[int]
        List of cluster sizes to select
        Sum of cluster sizes should be equal to n_subs
    make_space : bool, optional
        If True, clusters cannot be connected, default is False    
    
    Returns
    -------
    np.array
        Array of selected nodes
    '''
        
    G = G.copy()
    _validate_n_subs(n_subs, G.number_of_nodes())

    if any(cluster <= 0 for cluster in cluster_sizes):
        raise ValueError('Cluster sizes must be positive')
    if n_subs == 0:
        return np.array([], dtype=int)

    if n_subs != sum(cluster_sizes):
        raise ValueError('Sum of cluster sizes should be equal to n_subs')
    
    al_subs = []

    for cluster in cluster_sizes:
        
        new_al_subs = clusters(G, cluster, *args, **kwargs)
        
        if make_space:
            to_be_removed = set()
            # remove remaining neighbours of the selected nodes
            for node in new_al_subs:
                neighbours = set(G.neighbors(node))
                to_be_removed = to_be_removed.union(neighbours)
        
            to_be_removed = to_be_removed.difference(new_al_subs)
            G.remove_nodes_from(to_be_removed)

        # remove selected nodes from the graph
        G.remove_nodes_from(new_al_subs)

        al_subs.extend(new_al_subs)

    return np.array(al_subs)
        


def maximize_entropy(G : nx.Graph, n_subs : int, stochastic : bool = False, scaling : float = 1.0, *args, **kwargs):	
    '''
    Selects nodes to maximize the entropy of the selected nodes

    Parameters
    ----------
    G : nx.Graph
        Graph to select nodes from
    n_subs : int
        Number of nodes to select
    stochastic : bool, optional
        If True, use softmax on the distances to select nodes
    scaling : float, optional
        Scaling factor for the stochastic method
    
    Returns
    -------
    np.array
        Array of selected nodes
    '''
    nodes = tuple(G.nodes())
    n_nodes = len(nodes)

    _validate_n_subs(n_subs, n_nodes)
    if n_subs <= 0:
        return np.array([], dtype=int)

    node_array = np.array(nodes, dtype=object)
    selected_mask = np.zeros(n_nodes, dtype=bool)
    distance_sums = np.zeros(n_nodes, dtype=float)

    selected_idx = np.random.randint(n_nodes)
    selected_mask[selected_idx] = True
    selected_nodes = [node_array[selected_idx]]

    distance_sums += _distance_vector(G, nodes, selected_nodes[0])

    while len(selected_nodes) < n_subs:
        candidate_indices = np.flatnonzero(~selected_mask)
        candidate_scores = distance_sums[candidate_indices] / len(selected_nodes)

        if stochastic:
            logits = scaling * candidate_scores
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            selected_idx = np.random.choice(candidate_indices, p=probs)
        else:
            selected_idx = candidate_indices[np.argmax(candidate_scores)]

        selected_mask[selected_idx] = True
        selected_node = node_array[selected_idx]
        selected_nodes.append(selected_node)

        distance_sums += _distance_vector(G, nodes, selected_node)

    return np.array(selected_nodes)