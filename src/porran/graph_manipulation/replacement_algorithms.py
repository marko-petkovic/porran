import numpy as np

import networkx as nx

from typing import List, Optional
from copy import deepcopy

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
    n_nodes = len(G.nodes)
    return np.random.choice(np.arange(n_nodes), n_subs, replace=False)

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
    G = deepcopy(G)
    n_subs -= 1 # to account for the source node

    if node_idx is None:
        node_idx = np.random.choice(list(G.nodes))

    # get first neighbours of node_idx
    neighbours = set(G.neighbors(node_idx))

    if len(neighbours) > n_subs:
        # select n_subs random neighbours
        neighbours = np.random.choice(list(neighbours), n_subs, replace=False)
        return neighbours

    while len(neighbours) < n_subs:
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
        if len(neighbours) + len(added_neighbours) > n_subs:
            added_neighbours = np.random.choice(list(added_neighbours), n_subs - len(neighbours), replace=False)
            neighbours = neighbours.union(added_neighbours)
            break
        else:
            neighbours = neighbours.union(added_neighbours)
    
    return np.array(list(neighbours))



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
    G = deepcopy(G)

    if n_subs != sum(chain_lengths):
        raise ValueError('Sum of chain lengths should be equal to n_subs')

    # sort chains from long to short
    chain_lengths.sort(reverse=True)

    al_subs = []

    for chain in chain_lengths:
        
        if list(G.nodes) == []:
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
    G = deepcopy(G)

    selected_nodes = set()
    remaining_nodes = set(G.nodes())

    # Select a random node to start
    selected_node = np.random.choice(list(remaining_nodes))
    selected_nodes.add(selected_node)
    remaining_nodes.remove(selected_node)

    # Continue until desired number of nodes is selected
    while len(selected_nodes) < n_subs:
        max_avg_distance = -1
        selected_node = None

        distances = np.zeros(len(remaining_nodes))
        x = 0
        # Calculate average distance for each remaining node
        for node in remaining_nodes:
            avg_distance = np.mean([nx.shortest_path_length(G, node, selected_node)
                                    for selected_node in selected_nodes])
            
            distances[x] = avg_distance

            # if we are using the deterministic method
            if not stochastic and avg_distance > max_avg_distance:
                max_avg_distance = avg_distance
                selected_node = node

            x += 1
        
        # if we are using the stochastic method
        if stochastic > 0:
            
            probs = np.exp(scaling * distances)
            probs /= np.sum(probs)
            selected_node = np.random.choice(list(remaining_nodes), p=probs)


        # Add the node with maximum average distance to selected nodes
        selected_nodes.add(selected_node)
        remaining_nodes.remove(selected_node)

    return np.array(list(selected_nodes))