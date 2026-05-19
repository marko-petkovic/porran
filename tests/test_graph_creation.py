import warnings

import networkx as nx

from porran.graph_creation import check_graph


def test_check_graph_warns_for_empty_graph_without_crashing() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_graph(nx.Graph())

    messages = {str(warning.message) for warning in caught}
    assert "Graph has no edges" in messages
    assert "Graph is empty" in messages