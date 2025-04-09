# weighted_graph.py

from typing import TypeVar, Generic, List, Tuple, Optional
from graph import Graph
from weighted_edge import WeightedEdge


V = TypeVar('V') # type of the vertices in the graph


class WeightedGraph(Generic[V], Graph[V]):
    """ A collection of vertices and weighted edges, represented with adjacency list. """

    def __init__(self, vertices: Optional[List[V]] = None) -> None:
        if vertices is None:
            vertices = []
        self._vertices: List[V] = vertices
        self._edges: List[List[WeightedEdge]] = [[] for _ in vertices]

    def add_edge_by_indices(self, u: int, v: int, weight: float) -> None:
        """ Add an edge with the specified weight, between vertices at the specified indices. """
        edge: WeightedEdge = WeightedEdge(u, v, weight)
        self.add_edge(edge) # call superclass's

    def add_edge_by_vertices(self, first: V, second: V, weight: float) -> None:
        """ Add an edge with the specified weight, between the specified pair of vertices. """
        u: int = self._vertices.index(first)
        v: int = self._vertices.index(second)
        self.add_edge_by_indices(u, v, weight)

    def neighbors_for_index_with_weights(self, index: int) -> List[Tuple[V, float]]:
        """ Find the vertices that are adjcent from a vertex at the specified index. Return tuples of such vertices and weights of the corresponding edges. """
        distance_tuples: List[Tuple[V, float]] = []
        for edge in self.edges_for_index(index):
            distance_tuples.append((self.get_vertex_at(edge.v), edge.weight))
        return distance_tuples

    def __str__(self) -> str:
        desc: str = ""
        for i in range(self.vertex_count):
            desc += f"{self.get_vertex_at(i)} -> {self.neighbors_for_index_with_weights(i)}\n"
        return desc


if __name__ == "__main__":
    # test basic Graph construction
    graph: WeightedGraph[str] = WeightedGraph(["London", "Manchester", "Edinburgh", "Cardiff", "Birmingham"])
    graph.add_edge_by_vertices("London", "Manchester", 1000)
    graph.add_edge_by_vertices("London", "Edinburgh", 50)
    graph.add_edge_by_vertices("London", "Cardiff",1250)
    graph.add_edge_by_vertices("London", "Birmingham", 575)
    graph.add_edge_by_vertices("Edinburgh", "Manchester", 672)
    graph.add_edge_by_vertices("Edinburgh", "Birmingham", 652)
    print(graph)
    print(graph.vertex_count)
    print(graph.edge_count)
