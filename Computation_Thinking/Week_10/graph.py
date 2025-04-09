# graph.py

from typing import TypeVar, Generic, List, Optional
from edge import Edge


V = TypeVar('V') # type of the vertices in the graph (for typehinting purposes)


class Graph(Generic[V]):
    """ A collection of vertices and unweighted edges, represented with adjacency list. Herein, an undirected edge is represented with two directed edges with reversed direction. """

    def __init__(self, vertices: Optional[List[V]] = None) -> None:
        if vertices is None:
            vertices = []
        self._vertices: List[V] = vertices
        self._edges: List[List[Edge]] = [[] for _ in vertices]

    @property
    def vertex_count(self) -> int:
        """ Return total number of vertices. """
        return len(self._vertices) 

    @property
    def edge_count(self) -> int:
        """ Return total number of edges. """
        return sum(map(len, self._edges)) 

    def add_vertex(self, vertex: V) -> int:
        """ Add the specified vertex to this graph and return its index. """
        self._vertices.append(vertex)
        self._edges.append([]) # add empty list for containing edges
        return self.vertex_count - 1 # return index of added vertex

    def add_edge(self, edge: Edge) -> None:
        """ Add the specified edge and its reversed edge to this graph. """
        self._edges[edge.u].append(edge)
        self._edges[edge.v].append(edge.get_reversed_edge())

    def add_edge_by_indices(self, u: int, v: int) -> None:
        """ Add an edge between vertices at the specified indices. """
        edge: Edge = Edge(u, v)
        self.add_edge(edge)

    def add_edge_by_vertices(self, first: V, second: V) -> None:
        """ Add an edge between the specified pair of vertices. """
        u: int = self._vertices.index(first)
        v: int = self._vertices.index(second)
        self.add_edge_by_indices(u, v)

    def get_vertex_at(self, index: int) -> V:
        """ Return the vertex at the specified index. """
        return self._vertices[index]

    def index_of(self, vertex: V) -> int:
        """ Find the index of the specified vertex. """
        return self._vertices.index(vertex)

    def neighbors_for_index(self, index: int) -> List[V]:
        """ Find the vertices that are adjcent from a vertex at the specified index. """
        return list(map(self.get_vertex_at, [e.v for e in self._edges[index]]))

    def neighbors_for_vertex(self, vertex: V) -> List[V]:
        """ Find the vertices that are adjcent from the specified vertex. """
        return self.neighbors_for_index(self.index_of(vertex))

    def edges_for_index(self, index: int) -> List[Edge]:
        """ Return all edges that are outward from a vertex at the specified index. """
        return self._edges[index]

    def edges_for_vertex(self, vertex: V) -> List[Edge]:
        """ Return all edges that are outward from the specified vertex. """
        return self.edges_for_index(self.index_of(vertex))

    def __str__(self) -> str:
        desc: str = ""
        for i in range(self.vertex_count):
            desc += f"{self.get_vertex_at(i)} -> {self.neighbors_for_index(i)}\n"
        return desc


if __name__ == "__main__":
    # test basic Graph construction
    graph: Graph[str] = Graph(["London", "Manchester", "Edinburgh", "Cardiff", "Birmingham"])
    graph.add_edge_by_vertices("London", "Manchester")
    graph.add_edge_by_vertices("London", "Edinburgh")
    graph.add_edge_by_vertices("London", "Cardiff")
    graph.add_edge_by_vertices("London", "Birmingham")
    graph.add_edge_by_vertices("Edinburgh", "Manchester")
    graph.add_edge_by_vertices("Edinburgh", "Birmingham")
    print(graph)
    print(graph.vertex_count)
    print(graph.edge_count)
