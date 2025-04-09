# weighted_edge.py

from __future__ import annotations
from dataclasses import dataclass
from edge import Edge


@dataclass
class WeightedEdge(Edge):
    """ Weighted directed edge. """
    weight: float

    def get_reversed_edge(self) -> WeightedEdge:
        """ Return a new edge with the same weight but a reversed direction. """
        return WeightedEdge(self.v, self.u, self.weight)

    def __lt__(self, other: WeightedEdge) -> bool:
        """ Dunder method for "<" operator. """
        return self.weight < other.weight

    def __str__(self) -> str:
        return f"{self.u} {self.weight}> {self.v}"


if __name__ == "__main__":
    edge1 = WeightedEdge(0, 1, 1)
    edge2 = WeightedEdge(1, 2, 3)
    print(edge1)
    print(edge1.get_reversed_edge())
    print(edge2)
    print(edge2.get_reversed_edge())
    print(edge2 < edge1)
    print(edge1 < edge2)
