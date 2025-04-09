# edge.py

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Edge:
    """ Unweighted directed edge """
    u: int # the index of "from" vertex
    v: int # the index of "to" vertex

    def get_reversed_edge(self) -> Edge:
        """ Return a new edge with reverse direction. """
        return Edge(self.v, self.u)
    
    def __str__(self) -> str:
        return f"{self.u} -> {self.v}"


if __name__ == "__main__":
    edge = Edge(0, 1)
    print(edge)
    print(edge.get_reversed_edge())
    