# problems.py

from graph import Graph
from weighted_graph import WeightedGraph


CITIES = ["Seattle", "San Francisco", "Los Angeles", "Riverside", "Phoenix", "Chicago", "Boston", "New York", "Atlanta", "Miami", "Dallas", "Houston", "Detroit", "Philadelphia", "Washington"]


def problem1() -> Graph[str]:
    graph: Graph[str] = Graph(CITIES)
    graph.add_edge_by_vertices(CITIES[0], CITIES[5])
    graph.add_edge_by_vertices(CITIES[0], CITIES[1])
    graph.add_edge_by_vertices(CITIES[1], CITIES[3])
    graph.add_edge_by_vertices(CITIES[1], CITIES[2])
    graph.add_edge_by_vertices(CITIES[2], CITIES[3])
    graph.add_edge_by_vertices(CITIES[2], CITIES[4])
    graph.add_edge_by_vertices(CITIES[3], CITIES[4])
    graph.add_edge_by_vertices(CITIES[3], CITIES[5])
    graph.add_edge_by_vertices(CITIES[4], CITIES[10])
    graph.add_edge_by_vertices(CITIES[4], CITIES[11])
    graph.add_edge_by_vertices(CITIES[10], CITIES[5])
    graph.add_edge_by_vertices(CITIES[10], CITIES[8])
    graph.add_edge_by_vertices(CITIES[10], CITIES[11])
    graph.add_edge_by_vertices(CITIES[11], CITIES[8])
    graph.add_edge_by_vertices(CITIES[11], CITIES[9])
    graph.add_edge_by_vertices(CITIES[8], CITIES[5])
    graph.add_edge_by_vertices(CITIES[8], CITIES[14])
    graph.add_edge_by_vertices(CITIES[8], CITIES[9])
    graph.add_edge_by_vertices(CITIES[9], CITIES[14])
    graph.add_edge_by_vertices(CITIES[5], CITIES[12])
    graph.add_edge_by_vertices(CITIES[12], CITIES[6])
    graph.add_edge_by_vertices(CITIES[12], CITIES[14])
    graph.add_edge_by_vertices(CITIES[12], CITIES[7])
    graph.add_edge_by_vertices(CITIES[6], CITIES[7])
    graph.add_edge_by_vertices(CITIES[7], CITIES[13])
    graph.add_edge_by_vertices(CITIES[13], CITIES[14])
    return graph


def problem2() -> WeightedGraph[str]:
    graph: WeightedGraph[str] = WeightedGraph(CITIES)
    graph.add_edge_by_vertices(CITIES[0], CITIES[5], 1737)
    graph.add_edge_by_vertices(CITIES[0], CITIES[1], 678)
    graph.add_edge_by_vertices(CITIES[1], CITIES[3], 386)
    graph.add_edge_by_vertices(CITIES[1], CITIES[2], 348)
    graph.add_edge_by_vertices(CITIES[2], CITIES[3], 50)
    graph.add_edge_by_vertices(CITIES[2], CITIES[4], 357)
    graph.add_edge_by_vertices(CITIES[3], CITIES[4], 307)
    graph.add_edge_by_vertices(CITIES[3], CITIES[5], 1704)
    graph.add_edge_by_vertices(CITIES[4], CITIES[10], 887)
    graph.add_edge_by_vertices(CITIES[4], CITIES[11], 1015)
    graph.add_edge_by_vertices(CITIES[10], CITIES[5], 805)
    graph.add_edge_by_vertices(CITIES[10], CITIES[8], 721)
    graph.add_edge_by_vertices(CITIES[10], CITIES[11], 225)
    graph.add_edge_by_vertices(CITIES[11], CITIES[8], 702)
    graph.add_edge_by_vertices(CITIES[11], CITIES[9], 968)
    graph.add_edge_by_vertices(CITIES[8], CITIES[5], 588)
    graph.add_edge_by_vertices(CITIES[8], CITIES[14], 543)
    graph.add_edge_by_vertices(CITIES[8], CITIES[9], 604)
    graph.add_edge_by_vertices(CITIES[9], CITIES[14], 923)
    graph.add_edge_by_vertices(CITIES[5], CITIES[12], 238)
    graph.add_edge_by_vertices(CITIES[12], CITIES[6], 613)
    graph.add_edge_by_vertices(CITIES[12], CITIES[14], 396)
    graph.add_edge_by_vertices(CITIES[12], CITIES[7], 482)
    graph.add_edge_by_vertices(CITIES[6], CITIES[7], 190)
    graph.add_edge_by_vertices(CITIES[7], CITIES[13], 81)
    graph.add_edge_by_vertices(CITIES[13], CITIES[14], 123)
    return graph


if __name__ == "__main__":
    problem = problem1()
    print(problem)
