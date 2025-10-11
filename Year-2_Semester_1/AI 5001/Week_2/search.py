"""
Implementation of various search algorithms for problem solving.

Each algorithm is implemented to work with the Problem class that defines
the problem space, states, actions, and goal tests.

The Node class represents states in the search tree/graph and provides
methods for expanding nodes, tracing solutions, and comparing states.

Code is adapted from AIMA Python code (https://github.com/aimacode/aima-python) for the purpose
of educational demonstration. The original code is licensed under the MIT License, and all copyright
belongs to the original authors. The MIT License is reproduced below.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
from collections import deque
from typing import Callable, List
from heapq import heapify, heappop, heappush
from numbers import Number  # For type hint/checking
from problems import Problem, EightPuzzle, ThreePuzzle, Generate_Tuple, FifteenPuzzle


class Node:
    """
    A node in a search tree.

    Contains a pointer to the parent (the node that this is a successor of)
    and to the actual state for this node.

    You should not need to subclass this class.
    """

    def __init__(self, state, parent_node: "Node" = None, action=None, path_cost=0):
        self.state = state
        self.parent_node = parent_node
        self.action = action
        self.path_cost = path_cost
        if parent_node:
            self.depth = parent_node.depth + 1
        else:
            self.depth = 0

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node: "Node"):
        return self.state < node.state

    def expand(self, problem: Problem) -> List["Node"]:
        """List the nodes reachable in one step from this node."""
        return [
            self.child_node(problem, action) for action in problem.actions(self.state)
        ]

    def child_node(self, problem: Problem, action) -> "Node":
        next_state = problem.result(self.state, action)
        next_node = Node(
            next_state,
            self,
            action,
            problem.path_cost(self.path_cost, self.state, action, next_state),
        )
        return next_node

    def solution(self) -> list:
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self) -> List["Node"]:
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent_node
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


def breadth_first_tree_search(problem: Problem) -> Node:
    """
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = deque([Node(problem.initial_state)])  # FIFO queue

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_tree_search(problem: Problem) -> Node:
    """
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial_state)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem: Problem) -> Node:
    frontier = deque([Node(problem.initial_state)])
    explored = set()
    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)

    return None


def depth_first_graph_search(problem: Problem) -> Node:
    """
    Search the deepest nodes in the search tree first.
    """
    frontier = [Node(problem.initial_state)]
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node

        explored.add(node.state)
        frontier.extend(
            child
            for child in node.expand(problem)
            if child.state not in explored and child not in frontier
        )

    return None


def best_first_graph_search(
    problem: Problem, f: Callable[[Node], Number], display=False
) -> Node:
    """
    Search the nodes with the lowest evaluation f scores first.

    You specify the function f(node) that you want to minimize.
    """

    node = Node(problem.initial_state)
    frontier = [(f(node), node)]  # Priority queue with heapq
    explored = set()
    while frontier:
        current_f_score, node = heappop(frontier)
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "states explored;", len(frontier), "states remained")
            return node

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored:
                f_score = f(child)
                in_frontier = False
                for current_f_score, n in frontier:
                    # if it's already in the frontier
                    if n == child:
                        in_frontier = True
                        if current_f_score > f_score:
                            frontier.remove((current_f_score, n))
                            heapify(frontier)
                            heappush(frontier, (f_score, child))
                            break
                # if it's not in the frontier
                if not in_frontier:
                    heappush(frontier, (f_score, child))
    return None


def uniform_cost_search(problem: Problem, display=False) -> Node:
    def path_cost_func(node: Node) -> Number:
        return node.path_cost

    return best_first_graph_search(problem, path_cost_func, display)


def depth_limited_search(problem: Problem, limit: int = 50) -> Node:
    def recursive_dls(node: Node, problem: Problem, limit: int) -> Node | str:
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return "cutoff"
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == "cutoff":
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return "cutoff" if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem: Problem) -> Node:
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != "cutoff":
            return result


def astar_search(problem: Problem, h=None, display=False) -> Node:
    """A* search.

    You need to specify the h heuristic function.
    """

    return best_first_graph_search(problem, lambda n: n.path_cost + h(n.state), display)


if __name__ == "__main__":
    # Example usage:
    type=int(input("""Pick Puzzle Type:
          1. 8-Puzzle
          2. 3-Puzzle
          3. 15-Puzzle
          \n"""))

    match type:
        case 1:
            initial_state = Generate_Tuple(9).generate(9)
            problem = EightPuzzle(initial_state)
        case 2:
            initial_state = Generate_Tuple(4).generate(4)
            problem = ThreePuzzle(initial_state)
        case 3:
            initial_state = Generate_Tuple(16).generate(16)
            problem = FifteenPuzzle(initial_state)
    print("Initial State:")
    print(problem.initial_state)
    print("Goal State:")
    print(problem.goal_state)

    if not problem.check_solvability(problem.initial_state):
        print("The given initial state is not solvable.")
    else:
        print("\nSolving with A* Search (Misplaced Tile Heuristic):")
        solution_node = astar_search(problem, problem.h, display=True)
        if solution_node:
            print("Solution found!")
            print("Actions to reach goal:", solution_node.solution())
            print("Number of moves:", len(solution_node.solution()))
        else:
            print("No solution found.")
