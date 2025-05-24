
"""
generic_search.py

A collection of generic search algorithms for solving problems.
This module provides implementations of fundamental search algorithms
that can be applied to various problem domains.
"""
from __future__ import annotations
from typing import TypeVar, Iterable, Sequence, Generic, List, Callable, Set, Deque, Any, Optional, Protocol
from collections import deque
from dataclasses import dataclass

# Type variable for generic functions and classes
T = TypeVar('T')


def linear_contains(iterable: Iterable[T], key: T) -> bool:
    """
    Linear search algorithm - checks each item in order.
    Time complexity: O(n) where n is the number of items in the iterable.
    
    Args:
        iterable: Any iterable containing comparable items
        key: The item to search for
        
    Returns:
        True if the key is found, False otherwise
    """
    return any(item == key for item in iterable)


class Comparable(Protocol):
    """
    Protocol defining methods needed for comparison operations.
    Any class implementing these methods can be used with functions
    requiring comparable objects.
    """
    def __eq__(self, other: Any) -> bool:
        """Equal to operator"""
        ...

    def __lt__(self: C, other: C) -> bool:
        """Less than operator"""
        ...

    def __gt__(self: C, other: C) -> bool:
        """
        Greater than operator - defaults to not less than and not equal
        Can be overridden for custom implementation
        """
        return (not self < other) and self != other

    def __le__(self: C, other: C) -> bool:
        """
        Less than or equal operator - defaults to less than or equal
        Can be overridden for custom implementation
        """
        return self < other or self == other

    def __ge__(self: C, other: C) -> bool:
        """
        Greater than or equal operator - defaults to not less than
        Can be overridden for custom implementation
        """
        return not self < other


# Type variable for comparable objects
C = TypeVar("C", bound=Comparable)


def binary_contains(sequence: Sequence[C], key: C) -> bool:
    """
    Binary search algorithm - requires sorted sequence.
    Time complexity: O(log n) where n is the length of the sequence.
    
    Args:
        sequence: A sorted sequence of comparable items
        key: The item to search for
        
    Returns:
        True if the key is found, False otherwise
    
    Raises:
        ValueError: If the sequence is empty
    """
    if not sequence:
        raise ValueError("Cannot search in an empty sequence")
        
    low: int = 0
    high: int = len(sequence) - 1
    
    while low <= high:  # while there is still a search space
        mid: int = low + (high - low) // 2  # Avoid potential overflow
        
        if sequence[mid] < key:  # If the key is in the upper half
            low = mid + 1  # Adjust search space to upper half
        elif sequence[mid] > key:  # If the key is in the lower half 
            high = mid - 1  # Adjust search space to lower half
        else:  # If the key is found
            return True
            
    return False  # If the key is not found after exhausting the search space


class Queue(Generic[T]):
    """
    FIFO (First-In-First-Out) data structure.
    Used in breadth-first search algorithm.
    """
    def __init__(self) -> None:
        """Initialize an empty queue using a deque as the underlying container"""
        self._container: Deque[T] = deque()

    @property
    def empty(self) -> bool:
        """Check if the queue is empty"""
        return len(self._container) == 0

    def push(self, item: T) -> None:
        """Add an item to the back of the queue"""
        self._container.append(item)

    def pop(self) -> T:
        """
        Remove and return the front item from the queue
        
        Raises:
            IndexError: If the queue is empty
        """
        if self.empty:
            raise IndexError("Pop from an empty queue")
        return self._container.popleft()  # FIFO - first item is removed first

    def __repr__(self) -> str:
        """String representation of the queue for debugging"""
        return f"Queue({self._container})"


class Stack(Generic[T]):
    """
    LIFO (Last-In-First-Out) data structure.
    Used in depth-first search algorithm.
    """
    def __init__(self) -> None:
        """Initialize an empty stack using a list as the underlying container"""
        self._container: List[T] = []

    @property
    def empty(self) -> bool:
        """Check if the stack is empty"""
        return len(self._container) == 0

    def push(self, item: T) -> None:
        """Add an item to the top of the stack"""
        self._container.append(item)

    def pop(self) -> T:
        """
        Remove and return the top item from the stack
        
        Raises:
            IndexError: If the stack is empty
        """
        if self.empty:
            raise IndexError("Pop from an empty stack")
        return self._container.pop()  # LIFO - last item is removed first

    def __repr__(self) -> str:
        """String representation of the stack for debugging"""
        return f"Stack({self._container})"


@dataclass
class Node(Generic[T]):
    """
    A node in a search tree/graph.
    Contains the current state, reference to parent node, and cost.
    """
    state: T
    parent: Optional[Node[T]] = None
    cost: float = 0.0
    
    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on state only.
        This allows comparing nodes regardless of their path or cost.
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state


def node_to_path(node: Node[T]) -> List[T]:
    """
    Converts a node with parent references to a path from start to goal.
    
    Args:
        node: The final node, typically containing the goal state
        
    Returns:
        A list of states forming a path from the start to the goal
    """
    if node is None:
        return []
        
    path: List[T] = []
    current: Optional[Node[T]] = node
    
    # Work backwards from end to front
    while current is not None:
        path.append(current.state)
        current = current.parent
        
    path.reverse()  # Reverse to get path from start to goal
    return path


def dfs(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]]) -> Optional[Node[T]]:
    """
    Depth-First Search algorithm.
    Explores as far as possible along each branch before backtracking.
    
    Args:
        initial: The initial state
        goal_test: A function that determines if a state is the goal
        successors: A function that returns all possible next states
        
    Returns:
        A solution node containing the goal state, or None if no solution is found
    """
    # frontier is where we've yet to go
    frontier: Stack[Node[T]] = Stack()
    frontier.push(Node(initial))
    
    # explored is where we've been - used to avoid cycles
    explored: Set[T] = {initial}

    # keep going while there is more to explore
    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        
        # if we found the goal, we're done
        if goal_test(current_state):
            return current_node
            
        # check where we can go next and haven't explored
        for child in successors(current_state):
            if child in explored:  # skip children we already explored
                continue
                
            explored.add(child)  # mark child as explored
            frontier.push(Node(child, current_node))  # add to frontier
            
    return None  # went through everything and never found goal


def bfs(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]]) -> Optional[Node[T]]:
    """
    Breadth-First Search algorithm.
    Explores all neighbor nodes at the present depth before moving to nodes at the next depth.
    Guarantees the shortest path in terms of number of steps.
    
    Args:
        initial: The initial state
        goal_test: A function that determines if a state is the goal
        successors: A function that returns all possible next states
        
    Returns:
        A solution node containing the goal state, or None if no solution is found
    """
    # frontier is where we've yet to go
    frontier: Queue[Node[T]] = Queue()
    frontier.push(Node(initial))
    
    # explored is where we've been - used to avoid cycles
    explored: Set[T] = {initial}

    # keep going while there is more to explore
    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        
        # if we found the goal, we're done
        if goal_test(current_state):
            return current_node
            
        # check where we can go next and haven't explored
        for child in successors(current_state):
            if child in explored:  # skip children we already explored
                continue
                
            explored.add(child)  # mark child as explored
            frontier.push(Node(child, current_node))  # add to frontier
            
    return None  # went through everything and never found goal


# Demo code for linear and binary search
if __name__ == "__main__":
    print("Testing linear search:")
    print(f"linear_contains([1, 5, 15, 15, 15, 15, 20], 5) = {linear_contains([1, 5, 15, 15, 15, 15, 20], 5)}")  # True
    print(f"linear_contains([1, 5, 15, 15, 15, 15, 20], 42) = {linear_contains([1, 5, 15, 15, 15, 15, 20], 42)}")  # False
    
    print("\nTesting binary search:")
    print(f"binary_contains(['a', 'd', 'e', 'f', 'z'], 'f') = {binary_contains(['a', 'd', 'e', 'f', 'z'], 'f')}")  # True
    print(f"binary_contains(['john', 'mark', 'ronald', 'sarah'], 'sheila') = {binary_contains(['john', 'mark', 'ronald', 'sarah'], 'sheila')}")  # False
    
    # Show error handling
    try:
        binary_contains([], 42)
    except ValueError as e:
        print(f"\nHandled error: {e}")