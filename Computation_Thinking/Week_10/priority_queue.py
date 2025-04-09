# priority_queue.py

from typing import TypeVar, Generic, List
from heapq import heappush, heappop


T = TypeVar('T')


class PriorityQueue(Generic[T]):
    """ A priority queue, implemented with min-heap algorithm. """

    def __init__(self) -> None:
        self._container: List[T] = []

    @property
    def empty(self) -> bool:
        return not self._container  # not is true for empty container

    def push(self, item: T) -> None:
        """ Push the specified item into this priority queue. """
        heappush(self._container, item) 

    def pop(self) -> T:
        """ Pop and return the smallest item from this priority queue. Raise IndexError is raised if the queue is empty. """
        return heappop(self._container) 

    def __repr__(self) -> str:
        return repr(self._container)
    

if __name__ == "__main__":
    pq: PriorityQueue[str] = PriorityQueue()
    print(pq.empty)
    pq.push("Bat")
    pq.push("Cat")
    pq.push("Ant")
    pq.push("Amoeba")
    print(pq)
    