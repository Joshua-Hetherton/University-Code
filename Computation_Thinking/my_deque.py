from array import array


class Deque:
    """ An abstract class for deques, that supports insertion and removal at both ends. """

    def addFirst(self, item):
        """ Inserts the specified item at the front of this deque. """
        pass

    def addLast(self, item):
        """ Inserts the specified item at the end of this deque. """
        pass

    def removeFirst(self):
        """ Removes and returns the first item of this deque. Raises an IndexError if this deque is empty. """
        pass

    def removeLast(self):
        """ Removes and returns the last item of this deque. Raises an IndexError if this deque is empty. """
        pass

    def size(self) -> int:
        """ Returns the number of items in this deque.. """
        pass


class Node:
    """ A wrapper/node of deque's item with two pointers. """

    def __init__(self, item):
        self.item = item
        self.left_pointer: Node = None  # points to the previous node
        self.right_pointer: Node = None  # points to the next node


class LinkedDeque(Deque):
    """ An unbounded-capacity deque based on linked nodes. """

    def __init__(self):
        self.__head: Node = None  # points to front of deque
        self.__tail: Node = None  # points to rear of deque
        self.__size: int = 0  # counts total number of items

    def addFirst(self, item):
        """ Inserts the specified item at the front of this deque. """
        raise NotImplementedError  # TODO

    def addLast(self, item):
        """ Inserts the specified item at the end of this deque. """
        raise NotImplementedError  # TODO

    def removeFirst(self):
        """ Removes and returns the first item of this deque. Raises an IndexError if this deque is empty. """
        raise NotImplementedError  # TODO

    def removeLast(self):
        """ Removes and returns the last item of this deque. Raises an IndexError if this deque is empty. """
        raise NotImplementedError  # TODO

    def size(self) -> int:
        return self.__size


class ArrayDeque(Deque):
    """ Resizable-array implementation of Deque for Unicode characters. Array deques have no capacity restrictions; they grow as necessary to support usage. """

    def __init__(self):
        self.__items: array = array('u')

    def addFirst(self, item):
        """ Inserts the specified item at the front of this deque. """
        raise NotImplementedError  # TODO

    def addLast(self, item):
        """ Inserts the specified item at the end of this deque. """
        raise NotImplementedError  # TODO

    def removeFirst(self):
        """ Removes and returns the first item of this deque. Raises an IndexError if this deque is empty. """
        raise NotImplementedError  # TODO

    def removeLast(self):
        """ Removes and returns the last item of this deque. Raises an IndexError if this deque is empty. """
        raise NotImplementedError  # TODO

    def size(self) -> int:
        return len(self.__items)
