from linked_list import LinkedList


class Node:
    """ A doubly-linked node. """

    def __init__(self, data):
        self.data = data
        self.left_pointer: Node = None  # to point to the previous node
        self.right_pointer: Node = None  # to point to the next node


class DoublyLinkedList(LinkedList):
    def __init__(self):
        """ Create an empty list. """

        self.head: Node = None  # indicating that there is nothing at the head of the list

    def insert(self, index: int, item):
        """ Inserts the specified item at the specified index. Shifts the item currently at that position and any subsequent elements to the right (adds one to their indices). """
        raise NotImplementedError

    def delete(self, index: int):
        """ Removes the item at the specified index. Shifts any subsequent elements to the left (subtracts one from their indices). """
        raise NotImplementedError

    def size(self) -> int:
        """ Returns the number of items in this list. """
        raise NotImplementedError

    def get(self, index: int):
        """ Returns the item at the specified index in this list. """

        # Check index
        if index < 0 or index >= self.size():
            raise IndexError

        # Traverse from the head
        count = 0
        node = self.head
        while count < index:
            node = node.right_pointer
            count += 1
        return node.data
