class LinkedList:
    """ An abstract class for linked-lists. """

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
        raise NotImplementedError
