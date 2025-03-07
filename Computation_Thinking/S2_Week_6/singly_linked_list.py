from linked_list import LinkedList


class Node:
    """ A singly-linked node. """
    
    def __init__(self, data):
        self.data = data
        self.pointer: Node = None  # Pointer to the next node


class SinglyLinkedList:
    """ A singly-linked list. """

    def __init__(self):
        """ Create an empty list. """
        self.head: Node = None
    
    def insert(self, index: int, item):
        """ Inserts the specified item at the specified index. """
        if index < 0 or index > self.size():
            raise IndexError("Index out of bounds")
        
        new_node = Node(item)
        
        if index == 0:
            # Insert at the head
            new_node.pointer = self.head
            self.head = new_node
        else:
            # Traverse to the node before the desired position
            count = 0
            current = self.head
            while count < index - 1:
                current = current.pointer
                count += 1
            
            # Insert the new node
            new_node.pointer = current.pointer
            current.pointer = new_node

    def delete(self, index: int):
        """ Removes the item at the specified index. """
        if index < 0 or index >= self.size():
            raise IndexError("Index out of bounds")
        
        if index == 0:
            # Remove the head node
            self.head = self.head.pointer
        else:
            # Traverse to the node before the one to delete
            count = 0
            current = self.head
            while count < index - 1:
                current = current.pointer
                count += 1
            
            # Bypass the node to delete
            current.pointer = current.pointer.pointer

    def size(self) -> int:
        """ Returns the number of elements in the list. """
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.pointer
        return count

    def get(self, index: int):
        """ Returns the item at the specified index in this list. """
        if index < 0 or index >= self.size():
            raise IndexError("Index out of bounds")
        
        # Traverse to the desired index
        count = 0
        node = self.head
        while count < index:
            node = node.pointer
            count += 1
        return node.data

# Now the linked list logic is solid! Check your test cases, and they should work fine. If you want, I can help adjust the tests too. 🚀
