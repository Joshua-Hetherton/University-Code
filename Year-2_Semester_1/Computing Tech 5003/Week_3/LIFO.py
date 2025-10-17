# LIFO (Last In, First Out) Stack Implementation
class LIFOStack:
    def push(self, item):
        """Adds an item to the Stack(to the top)"""
        self.stack.append(item)
    def pop(self):
        """Takes an item from the stack and returns it."""
        if self.is_empty():
            raise IndexError("pop from an empty stack")
        return self.stack.pop()
    def peek(self):
        """Returns the top item of the stack without removing it."""
        if self.is_empty():
            raise IndexError("peek from an empty stack")
        return self.stack[-1]
    def is_empty(self):
        """Checks if the stack is empty."""
        return len(self.stack) == 0
    def size(self):
        """Returns the number of items in the stack."""
        return len(self.stack)
    
    def __init__(self):
        self.stack = []


# Example usage, Undo functionality
if __name__ == "__main__":
    input=input("Enter a string of words:").split(" ")
    undo_stack = LIFOStack()
    for word in input:
        undo_stack.push(word)

    print("Reversed words:", end=" ")
    while not undo_stack.is_empty():
        print(undo_stack.pop(), end=" ")
        print(f"\nWords left in stack: {undo_stack.size()}")