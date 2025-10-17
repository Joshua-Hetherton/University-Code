# FILO (First In, Last Out) Stack Implementation
class FILOStack:
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

# Example usage, Reversing a string

if __name__ == "__main__":
    input=input("Enter a string of words:").split(" ")
    reverse_stack = FILOStack()
    for word in input:
        reverse_stack.push(word)


    output_list = []
    while not reverse_stack.is_empty():
        output = reverse_stack.pop()
        output_list.append(output)
    print("Reversed words:", " ".join(output_list))