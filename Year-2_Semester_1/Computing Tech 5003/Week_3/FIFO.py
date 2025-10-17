# FIFO (First In, First Out) Queue Implementation
class FIFOQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        """Add an item to the end of the queue."""
        self.queue.append(item)

    def dequeue(self):
        """Remove and return the item from the front of the queue. 
        Raises IndexError if the queue is empty."""
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self.queue.pop(0)

    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.queue) == 0

    def size(self):
        """Return the number of items in the queue."""
        return len(self.queue)

    def peek(self):
        """Return the item at the front of the queue without removing it.
        Raises IndexError if the queue is empty."""
        if self.is_empty():
            raise IndexError("peek from an empty queue")
        return self.queue[0]
    
# Example usage, Seeing next patient in the queue
if __name__ == "__main__":
    patient_queue = FIFOQueue()
    
    # Enqueue some items
    patient_queue.enqueue("Patient 1")
    patient_queue.enqueue("Patient 2")
    patient_queue.enqueue("Patient 3")

    print("Next patient to be seen:", patient_queue.peek())  # Output: Patient 1

    while not patient_queue.is_empty():
        patient = patient_queue.dequeue()
        print(f"{patient} is being seen.")
        print(f"Patients left in queue: {patient_queue.size()}")
    
    # Trying to dequeue from an empty queue will raise an error
    try:
        patient_queue.dequeue()
    except IndexError as e:
        print(e)  # Output: dequeue from an empty queue