#Simple garbage collection example
import gc

input=input("Enter a string of words:").split(" ")
temp_list = []
for word in input:
    temp_list.append(word)
    print(f"Added '{word}' to list. Current list size: {len(temp_list)}")

list=temp_list.copy()  # Copy to another list to retain data
del temp_list  # Delete the temporary list
collected=gc.collect()  # Manually triggers garbage collection
print(f"Garbage collector: collected {collected} objects.")
print("Final list of words:", list)

