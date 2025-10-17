# import ctypes
# libc = ctypes.CDLL(None)
# ptr = libc.malloc(10)
# # Calloc: allocate 5 integers, initialized to 0
# ptr2 = libc.calloc(5, ctypes.sizeof(ctypes.c_int))
# # Realloc: resize the malloc'd memory to 20 bytes
# ptr = libc.realloc(ptr, 20)
# # Free: release memory
# libc.free(ptr)

import ctypes

# On Windows, use the Microsoft C runtime library
libc = ctypes.CDLL("msvcrt.dll")  

# Allocate memory for 5 integers (calloc initializes to 0)
num_elements = 5
arr = libc.calloc(num_elements, ctypes.sizeof(ctypes.c_int))

# Resize memory to hold 10 integers
arr = libc.realloc(arr, 10 * ctypes.sizeof(ctypes.c_int))

# Use the memory (example: write to first element)
int_ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_int))
int_ptr[0] = 42
int_ptr[1] = 99

# Free the allocated memory
libc.free(arr)

print("Memory allocated, modified, resized, and freed successfully!")
