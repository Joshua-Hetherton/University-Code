counter=0


def fib1(n):
    global counter
    counter+=1
    if n<=1:
        return n
    return fib1(n-1)+fib1(n-2)

print(f"{fib1(5)}, {counter}")
counter=0
print(f"{fib1(10)}, {counter}")
counter=0
print(f"{fib1(15)}, {counter}")
counter=0
print(f"{fib1(20)}, {counter}")
counter=0
print(f"{fib1(25)}, {counter}")
counter=0
print(f"{fib1(30)}, {counter}")
counter=0
print(f"{fib1(35)}, {counter}")
counter=0
print(f"{fib1(40)}, {counter}")
counter=0




