memo = {0: 0, 1: 1}

def fib2(n: int):
    global counter
    if n not in memo:
        counter += 1  
        memo[n] = fib2(n - 1) + fib2(n - 2)
    return memo[n]

numbers = {5, 10, 15, 20, 25, 30, 35, 40}

for num in numbers:
    counter = 0  
    print(f"fib2({num}) = {fib2(num)}, Total calls = {counter}")