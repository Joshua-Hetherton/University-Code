import math


def pick_present(prices):

    N = len(prices)
    
    if N < 3:
        return "A random box is picked"


    J = int(math.floor(0.3679 * N))
    highest_price = max(prices[:J])

    for i in range(J, N):
        if prices[i] > highest_price:
            return prices[i]
    
    return "None is picked"


prices =[7, 6, 8, 2, 3, 9, 4, 5, 1, 10]
accepted_present = pick_present(prices)

print(f"The accepted present: {accepted_present}")