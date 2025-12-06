# # Solver for Quadratic Equations

# ###Given Example:
# # import math
# import math
# # define function
# def calc(a, b, c):
#     # check if a is zero
#     if a == 0:
#         return None
#     d = b * b - 4 * a * c
#     if d < 0:
#         return []
#     s = math.sqrt(d)
#     x1 = (-b + s) / (2 * a)
#     x2 = (-b - s) / (2 * a)
#     return [x1, x2]

import math
#My Example
def quadratic_solver(a:int, b:int, c:int):
    """
    Parameters:
    a (int): Coefficient of x^2
    b (int): Coefficient of x
    c (int): Constant term

    Example:
    ax^2 + bx + c = 0

    quadratic_solver(1,2,3)
    before_rooting= 2 * 2 - 4 * 1 * 3 = -8
    rooting=math.sqrt(before_rooting) = 

    Returns:
    Returns a List of the two solutions to the quadratic equation, or an empty list if there are no real solutions.

    """
    try:

        #Checks if the Coefficient of x^2 is zero, if it is, returns None, 
        # because the equation is not quadratic, and therefore not possible
        if a == 0:
            return None
        #Combines the numbers before rooting
        before_rooting= b * b - 4 * a * c


        rooting=math.sqrt(before_rooting)
        
        #Finds the 2 solutions from the quadratic equation, as it'll always give a positive and negative root
        solution_1 = (-b + rooting) / (2 * a)
        solution_2 = (-b - rooting) / (2 * a)

        return [solution_1, solution_2]
    
    except ValueError:
        print("Problem Occured, Please Try Again")
        return []







    pass
# if __name__ == "__main__":
#     print(calc(1, -3, 2))  # Expected output: [2.0, 1.0]
#     print(calc(1, 2, 1))   # Expected output: [-1.0, -1.0]
#     print(calc(1, 0, 1))   # Expected output: []
#     print(calc(0, 2, 1))   # Expected output: None

