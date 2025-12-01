"""
Find the numbers in the boxes

   15, 19, 21, 18
  |  |    |   |   |
[]   []  []  []  []
     |    |   |
        [30]

"""


def sympy_solve():
    import sympy as sp

    A,B,C,D,E = sp.symbols('A B C D E')

    sol = sp.solve([A+B-15,B+C-19,C+D-21,D+E-18,B+C+D-30],[A,B,C,D,E])

    print(sol)



def self_solve():
    A=0
    B=0
    C=0
    D=0
    E=0
    pairs=[[A,B,15], [B,C,19], [C,D,21], [D,E,18], [B,C,D,30]]
    A+B=15
    B+C=19
    C+D=21
    D+E=18
    B+C+D=30
    


        



        
        









    
    
    pass

if __name__ == "__main__":
    sympy_solve()
    self_solve()




