from sympy import Matrix, symbols, solve, Eq
import numpy as np

# Define symbolic variable
alpha = symbols('alpha')

def dual_feasibility(alpha):
    # Create the matrices
    v1 = Matrix([[2, 1-alpha]])  # 1x2 matrix
    m1 = Matrix([[2/5, -1/5],    # 2x2 matrix
                [-1/5, 3/5]])
    m2 = Matrix([[2-alpha, 1, 0],   # 2x3 matrix
                [1, 0, 1]])

    # Create the vector to subtract
    v2 = Matrix([[1, 0, 0]])

    # Perform the multiplication and subtraction
    result = v1 * m1 * m2 - v2

    return result

# Solve each element for alpha such that the element equals zero
result = dual_feasibility(alpha)
print(f"\nResult: {result}")
solutions = []
for elem in result:
    sol = solve(elem, alpha)
    solutions.append(sol)

print(f"\nSolutions for alpha: {solutions}\n")

flattened_matrix = [elem for row in solutions for elem in row]

# Find the minimum value
alpha = max(flattened_matrix)

print(f"\Max alpha: {alpha}\n")

alpha_val = alpha + 1
if min(dual_feasibility(alpha_val)) >= 0:
    print(f"Solution is not optimal for less than {alpha}\n")
else:
    print(f"Solution is optimal for less than {alpha}\n")


        



        

        
        
            

        
        
