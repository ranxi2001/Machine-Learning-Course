def solve_eq(a1,b1,c1,a2,b2,c2):
    det = a1*b2-a2*b1 # calculate the determinant
    if det == 0: # if determinant is zero, no solution exists
        return None
    else:
        # calculate the x and y values using Cramer's rule
        x = (c1*b2-c2*b1)/det
        y = (a1*c2-a2*c1)/det
        return x, y # return the solution as a tuple
print(solve_eq(19,-7,1,7,-21,-1))