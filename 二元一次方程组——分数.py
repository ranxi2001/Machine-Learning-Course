from fractions import Fraction

def solve_eq(a1,b1,c1,a2,b2,c2):
    det = a1*b2-a2*b1
    if det == 0:
        return None
    else:
        x = Fraction((c1*b2-c2*b1), det)
        y = Fraction((a1*c2-a2*c1), det)
        return x, y

# 假设要解决的方程组是 2x + 3y = 5，3x + 5y = 8
x, y = solve_eq(19,-7,1,7,-21,-1)
print("x =", x)
print("y =", y)
