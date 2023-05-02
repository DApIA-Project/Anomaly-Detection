def mult(x, dir, f=2):
    if (x == 0): # avoid the problem of 0**0
        if (dir > 0):
            return 1
        else:
            return -1

    return x * (float(f) ** dir)

def add(x, dir, f=1):
    return x + dir * f
def swap(x, dir, f=1):
    return not(x)