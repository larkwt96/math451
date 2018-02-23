def zeroOut(A, e=1e-10):
    """Don't pass non squares"""
    for r in range(len(A)):
        for c in range(len(A)):
            if abs(A[r,c]) < e:
                A[r,c] = 0
    return A

def pretty_print(A):
    zeroOut(A)
    print('\\\\')
    print('$')
    print('\\left[ {\\begin{array}{'+'c'*len(A)+'}')
    for r in range(len(A)):
        print(A[r,0], end='')
        for c in range(1, len(A)):
            print(' & {}'.format(A[r,c]), end='')
        print(' \\\\')
    print('\\end{array} } \\right]')
    print('$')
    print('\\\\')
