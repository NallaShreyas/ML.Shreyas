def power(A, m):
    result = A
    for i in range(m-1):
        result = multiply(result, A)
    return result

def multiply(A, B):
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_input():
    n = int(input("Enter the size of the square matrix: "))
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(input(f"Enter element [{i+1}][{j+1}]: ")))
        A.append(row)
    return A

def printing(A):
    for row in A:
        print(row)

# matrix input
A = matrix_input()
print("Matrix A:")
printing(A)

# power input
m = int(input("Enter the power to which the matrix should be raised: "))

# output
result = power(A, m)
print(f"A^{m}:")
printing(result)