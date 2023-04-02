import numpy as np
# print(np.__version__)

a = np.array([1, 2, 3], dtype=np.int64)

# print(a)
# print(a.shape)
# print(a.dtype)
# print(a.ndim)
# print(a.size)
# print(a.itemsize)

# print(a[0])
# a[0] = 10
# print(a[0])

# b = a * np.array([2, 0, 2])
# print(b)

l = [1, 2, 3]
a = np.array([1, 2, 3])

# l.append(4)
# print(l)
# a.append(4)
# print(a)

# l = l + [4]
# print(l)
# a = a + np.array([4])
# print(a)

# l = l * 2
# print(l)
# a = a * 2
# print(a)

# a = np.sqrt(a)
# print(a)
# a = np.log(a)
# print(a)

# l1 = [1, 2, 3]
# l2 = [4, 5, 6]
# a1 = np.array(l1)
# a2 = np.array(l2)
# 
# # dot product
# dot = 0
# for i in range(len(l1)):
#     dot += l1[i] * l2[i]
# print(dot)
# 
# dot = np.dot(a1, a2)
# print(dot)
# 
# prd = a1 * a2
# dot = np.sum(prd)
# print(dot)
# 
# dot = (a1 * a2).sum()
# print(dot)
# 
# dot = a1 @ a2
# print(dot)

# from timeit import default_timer as timer
# 
# a = np.random.randn(1000)
# b = np.random.randn(1000)
# 
# A = list(a)
# B = list(b)
# 
# T = 1000
# 
# def dot1():
#     dot = 0
#     for i in range(len(A)):
#         dot += A[i] * B[i]
#     return dot
# 
# def dot2():
#     return np.dot(a, b)
# 
# start = timer()
# for t in range(T):
#     dot1()
# end = timer()
# t1 = end - start
# 
# start = timer()
# for t in range(T):
#     dot2()
# end = timer()
# t2 = end - start
# 
# print(t1)
# print(t2)
# print('ratio: ', t1/t2)

# a = np.array([[1, 2, 6], [3, 4, 8]])
# print(a.shape)
# 
# print(a[0])     # all elements in row 0
# print(a[0][0])  # element at row 0 column 0
# print(a[0, 0])  # element at row 0 column 0
# print(a[:, 0])  # all elements in column 0
# print(a[0, :])  # all elements in row 0

# print(a.T)
# 
# a = np.array([[1, 2], [3, 4]])
# print(np.linalg.inv(a))
# print(np.linalg.det(a))
# print(np.diag(a))
# c = np.diag(a)
# print(np.diag(c))

# a = np.array([[1, 2], [3, 4], [5, 6]])
# print(a)
# 
# print(a[a>2])
# 
# b = np.where(a>2, a, -1)
# print(b)

# a = np.array([10, 19, 30, 41, 50, 61])
# print(a)
# b = [1, 3, 5]
# print(a[b])

# a = np.array([10, 19, 30, 41, 50, 61])
# print(a)
# even = np.argwhere(a%2 == 0).flatten()
# print(even)
# print(a[even])

# a = np.arange(1, 7)
# print(a)
# print(a.shape)
# b = a.reshape((2,3))
# print(b)
# print(b.shape)

# a = np.arange(1, 7)
# print(a)
# print(a.shape)
# b = a[np.newaxis, :]
# print(b)
# print(b.shape)
# b = a[:, np.newaxis]
# print(b)
# print(b.shape)
# c = b.reshape((2,3))
# print(c)
# print(c.shape)

# a = np.array([[1, 2], [3, 4]])
# print(a)
# b = np.array([[5, 6]])
# print(b)
# c = np.concatenate((a, b))
# print(c)
# d = np.concatenate((a, b), axis=0)
# print(d)
# d = np.concatenate((a, b.T), axis=1)
# print(d)
# e = np.concatenate((a, b), axis=None)
# print(e)

# a = np.array([1, 2, 3, 4])
# b = np.array([5, 6, 7, 8])
# # hstack
# c = np.hstack((a, b))
# print(c)
# # vstack
# d = np.vstack((a, b))
# print(d)

# # broadcasting
# x = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
# a = np.array([1, 0, 1])
# y = x + a
# print(x)
# print(a)
# print(y)

# a = np.array([[7, 8, 9, 10, 11, 12, 13], [17, 18, 19, 20, 21, 22, 23]])
# print(a)
# print(a.sum())
# print(a.sum(axis=None))
# print(a.sum(axis=0))
# print(a.sum(axis=1))
# b = a.sum(axis=0)
# print(b)
# print(b.shape)
# print(a.mean(axis=None))
# print(a.mean(axis=0))
# print(a.mean(axis=1))
# print(a.var(axis=None))
# print(a.std(axis=None))
# print(np.std(a, axis=None))
# print(np.max(a, axis=None))

# x = np.array([1, 2])
# print(x)
# print(x.dtype)
# x = np.array([1, 2], dtype=np.int64)
# print(x)
# print(x.dtype)
# x = np.array([1.0, 2.0], dtype=np.float32)
# print(x)
# print(x.dtype)

# a = np.array([1, 2, 3])
# b = a
# b[0] = 42
# print(a)
# print(b)
# a = np.array([1, 2, 3])
# b = a.copy

# a = np.zeros((2, 3))
# print(a)
# a = np.ones((2, 3))
# print(a)
# print(a.dtype)
# a = np.full((2, 3), 5.0)
# print(a)
# a = np.eye(3)
# print(a)
# a = np.arange(20)
# print(a)
# a = np.linspace(0, 10, 5)
# print(a)

# a = np.random.random((3, 2))    # uniform distribution, between 0 and 1
# print(a)
# a = np.random.randn(3, 2)       # normal (or Gaussian) distribution, mean = 0 & var = 1
# print(a)
# a = np.random.randn(1000)
# print(a.mean(), a.var())
# a = np.random.randint(10, size=(3, 3))      # Lower bound = 0, Upper bound = 10
# print(a)
# a = np.random.randint(3, 10, size=(3, 3))   # Lower bound = 3, Upper bound = 10
# print(a)
# a = np.random.choice(5, size=10) # 10 random integers between 0 and 5, 
# print(a)
# a = np.random.choice([-8, -7, -6], size=10)
# print(a)

# a = np.array([[1, 2], [3, 4]])
# eigenvalues, eigenvectors = np.linalg.eig(a)
# print(eigenvalues)
# print(eigenvectors) # column vectors!!!
# # e_vec * e_val = A * e_vec
# b = eigenvectors[:, 0] * eigenvalues[0]
# print(b)
# c = a @ eigenvectors[:, 0]
# print(c)
# print(b == c)
# print(np.allclose(b, c))

# Ax = b <=> x = A^-1b
A = np.array([[1, 1], [1.5, 4.0]])
b = np.array([2200, 5050])
x = np.linalg.inv(A).dot(b)
print(x)
x = np.linalg.solve(A, b)
print(x)

print(A)
print(A.T)

# # np.loadtxt, np.getfromtxt
# data = np.loadtxt('spambase.csv', delimiter=',', dtype=np.float32)
# print(data.shape)
