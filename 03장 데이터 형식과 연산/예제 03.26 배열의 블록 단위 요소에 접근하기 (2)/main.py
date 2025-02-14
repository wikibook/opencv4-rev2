import numpy as np

array = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20]])

print(array[1:3])
print(array[::2])
print(array[1:3, 2:4])
print(array[..., 3, 1])