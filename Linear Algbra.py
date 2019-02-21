"""
Name: YANYAO CAO
School:UoM
Date: Jan 25, 2019
"""

import gc
import numpy as np
import logging
import traceback


class MyException(Exception):
    def __init__(self):
        pass


class Distance:
    temp_cls = 0

    def __init__(self):
        self.temp = 0

    def euc_distance(self, vec):
        self.temp = 0
        for i in range(0, len(vec)):
            self.temp += vec[i] ** 2
        return np.sqrt(self.temp)

    def man_distance(self, vec):
        self.temp = 0
        for i in range(0, len(vec)):
            self.temp += vec[i]
        return self.temp

    @ classmethod
    def sta_ecu_distance(cls, vec):
        cls.temp_cls = 0
        for i in range(0, len(vec)):
            cls.temp_cls += vec[i] ** 2
        return np.sqrt(cls.temp_cls)


class Projection:

    def __init__(self):
        self.temp = 0

    def along(self, vec, vec2):
        self.temp = 0
        multidot = 0
        if len(vec) is not len(vec2):
            raise MyException
        else:
            for i in range(0, len(vec)):
                multidot += vec[i] * vec2[i]
            return multidot/Distance.sta_ecu_distance(vec2)


u = np.matrix([[2], [3], [1]])
v = np.matrix([[5], [1], [3]])
w = np.matrix([[4], [4]])
L = np.matrix([[2, 0, 4], [5, 3, 0], [1, 1, 1]])
M = np.matrix([[4, 1], [4, 2], [1, 3]])
R = np.matrix([[1, 0.5], [0.5, 1]])

# Q1:
# print(np.dot(v.transpose(), w))

# Q2:
print(np.dot(v.transpose(), u))

# Q3:
print(np.dot(M, w))

# Q4:
print(np.multiply(u, v))

# Q6:
print(np.dot(u, v.transpose()))

# Q7:
print(np.dot(u, w.transpose()))

# Q8:
print(np.dot(v.transpose(), np.dot(L, u)))

# Q9:
print(np.dot((u-v).transpose(), M))

e_val, e_vec = np.linalg.eig(R)
print(e_val)
print(e_vec)

obj1 = Distance()
obj2 = Projection()
print(obj1.euc_distance(u) + obj1.man_distance(u))

print(np.multiply(obj1.man_distance(u), M))

new = np.matrix([[1], [0]])
try:
    # print(obj2.along(u, v))
    # print(obj2.along(v, u))
    temp_result = obj2.along(w, new)
except MyException:
    traceback.print_exc()
    logging.warning("Exec failed, failed msg:" + traceback.format_exc())
else:
    print("Successfully Calculated! %s" % (str(temp_result)))
finally:
    gc.collect()

