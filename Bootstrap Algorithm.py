"""
Name: YANYAO CAO
ID: 10329521
DATE: OCT 1, 2018
"""
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as pl

global random_number  # The global random value
global seed  # The random seed value
M = 50
Percentage = 0.025
random_number = np.random.rand()*0.98+0.01
seed = 3.8 + np.random.rand()*0.099


# Find age refer to index
def findAge(index):
    counter = 0
    for current in dict_original.keys():
        counter += int(dict_original[current])
        if counter >= index:
            return current
    print("Out of boundary Error")
    return -1


# define a function for random
def my_random(maxValue):
    global random_number
    global seed
    # random_number = seed * random_number * (1 - random_number)
    random_number = np.random.rand()
    return int(random_number * maxValue+1)


# Open file for reading
fileName = "D:/data/Data_age_unit.csv"
data = pd.read_csv(fileName)
count_column = data.shape[1]
data = pd.read_csv(fileName, usecols=range(5, count_column))
print("Read file Successfully")

# Build dictionary to store data
dict_original = {}
for i in range(0, count_column-6):
    data_age = data.iat[0, i]
    temp_amount = data.iat[1, i]
    temp_age = re.search(r'Age : Age ([\d]+)', str(data_age))
    try:
        temp_age.group(1)
    except AttributeError:
        print("Error! At "+str(i))
    else:
        if temp_age is not None:
            dict_original[temp_age.group(1)] = temp_amount
print("Created the dictionary successfully")
print("The dictionary contains: ", len(dict_original.keys()), "Ages")
print(dict_original)

# define a count dic
dict_count = {}
for item in dict_original.keys():
    dict_count[item] = []

# Calculate total amount
total = 0
for i in range(0, 101, 1):
    total += int(dict_original[str(i)])
print(total)

# Processing(Re-sampling)
print("Starting re-sampling")
for m in range(0, M):
    for i in range(0, 101):
        dict_count[str(i)].append(0)
    for i in range(0, total):
        temp_age = findAge(my_random(total))
        dict_count[temp_age][m] += 1
    print(str(m+1)+" cycles Finished")
print("Re-sampling finished")


# # Plot all lines
# for m in range(0, M-1):
#     temp_arr = []
#     temp_arr2 = []
#     for i in range(0, 101, 1):
#         temp_arr.append(dict_count[str(i)][m])
#         temp_arr2.append(i)
#     pds1 = pd.Series(temp_arr2)
#     pds2 = pd.Series(temp_arr)
#     pl.plot(pds1, pds2)
# pl.show()

# # Plot all lines++
# for m in range(0, len(dict_count['0'])):
#     temp_arr = []
#     temp_arr2 = []
#     for i in range(0, 101, 1):
#         temp_arr.append(dict_count[str(i)][m])
#         temp_arr2.append(i)
#     pds1 = pd.Series(temp_arr2)
#     pds2 = pd.Series(temp_arr)
#     pl.plot(pds1, pds2)
# pl.show()


# Clean 5% noise
for i in range(0, 101, 1):
    temp_arr3 = []
    for j in range(0, M):
        temp_arr3.append(dict_count[str(i)][j])
    amount_delete = int(Percentage * (M-1))
    for k in range(0, amount_delete):
        temp_index = temp_arr3.index(min(temp_arr3))
        del(temp_arr3[temp_index])
        del(dict_count[str(i)][temp_index])
    for k in range(0, amount_delete):
        temp_index = temp_arr3.index(max(temp_arr3))
        del(temp_arr3[temp_index])
        del(dict_count[str(i)][temp_index])

# Plot MAX and MIN
temp_arr = []
temp_arr2 = []
temp_arr4 = []
for i in range(0, 101, 1):
    temp_arr3 = []
    for j in range(0, M-2*amount_delete):
        temp_arr3.append(dict_count[str(i)][j])
    temp_arr.append(dict_count[str(i)][temp_arr3.index(min(temp_arr3))])
    temp_arr4.append(dict_count[str(i)][temp_arr3.index(max(temp_arr3))])
    temp_arr2.append(i)
min = temp_arr
max = temp_arr4
pds1 = pd.Series(temp_arr2)
pds2 = pd.Series(temp_arr)
pds3 = pd.Series(temp_arr4)
pl.plot(pds1, pds2)
pl.plot(pds1, pds3)

# # draw original hist
# temp_arr5 = []
# for i in range(1, 101, 1):
#     for j in range(0, int(dict_original[str(i)])):
#         temp_arr5.append(i)
# pds4 = pd.Series(temp_arr5)
# pl.hist(pds4, bins=99, normed=False)

# find significant wiggles
for i in range(1, 100, 1):
    if min[i-1] < max[i] and max[i-1] < min[i] and max[i+1] < min[i] and max[i+1] < min[i]:
        print("↑up-down↓ significant wiggle is at "+str(i))
    elif min[i - 1] > max[i] and max[i - 1] > min[i] and min[i + 1] > max[i] and max[i + 1] > min[i]:
        print("↓down-up↑ significant wiggle is at "+str(i))

# draw original line
temp_arr5 = []
for i in range(0, 101, 1):
    temp_arr5.append(int(dict_original[str(i)]))
pds4 = pd.Series(temp_arr5)
pl.plot(pds1, pds4)

pl.show()


