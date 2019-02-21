
import pandas as pd

data = pd.read_csv("D:\\Data\\shares\\ab_flow.csv")
data2 = pd.read_csv("D:\\Data\\shares\\a_indicator.csv")
print(data)
print(data2)

# # Calculate EMA 12
# temp_total = 0
# for i in range(87, 99):
#     temp_total += data.iat[i, 1]
#
# ave_12 = temp_total/12
# print(ave_12)
#
# for i in range(86, -1, -1):
#     ave_12 = data.iat[i, 1]*(2/(12+1)) + ave_12*(1-(2/(12+1)))
#     print(str(data.iat[i, 0])+","+str(ave_12))

# # Calculate EMA 26
# temp_total = 0
# for i in range(73, 99):
#     temp_total += data.iat[i, 1]
#
# ave_26 = temp_total/26
# print(ave_26)
#
# print(data.iat[72, 0])
# for i in range(72, -1, -1):
#     ave_26 = data.iat[i, 1]*(2/(26+1)) + ave_26*(1-(2/(26+1)))
#     print(str(ave_26))

# reverse index
# for i in range(86, -1, -1):
#     print(data.iat[i, 1])

#
# calculate DEA
# temp_total = 0
# for i in range(0, 9):
#     temp_total += data.iat[i, 5]
# DEA = temp_total/9
# print(DEA)
#
# for i in range(9, 73):
#     DEA = data.iat[i, 5]*0.2 + DEA*0.8
#     print(DEA)

# # Calculate MACD
# for i in range(0, 65):
#     MACD = (data.iat[i, 5] - data.iat[i, 6])*2
#     print(MACD)

# Update
newAve_12 = data.iat[0, 1]*(2/(12+1)) + data2.iat[data2.shape[0]-1, 3]*(1-(2/(12+1)))
newAve_26 = data.iat[0, 1]*(2/(26+1)) + data2.iat[data2.shape[0]-1, 4]*(1-(2/(26+1)))
newDIFF = newAve_12 - newAve_26
newDEA = newDIFF*0.2 + data2.iat[data2.shape[0]-1, 6]*0.8
newMACD = 2*(newDIFF - newDEA)
print(str(data.iat[0, 0])+","+str(data.iat[0, 1])+","+str(newAve_12)+","+str(newAve_26)+","+str(newDIFF)+","+str(newDEA)+","+str(newMACD))



