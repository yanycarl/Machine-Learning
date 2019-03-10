
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CalculateThreshold:
    __Q1Array = []
    __Q3Array = []
    floorArray = []
    ceilArray = []

    @classmethod
    def calculateThreshold(cls, m):
        for row in m:
            cls.__Q1Array.append(np.percentile(row, 25))
            cls.__Q3Array.append(np.percentile(row, 75))

        for i in range(0, len(cls.__Q1Array)):
            cls.floorArray.append(cls.__Q1Array[i] - 1.5 * (cls.__Q3Array[i] - cls.__Q1Array[i]))
            cls.ceilArray.append(cls.__Q3Array[i] + 1.5 * (cls.__Q3Array[i] - cls.__Q1Array[i]))

        print(cls.ceilArray)


class ReadFile:
    matrix = []

    @classmethod
    def openFile(cls):
        with open('D:\\UDisk\\Machine Learning\\Class\\Week 4\\Data.txt', newline='\n') as csvFile:
            read = csv.reader(csvFile, delimiter=' ', quotechar='|')
            for row in read:
                temp = []
                for item in row:
                    if item == 'AB':
                        temp.append(1.0)
                    elif item == 'NO':
                        temp.append(0.0)
                    else:
                        temp.append(item)
                cls.matrix.append(temp)
        cls.matrix = np.mat(cls.matrix).A
        cls.matrix = np.transpose(cls.matrix)
        cls.matrix = cls.matrix.astype(float)

    @classmethod
    def getData(cls):
        cls.openFile()


class ReadFile2:
    matrix = []

    @classmethod
    def openFile(cls):
        with open('C:\\Users\\YanyCarl\\Documents\\Anaconda Project\\Data3.csv', newline='\n') as csvFile:
            read = csv.reader(csvFile)
            for row in read:
                temp = []
                for item in row:
                    temp.append(item)
                cls.matrix.append(temp)
        cls.matrix = np.matrix(cls.matrix).A
        cls.matrix = cls.matrix.astype(float)
        cls.matrix = np.transpose(cls.matrix)

    @classmethod
    def getData(cls):
        cls.openFile()

class EDA:

    @classmethod
    def drawHistogram(cls, m):
        for i2 in range(0, 6):
            plt.subplot(2, 3, i2 + 1)
            plt.hist(m[i2], bins=50)
        plt.show()

    @classmethod
    def drawScatter(cls, m):
        for i2 in range(0, 6):
            for i in range(0, 6):
                plt.subplot(6, 6, i2 * 6 + i + 1)
                sns.scatterplot(m[i2], m[i])
        plt.show()

    @classmethod
    def drawScatter2(cls, m, m2):
        for i in range(0, 5):
            plt.subplot(2, 3, i + 1)
            sns.scatterplot(m[i], m2)
        plt.subplot(2, 3, 6)
        sns.scatterplot(m2, m2)
        plt.show()

    @classmethod
    def drawKDE(cls, m):
        for i2 in range(0, 6):
            for i in range(0, 6):
                plt.subplot(6, 6, i2 * 6 + i + 1)
                sns.kdeplot(m[i2], m[i])
        plt.show()

    @classmethod
    def drawDensity(cls, m):
        for i2 in range(0, 6):
            plt.subplot(2, 3, i2+1)
            sns.distplot(m[i2])
        plt.show()

    @classmethod
    def drawBox(cls, m):
        for i2 in range(0, 6):
            plt.subplot(2, 3, i2+1)
            sns.boxplot(m[i2])
        plt.show()

    @classmethod
    def drawLog(cls, m):
        sns.distplot(m, bins=50)
        plt.show()

class PreProcessing:

    @classmethod
    def logProcess(cls):
        logged = []
        for item in ReadFile.matrix[5]:
            if True:
                try:
                    logged.append(math.log(item + 12, 10.0))
                except RuntimeError:
                    print("Wrong")

        for i in range(0, len(logged)):
            ReadFile.matrix[5][i] = logged[i]


class CleanData:
    dirtyIndex = set()
    cleanIndex = []

    @classmethod
    def findIndex(cls, c, f, m):
        for i in range(0, len(m)):
            for j in range(0, len(m[0])):
                if j not in cls.dirtyIndex and (c[i] < m[i, j] or f[i] > m[i, j]):
                    cls.dirtyIndex.add(j)
        print("The problem rows: ")
        print(str(cls.dirtyIndex))
        print("Deleted: " + str(cls.dirtyIndex.__len__()))

        for i in range(1, len(m[0])):
            if i not in cls.dirtyIndex:
                cls.cleanIndex.append(i-1)
        print("Remaining: " + str(len(cls.cleanIndex)))


class RecreateFile:
    @staticmethod
    def writeFile():
        df = pd.read_csv("C:\\Users\\YanyCarl\\Documents\\Anaconda Project\\Data2.csv")
        df = pd.DataFrame(df.iloc[list(CleanData.cleanIndex), :])
        df.to_csv("C:\\Users\\YanyCarl\\Documents\\Anaconda Project\\Data4.csv")


class Reduction:

    def __init__(self):
        self.means = []
        self.cov = 0.0
        self.eigVal = None
        self.eigVec = None
        self.centeredM = None
        self.U = None
        self.V = None
        self.M = None

    @staticmethod
    def calcMean(array):
        sum2 = 0.0
        for item in array:
            sum2 += float(item)
        return sum2/len(array)

    def centralization(self, m):
        self.means = []
        for i2 in range(0, len(m)):
            self.means.append(Reduction.calcMean(m[i2]))

        for i2 in range(0, len(m)):
            for i3 in range(0, 282):
                m[i2][i3] = float(float(m[i2][i3]) - float(self.means[i2]))
            self.centeredM = m.astype(float)

    def getCovMatrix(self):
        self.cov = np.cov(self.centeredM)

    def getElg(self):
        self.eigVal, self.eigVec = np.linalg.eig(self.cov)

    def decrease(self, m):
        cut = self.eigVec[0:2]
        return np.dot(m.reshape(282, 6), cut.reshape(6, 2))


if __name__ == '__main__':
    ReadFile.getData()
    EDA.drawDensity(ReadFile.matrix)
    EDA.drawBox(ReadFile.matrix)
    EDA.drawScatter(ReadFile.matrix)
    PreProcessing.logProcess()
    EDA.drawLog(ReadFile.matrix[5])
    CalculateThreshold.calculateThreshold(ReadFile.matrix)
    CleanData.findIndex(CalculateThreshold.ceilArray, CalculateThreshold.floorArray, ReadFile.matrix)
    RecreateFile.writeFile()
    ReadFile2.getData()
    EDA.drawDensity(ReadFile2.matrix)
    EDA.drawBox(ReadFile2.matrix)
    EDA.drawScatter(ReadFile2.matrix)
    instance1 = Reduction()
    instance1.centralization(ReadFile2.matrix[0:6])
    instance1.getCovMatrix()
    instance1.getElg()
    new = instance1.decrease(ReadFile2.matrix[0:6])
    print(instance1.eigVal)

