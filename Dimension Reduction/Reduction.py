import numpy as np
from mylibs import ReadFile


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
            for i3 in range(0, 1985):
                m[i2][i3] = float(float(m[i2][i3]) - float(self.means[i2]))
            self.centeredM = m.astype(float)

    def getCovMatrix(self):
        self.cov = np.cov(self.centeredM)

    def getElg(self):
        self.eigVal, self.eigVec = np.linalg.eig(self.cov)
        print(self.eigVal)
        print(self.eigVec)


    def decrease(self, m):
        cut = self.eigVec[0:2]
        return np.dot(m.reshape(1985, 8), cut.reshape(8, 2))

    def getSingular(self, m):
        self.U, self.M, self.V = np.linalg.svd(m)

    def decrease2(self, m):
        return np.dot(np.transpose(m), np.transpose(self.U[0: 2]))


if __name__ == '__main__':
    i = ReadFile.ReadFile()
    i.openFile()
    i.getSomeCols()
    ii = Reduction()
    # ii.getSingular(i.matrix)
    ii.centralization(i.matrix)
    ii.getCovMatrix()
    ii.getElg()
    new = ii.decrease(i.matrix)
    print(np.cov(new))

